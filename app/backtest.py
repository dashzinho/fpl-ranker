"""Backtest: how good are the projections, and which settings would have worked best?

For every already-played gameweek g the model is re-run using ONLY data from before g: team ratings fitted on matches
before g, player histories cut at g-1, and the season-performance inputs recomputed from those same cut histories
(nothing from the full season leaks in). Injury flags are neutral because past news is not stored. The real
`perf_scores` and the page's own Score formula are used.

Two outputs:
  * data/backtest.json   accuracy of the single-gameweek projections by position, against two naive baselines, the
                         comparison of the team model's fixture difficulty with FPL's official rating, and the clean-sheet
                         calibration (python fpl.py --backtest)
  * data/backtest_sets.* ONE table of weight sets. A set = the four team-model weights (xG share, half-life, prior
                         strength, prior scale) + the Score weights (outlook / perf / value in steps of 5). Every
                         combination is scored on four measures, each refitted per gameweek on earlier data only:
                           pts  realised points over the next 3 GWs of the top picks per position (GW4 onward)
                           gd   rank correlation of the model's fixture ease with the real goal difference
                           xgd  the same with the xG difference
                           ll   Poisson log-likelihood of the goals each team scored
                         plus an overall score (mean percentile across the four). Each measure has a paired standard
                         error versus its best set (overlap-aware for Points), so near-ties show as ties, not a crown.
                         It also holds the walk-forward test of auto-pick (choose on earlier gameweeks, score on the next)
                         and `best_team`, the team-model-only ranking that auto-pick uses (never the Score weights).

Honest limits: one season, few gameweeks, prices are
today's, and picking the best of thousands of combinations on this data flatters the winner. Re-run as the season progresses.
Run:  python fpl.py --backtest      (the page also refreshes it automatically when a gameweek finishes)
"""
import copy, hashlib, itertools, json, math, os, time, warnings
import numpy as np
from scipy.stats import rankdata, spearmanr
import fpl_data as fd
import fpl_model as fm
from paths import DATA

FILE = os.path.join(DATA, "backtest.json")
SETS_META = os.path.join(DATA, "backtest_sets.json")
SETS_NPZ = os.path.join(DATA, "backtest_sets.npz")
TOPN = {1: 3, 2: 8, 3: 8, 4: 5}
FIRST_GW = 4
MIX = [(wo, wp, 100 - wo - wp) for wo in range(0, 101, 5) for wp in range(0, 101 - wo, 5)]


def _pct(vals, mask):
    """Same as the page's pctRank: percentile among eligible players, 0 for the rest."""
    a = np.asarray(vals, float); m = np.asarray(mask, bool)
    out = np.zeros(len(a))
    el = np.sort(a[m]); n = len(el)
    if n < 2:
        out[m] = 50.0
        return out
    out[m] = 100.0 * np.searchsorted(el, a[m], side="left") / (n - 1)
    return out


def _cut_stats(p, rows):
    """Season-to-date production recomputed from the cut history (the real model reads full-season boot values)."""
    mins = sum(r["minutes"] for r in rows)
    p["mins"] = mins
    p["pts"] = sum(r["total_points"] for r in rows)
    n90 = mins / 90.0
    p["p90"] = p["pts"] / n90 if n90 else 0.0
    p["xgi90"] = sum(float(r.get("expected_goal_involvements") or 0) for r in rows) / n90 if n90 else 0.0
    p["bn90"] = sum(r.get("bonus") or 0 for r in rows) / n90 if n90 else 0.0

def _fdr_hist():
    try:
        return json.load(open(os.path.join(DATA, "fdr_history.json"), encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


def _fdr_records(fixtures, xg, model, g):
    """One record per (finished fixture in GW g, team): the model's pre-GW expectation, FPL's official difficulty
    and what happened. Official FDR is today's value (FPL rarely changes it mid-season), the model is as-of before g."""
    out = []
    hist_fdr = _fdr_hist()
    for f in fixtures:
        if f["event"] != g or not f["finished"] or f["team_h_score"] is None:
            continue
        dh, da = fd.fdr_as_of(hist_fdr, f, g)
        for t, o, home, gf, ga, d in ((f["team_h"], f["team_a"], True, f["team_h_score"], f["team_a_score"], dh),
                                      (f["team_a"], f["team_h"], False, f["team_a_score"], f["team_h_score"], da)):
            lt, lo = model["lam"](t, o, home), model["lam"](o, t, not home)
            xf, xa = xg.get((f["id"], t)), xg.get((f["id"], o))
            out.append({"gw": g, "off": -float(d), "mod": float(np.log(lt / lo)), "lt": lt, "lo": lo, "gd": float(gf - ga),
                        "gf": float(gf), "ga": float(ga), "xgd": (xf - xa) if xf is not None and xa is not None else float(gf - ga),
                        "d": int(d)})
    return out


def _fdr_run(boot, fixtures, xg_all, prefs, last):
    """Fixture-difficulty benchmark over every finished GW from GW1 (only the cheap team fit is needed, so it
    does not share the player backtest's FIRST_GW). GW1 has no results: the model is purely FPL's prior there."""
    recs = []
    for g in range(1, last + 1):
        fx = [dict(f, finished=False if (f["event"] or 0) >= g else f["finished"]) for f in fixtures]
        model = fm.fit_teams(boot, fx, xg_all, g - 1, prefs)
        recs += _fdr_records(fixtures, xg_all, model, g)
    return _fdr_summary(recs, list(range(1, last + 1)))


def _fdr_summary(recs, gws):
    """Does the model's difficulty rank fixtures better than FPL's official 1-5 rating? Ease = higher is easier for the team."""
    if len(recs) < 60:
        return None
    sp = lambda r, k, y: float(spearmanr([x[k] for x in r], [x[y] for x in r])[0])      # noqa: E731
    out = {"n": len(recs), "targets": {}}
    for label, ek, ok, y in (("Goal difference", "off", "mod", "gd"), ("xG difference", "off", "mod", "xgd")):
        per = []
        for g in gws:
            r = [x for x in recs if x["gw"] == g]
            if len(r) > 10:
                per.append((sp(r, ek, y), sp(r, ok, y)))
        po = np.array(per)
        diff = po[:, 1] - po[:, 0]
        n = len(diff)
        out["targets"][label] = {"official": round(sp(recs, ek, y), 3), "model": round(sp(recs, ok, y), 3),
                                 "gw_official": round(float(po[:, 0].mean()), 3), "gw_model": round(float(po[:, 1].mean()), 3),
                                 "gw_diff": round(float(diff.mean()), 3), "gw_se": round(float(diff.std(ddof=1) / np.sqrt(n)), 3) if n > 2 else 0.0,
                                 "gw_n": n, "gw_model_wins": int((diff > 0).sum())}
    # split views: attack = own xG-for rate vs goals scored, defence = opponent's rate vs goals conceded
    out["attack_rho"] = round(sp(recs, "lt", "gf"), 3)
    out["defence_rho"] = round(sp(recs, "lo", "ga"), 3)
    out["official_gf_rho"] = round(sp(recs, "off", "gf"), 3)
    out["official_ga_rho"] = round(-sp(recs, "off", "ga"), 3)
    # buckets: official 1-5 vs model quantile 1-5 -> mean goal difference per bucket
    order = np.array([x["mod"] for x in recs]).argsort().argsort() / max(1, len(recs) - 1)
    mb = [int(5 - min(4, int(p * 5))) for p in order]                                    # 1 = easiest, like FPL
    tab = []
    for k in range(1, 6):
        o = [x["gd"] for x in recs if x["d"] == k]; m = [x["gd"] for x, b in zip(recs, mb) if b == k]
        tab.append({"d": k, "n_off": len(o), "gd_off": round(float(np.mean(o)), 2) if o else None,
                    "n_mod": len(m), "gd_mod": round(float(np.mean(m)), 2) if m else None})
    out["buckets"] = tab
    return out


# ---------------------------------------------------------------- weight sets (team model + score weights)
TM_KEYS = ("xg_weight", "half_life_gw", "prior_strength", "prior_scale")
# The team-model sets that are backtested: the named presets first, then a few around the region that scored best. The
# review of the first 144-set grid showed the team model barely changes which players are picked (87% of the variation in
# Points came from the Score mix, 7% from the team model), so a short list gives the same answer in seconds.
TM_SETS = [(0.65, 12, 5, 0.10), (0.65, 12, 20, 0.20), (0.65, 12, 1, 0.10), (0.65, 4, 5, 0.10), (0.65, 100, 5, 0.10),
           (1.0, 12, 5, 0.10), (0.0, 12, 5, 0.10),
           (0.65, 12, 10, 0.20), (1.0, 24, 20, 0.20), (1.0, 4, 10, 0.20), (0.65, 24, 10, 0.20), (0.65, 4, 10, 0.20),
           (1.0, 12, 20, 0.20), (1.0, 100, 20, 0.20), (0.65, 12, 10, 0.10), (0.65, 12, 20, 0.10), (0.0, 12, 20, 0.20)]
TM_GRID = {k: sorted({t[i] for t in TM_SETS}) for i, k in enumerate(TM_KEYS)}      # the values used, for display
WF_START = 6                                          # first gameweek of the walk-forward test (needs a few gameweeks of history)
TM_DEFAULT = {"xg_weight": 0.65, "half_life_gw": 12, "prior_strength": 5.0, "prior_scale": 0.10}
W_DEFAULT = (60, 25, 15)
SET_KEYS = TM_KEYS + ("wo", "wp", "wv")
MEAS = ("pts", "gd", "xgd", "ll")
VIEWS = ("ov",) + MEAS
PRESETS = [
    ("Default", (0.65, 12, 5, 0.10, 60, 25, 15), "The settings the page shipped with: 65% xG / 35% goals, half-life 12, a moderate pull toward FPL's ratings, Score mix 60/25/15."),
    ("Trust FPL ratings", (0.65, 12, 20, 0.20, 60, 25, 15), "Stays close to FPL's own strength ratings and moves slowly as results come in."),
    ("Results first", (0.65, 12, 1, 0.10, 60, 25, 15), "Almost ignores FPL's ratings once a few matches are played."),
    ("Recent form", (0.65, 4, 5, 0.10, 60, 25, 15), "The last few gameweeks count far more than earlier ones (half-life 4)."),
    ("Season-long", (0.65, 100, 5, 0.10, 60, 25, 15), "Every match counts about the same all season."),
    ("xG only", (1.0, 12, 5, 0.10, 60, 25, 15), "Fits expected goals only, ignoring finishing luck."),
    ("Goals only", (0.0, 12, 5, 0.10, 60, 25, 15), "Fits actual goals only."),
    ("Outlook only", (0.65, 12, 5, 0.10, 100, 0, 0), "Default team model; the Score is purely projected points."),
    ("Value tilt", (0.65, 12, 5, 0.10, 30, 25, 45), "Default team model; the Score leans toward points per pound."),
]


def _sets_key(last, prefs):
    code = hashlib.md5(open(fm.__file__, "rb").read() + open(__file__, "rb").read()).hexdigest()[:8]
    return f"{last}|{prefs.get('perf_min_minutes', 270)}|{code}"


def _rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return 0.0
    return float(spearmanr(a, b)[0])


def _se(diff, axis):
    """Standard error of the mean paired difference across gameweeks (nan-aware)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = np.sum(~np.isnan(diff), axis=axis)
        sd = np.nanstd(diff, axis=axis, ddof=1)
        return np.where(n > 2, sd / np.sqrt(np.maximum(n, 1)), 0.0)


def _se_hac(diff, lag=2):
    """Noise of the mean paired difference when neighbouring gameweeks overlap (3-GW windows share 2 gameweeks):
    Newey-West with lag 2, never smaller than the plain standard error. diff has gameweeks on the last axis."""
    n = diff.shape[-1]
    d = diff - diff.mean(axis=-1, keepdims=True)
    g0 = (d * d).sum(-1) / n
    v = g0.copy()
    for l in range(1, lag + 1):
        v = v + 2 * (1 - l / (lag + 1)) * (d[..., l:] * d[..., :-l]).sum(-1) / n
    plain = np.sqrt(g0 * n / max(n - 1, 1) / n)
    return np.maximum(plain, np.sqrt(np.maximum(v, 0) / n))


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(a, axis=axis)


def compute_sets(boot, fixtures, hist, prefs, log=print):
    last = max((e["id"] for e in boot["events"] if e["finished"]), default=0)
    if last < FIRST_GW + 1:
        return None
    t0 = time.time()
    xg_all = fm.team_match_table(boot, fixtures, hist)
    actual = {pid: {r["round"]: r["total_points"] for r in rows} for pid, rows in hist.items()}
    price = {e["id"]: e["now_cost"] / 10.0 for e in boot["elements"]}
    sets = list(TM_SETS)
    nS, nM = len(sets), len(MIX)
    gws = list(range(1, last + 1))
    pgws = [g for g in gws if g >= FIRST_GW]
    Mx = np.array(MIX, float)
    T = {k: np.full((nS, len(gws)), np.nan) for k in ("gd", "xgd", "ll")}
    P = np.full((nS, nM, len(pgws)), np.nan)
    off = {"gd": np.full(len(gws), np.nan), "xgd": np.full(len(gws), np.nan)}
    lg = np.vectorize(math.lgamma)
    n_obs = 0
    fdr_hist = _fdr_hist()
    log(f"  backtesting {nS * nM:,} weight sets ({nS} team-model sets x {nM} score mixes) over GW1-GW{last}...")
    for gi, g in enumerate(gws):
        rows = []                                      # (team, opp, home, gf, ga, xg diff, official difficulty)
        for f in fixtures:
            if f["event"] != g or not f["finished"] or f["team_h_score"] is None:
                continue
            xh, xa = xg_all.get((f["id"], f["team_h"])), xg_all.get((f["id"], f["team_a"]))
            ok = xh is not None and xa is not None
            dh, da = fd.fdr_as_of(fdr_hist, f, g)
            for t, o, home, gf, ga, d in ((f["team_h"], f["team_a"], True, f["team_h_score"], f["team_a_score"], dh),
                                          (f["team_a"], f["team_h"], False, f["team_a_score"], f["team_h_score"], da)):
                x = (xh - xa if home else xa - xh) if ok else float(gf - ga)
                rows.append((t, o, home, float(gf), float(ga), x, float(d)))
        have_team = len(rows) >= 10
        n_obs += len(rows)
        if have_team:
            gf_ = np.array([r[3] for r in rows]); gd_ = gf_ - np.array([r[4] for r in rows]); xd_ = np.array([r[5] for r in rows])
            ez = -np.array([r[6] for r in rows])
            off["gd"][gi] = _rho(ez, gd_); off["xgd"][gi] = _rho(ez, xd_)
        fx = [dict(f, finished=False if (f["event"] or 0) >= g else f["finished"]) for f in fixtures]
        players_ok = g >= FIRST_GW
        if players_ok:
            b = copy.deepcopy(boot)
            for ev in b["events"]:
                ev["is_current"] = ev["id"] == g - 1; ev["is_next"] = ev["id"] == g
            for e in b["elements"]:
                e["status"] = "a"; e["chance_of_playing_next_round"] = None; e["news"] = ""
            h = {pid: [r for r in rr if r["round"] < g] for pid, rr in hist.items()}
            h = {pid: rr for pid, rr in h.items() if rr}
            horizon = [x for x in range(g, g + 3) if x <= last]
            pi = pgws.index(g)
        else:
            b = boot
        for si, vals in enumerate(sets):
            pp = dict(prefs, **dict(zip(TM_KEYS, vals)))
            model = fm.fit_teams(b, fx, xg_all, g - 1, pp)
            if have_team:
                lt = np.array([model["lam"](r[0], r[1], r[2]) for r in rows]); lo = np.array([model["lam"](r[1], r[0], not r[2]) for r in rows])
                ease = np.log(lt / lo)
                T["gd"][si, gi] = _rho(ease, gd_); T["xgd"][si, gi] = _rho(ease, xd_)
                T["ll"][si, gi] = float(np.mean(gf_ * np.log(lt) - lt - lg(gf_ + 1)))
            if not players_ok:
                continue
            players, _, _ = fm.project(b, fx, h, model, pp, horizon_gws=3)
            for q in players:
                _cut_stats(q, h[q["id"]])
            players = fm.perf_scores(players, prefs.get("perf_min_minutes", 270))
            for q in players:
                q["xp3"] = sum(f[3] for f in q["fx"] if f[0] in horizon)
                q["y3"] = sum(actual[q["id"]].get(x, 0.0) for x in horizon)
                q["price_now"] = price[q["id"]]
                q["reg"] = q["pm1"] >= 0.25 and len(q["fx"]) > 0
            per_pos = []
            for pt, k in TOPN.items():
                grp = [q for q in players if q["p"] == pt]
                if len(grp) < k * 3:
                    continue
                mask = [q["reg"] for q in grp]; mk = np.array(mask)
                po = _pct([q["xp3"] for q in grp], mask)
                pv = _pct([q["xp3"] / q["price_now"] for q in grp], mask)
                pf = np.array([q["perf"] for q in grp], float)
                y3 = np.array([q["y3"] for q in grp])
                sc = np.where(mk[None, :], (Mx[:, 0:1] * po + Mx[:, 1:2] * pf + Mx[:, 2:3] * pv) / 100.0, 0.0)
                top = np.argsort(-sc, axis=1, kind="stable")[:, :k]
                per_pos.append(y3[top].mean(axis=1))
            if per_pos:
                P[si, :, pi] = np.mean(per_pos, axis=0)
        log(f"    GW{g} done ({time.time() - t0:.0f}s)")

    # ---- averages per measure over all combinations, shape (nS, nM)
    bc = lambda v: np.repeat(v[:, None], nM, axis=1)                                         # noqa: E731
    A = np.stack([_nanmean(P, 2)] + [bc(_nanmean(T[k], 1)) for k in ("gd", "xgd", "ll")])    # (4, nS, nM)
    pct = np.stack([(rankdata(A[k].ravel()) - 1) / max(A[k].size - 1, 1) * 100 for k in range(4)])
    OV = pct.mean(axis=0).reshape(nS, nM)
    # ---- paired standard errors versus each measure's best set
    SE = np.zeros_like(A)
    best = {}
    for k, name in enumerate(MEAS):
        si, mi = np.unravel_index(np.argmax(A[k]), A[k].shape)
        best[name] = (int(si), int(mi))
        if name == "pts":
            SE[k] = _se_hac(P[si, mi][None, None, :] - P)
        else:
            SE[k] = bc(_se(T[name][si][None, :] - T[name], 1))
    GAP = np.stack([A[k].max() - A[k] for k in range(4)])
    TIE = GAP <= 2 * SE + 1e-9
    obest = np.unravel_index(np.argmax(OV), OV.shape)
    best["ov"] = (int(obest[0]), int(obest[1]))
    # ---- odd/even gameweek stability of each ordering
    stab = {}
    for k, name in enumerate(MEAS):
        if name == "pts":
            stab[name] = round(_rho(_nanmean(P[:, :, 0::2], 2).ravel(), _nanmean(P[:, :, 1::2], 2).ravel()), 2)
        else:
            stab[name] = round(_rho(_nanmean(T[name][:, 0::2], 1), _nanmean(T[name][:, 1::2], 1)), 2)
    official = {}
    for name in ("gd", "xgd"):
        si = best[name][0]
        d = T[name][si] - off[name]
        official[name] = {"v": round(float(np.nanmean(off[name])), 4), "gap": round(float(np.nanmean(d)), 4), "se": round(float(_se(d, 0)), 4)}
    x_of = lambda si, mi: [*[float(v) for v in sets[si]], *MIX[mi]]                          # noqa: E731
    # auto-pick adopts only the team model: best mean percentile of the four measures among the team-model sets, with
    # Points averaged over every Score mix (so it does not depend on, or push, the user's Score weights)
    team_score = _pct_rank(P.mean(axis=(1, 2))) + _pct_rank(np.nanmean(T["gd"], 1)) + _pct_rank(np.nanmean(T["xgd"], 1)) + _pct_rank(np.nanmean(T["ll"], 1))
    best_team = [float(v) for v in sets[int(np.argmax(team_score))]]
    wf = _walkforward(T, P, gws, pgws)
    meta = {"key": _sets_key(last, prefs), "finished": last, "gws": [1, last], "pgws": [pgws[0], last], "computed": time.strftime("%d %b %H:%M"),
            "seconds": round(time.time() - t0), "n_sets": nS * nM, "n_team": nS, "n_mix": nM, "n_obs": n_obs, "grid": TM_GRID, "keys": list(SET_KEYS),
            "best": {k: x_of(*v) for k, v in best.items()}, "best_team": best_team, "walkforward": wf, "stability": stab, "official": official,
            "team_sets": [list(t) for t in TM_SETS],
            "n_tied": {name: int(TIE[k].sum()) for k, name in enumerate(MEAS)}}
    return meta, {"A": A.astype(np.float32), "GAP": GAP.astype(np.float32), "SE": SE.astype(np.float32), "TIE": TIE, "OV": OV.astype(np.float32),
                  "P": P.astype(np.float32), "Tgd": T["gd"].astype(np.float32), "Txgd": T["xgd"].astype(np.float32), "Tll": T["ll"].astype(np.float32)}


def _pct_rank(v):
    return (rankdata(v) - 1) / max(len(v) - 1, 1) * 100


def _walkforward(T, P, gws, pgws):
    """Would auto-pick have helped? For each test gameweek g, choose using ONLY earlier gameweeks, then score on g.
    team: the best team-model set by the mean percentile of the four measures (points averaged over all Score mixes),
    compared with just keeping the default set. mix: the best Score mix (by Points, default team model), compared with
    keeping 60/25/15. 'oracle' is the best in hindsight on g, i.e. the ceiling no rule can reach."""
    nS, nM, nP = P.shape
    s0 = TM_SETS.index(tuple(TM_DEFAULT[k] for k in TM_KEYS)); m0 = MIX.index(W_DEFAULT)
    tests = [g for g in gws if g >= WF_START]
    if not tests:
        return None
    team = {"gws": tests, "picked": {k: [] for k in MEAS}, "default": {k: [] for k in MEAS}, "oracle": {k: [] for k in MEAS}, "picks": []}
    mix = {"gws": tests, "picked": [], "default": [], "oracle": [], "picks": []}
    for g in tests:
        gi, pi = g - 1, g - pgws[0]                    # column of gameweek g in the team and Points arrays
        hist_t, hist_p = slice(0, gi), slice(0, pi)    # strictly earlier gameweeks
        meas = {"pts": P[:, :, hist_p].mean(axis=(1, 2)), "gd": T["gd"][:, hist_t].mean(1), "xgd": T["xgd"][:, hist_t].mean(1), "ll": T["ll"][:, hist_t].mean(1)}
        score = sum(_pct_rank(v) for v in meas.values())
        pick = int(np.argmax(score)); team["picks"].append(pick)
        oos = {"pts": P[:, m0, pi], "gd": T["gd"][:, gi], "xgd": T["xgd"][:, gi], "ll": T["ll"][:, gi]}   # Points at the default Score mix
        for k in MEAS:
            team["picked"][k].append(float(oos[k][pick])); team["default"][k].append(float(oos[k][s0])); team["oracle"][k].append(float(oos[k].max()))
        mp = int(np.argmax(P[s0, :, hist_p].mean(axis=1))); mix["picks"].append(list(MIX[mp]))
        mix["picked"].append(float(P[s0, mp, pi])); mix["default"].append(float(P[s0, m0, pi])); mix["oracle"].append(float(P[s0, :, pi].max()))

    def diff(a, b):
        d = np.array(a) - np.array(b)
        return {"mean": round(float(d.mean()), 4), "se": round(float(_se(d, 0)), 4), "wins": int((d > 0).sum()), "n": len(d)}
    out = {"gws": tests, "first": WF_START,
           "team": {"changes": int(sum(1 for a, b in zip(team["picks"], team["picks"][1:]) if a != b)), "picks": [list(TM_SETS[i]) for i in team["picks"]],
                    "picked": {k: round(float(np.mean(v)), 4) for k, v in team["picked"].items()}, "default": {k: round(float(np.mean(v)), 4) for k, v in team["default"].items()},
                    "oracle": {k: round(float(np.mean(v)), 4) for k, v in team["oracle"].items()},
                    "diff": {k: diff(team["picked"][k], team["default"][k]) for k in MEAS}},
           "mix": {"changes": int(sum(1 for a, b in zip(mix["picks"], mix["picks"][1:]) if a != b)), "picks": mix["picks"],
                   "picked": round(float(np.mean(mix["picked"])), 4), "default": round(float(np.mean(mix["default"])), 4), "oracle": round(float(np.mean(mix["oracle"])), 4),
                   "diff": diff(mix["picked"], mix["default"]), "value_weights": [m[2] for m in mix["picks"]]}}
    return out


def sets_get(boot, fixtures, hist, prefs, force=False, log=print):
    """Cached; recomputed when a gameweek finishes or the model code changes (not when you switch sets)."""
    last = max((e["id"] for e in boot["events"] if e["finished"]), default=0)
    try:
        meta = json.load(open(SETS_META, encoding="utf-8"))
        if not force and meta.get("key") == _sets_key(last, prefs) and os.path.exists(SETS_NPZ):
            return meta
    except Exception:                                   # noqa: BLE001
        pass
    r = compute_sets(boot, fixtures, hist, prefs, log)
    if not r:
        return None
    meta, arrs = r
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(SETS_NPZ, **arrs)
    json.dump(meta, open(SETS_META, "w", encoding="utf-8"))
    return meta


def _find(meta, x):
    """Index (set, mix) of a 7-value set among the tested ones, or None."""
    try:
        si = next(i for i, t in enumerate(TM_SETS) if all(abs(a - b) < 1e-9 for a, b in zip(t, x[:4])))
        return si, MIX.index((int(x[4]), int(x[5]), int(x[6])))
    except (StopIteration, ValueError):
        return None


def current_set(prefs):
    w = prefs.get("weights") or {"outlook": W_DEFAULT[0], "perf": W_DEFAULT[1], "value": W_DEFAULT[2]}
    return [float(prefs.get(k, TM_DEFAULT[k])) for k in TM_KEYS] + [w["outlook"], w["perf"], w["value"]]


def sets_for_page(meta, prefs, topn=8):
    """The rows the Model tab shows: presets, the current set, the best and the top few of every view."""
    if not meta or not os.path.exists(SETS_NPZ):
        return None
    z = np.load(SETS_NPZ)
    A, GAP, SE, TIE, OV = z["A"], z["GAP"], z["SE"], z["TIE"], z["OV"]
    sets = list(TM_SETS)
    chosen = {}                                        # (si, mi) -> row extras

    def add(idx, **kw):
        row = chosen.setdefault(tuple(idx), {"top": [], "nm": "", "ds": "", "pre": False, "cur": False})
        for k, v in kw.items():
            if k == "top":
                row["top"].append(v)
            else:
                row[k] = v

    for v, arr in zip(VIEWS, [OV] + [A[k] for k in range(4)]):
        if v in ("gd", "xgd", "ll"):
            # these measures do not depend on the Score mix: show the top distinct team-model sets, each paired with
            # the mix that does best overall for it
            for si in np.argsort(-arr[:, 0], kind="stable")[:topn]:
                add((int(si), int(np.argmax(OV[si]))), top=v)
        else:                                          # at most two mixes per team-model set, so the list shows variety
            seen, kept = {}, 0
            for f in np.argsort(-arr.ravel(), kind="stable")[:2000]:
                si, mi = np.unravel_index(f, arr.shape)
                if seen.get(int(si), 0) >= 2:
                    continue
                seen[int(si)] = seen.get(int(si), 0) + 1
                add((int(si), int(mi)), top=v)
                kept += 1
                if kept >= topn:
                    break
    for nm, x, ds in PRESETS:
        idx = _find(meta, x)
        if idx:
            add(idx, nm=nm, ds=ds, pre=True)
    cur = _find(meta, current_set(prefs))
    if cur:
        add(cur, cur=True)
    bo = _find(meta, meta["best"]["ov"])
    if bo:
        add(bo, nm=chosen.get(bo, {}).get("nm") or "Best overall on this data", ds=chosen.get(bo, {}).get("ds") or "Highest mean percentile across the four measures over all 7 numbers. Auto-pick does not simply copy this row: it uses its own team-model-only ranking and never touches the Score weights.")
    out = []
    for (si, mi), e in chosen.items():
        out.append({"x": [*[float(v) for v in sets[si]], *MIX[mi]], "v": [round(float(A[k][si, mi]), 4) for k in range(4)],
                    "gap": [round(float(GAP[k][si, mi]), 4) for k in range(4)], "se": [round(float(SE[k][si, mi]), 4) for k in range(4)],
                    "tie": [int(TIE[k][si, mi]) for k in range(4)], "ov": round(float(OV[si, mi]), 1),
                    "rk": [int((OV > OV[si, mi]).sum()) + 1, int((A[0] > A[0][si, mi]).sum()) + 1]
                          + [int((A[k][:, 0] > A[k][si, 0]).sum()) + 1 for k in (1, 2, 3)],       # team-only measures: rank among team sets
                    "cur": e["cur"], "pre": e["pre"], "nm": e["nm"], "ds": e["ds"], "top": e["top"]})
    return {**{k: meta[k] for k in ("finished", "gws", "pgws", "computed", "n_sets", "n_team", "n_mix", "n_obs", "grid", "best", "best_team", "walkforward", "stability", "official")},
            "n_tied": {k: (v if k == "pts" else v // meta["n_mix"]) for k, v in meta["n_tied"].items()},        # team-only measures count team sets
            "ov_best": round(float(OV.max()), 1), "rows": out, "on_grid": cur is not None}


def _cs_calibration(boot, fixtures, xg_all, prefs, last):
    """Is the model's clean-sheet probability honest? For every finished fixture, P(clean sheet) = exp(-opponent's
    expected goals) under the model fitted on earlier matches, against whether the team really kept one. Five equal-count
    bins, plus the Brier score against always predicting the base rate. This is the check to run before changing the
    goals model (for example a Dixon-Coles correction): it only matters if this is visibly off."""
    pred, act = [], []
    for g in range(2, last + 1):
        fx = [dict(f, finished=False if (f["event"] or 0) >= g else f["finished"]) for f in fixtures]
        m = fm.fit_teams(boot, fx, xg_all, g - 1, prefs)
        for f in fixtures:
            if f["event"] != g or not f["finished"] or f["team_h_score"] is None:
                continue
            for t, o, home, ga in ((f["team_h"], f["team_a"], True, f["team_a_score"]), (f["team_a"], f["team_h"], False, f["team_h_score"])):
                pred.append(math.exp(-m["lam"](o, t, not home))); act.append(1.0 if ga == 0 else 0.0)
    if len(pred) < 100:
        return None
    pred, act = np.array(pred), np.array(act)
    order = np.argsort(pred); bins = []
    for chunk in np.array_split(order, 5):
        bins.append({"n": int(len(chunk)), "pred": round(float(pred[chunk].mean()), 3), "act": round(float(act[chunk].mean()), 3)})
    base = float(act.mean())
    return {"n": int(len(pred)), "bins": bins, "brier": round(float(np.mean((pred - act) ** 2)), 4),
            "brier_base": round(float(np.mean((base - act) ** 2)), 4), "mean_pred": round(float(pred.mean()), 3), "mean_act": round(base, 3)}


def model_key(last, prefs):
    knobs = {k: prefs.get(k) for k in ("prior_strength", "prior_scale", "half_life_gw", "perf_min_minutes", "xg_weight")}
    code = hashlib.md5(open(fm.__file__, "rb").read() + open(__file__, "rb").read()).hexdigest()[:8]
    return f"{last}|{json.dumps(knobs, sort_keys=True)}|{code}"


def compute(boot, fixtures, hist, prefs):
    last = max((e["id"] for e in boot["events"] if e["finished"]), default=0)
    if last < FIRST_GW + 1:
        return None
    actual = {pid: {r["round"]: r["total_points"] for r in rows} for pid, rows in hist.items()}
    price = {e["id"]: e["now_cost"] / 10.0 for e in boot["elements"]}
    recs = []
    xg_all = fm.team_match_table(boot, fixtures, hist)
    for g in range(FIRST_GW, last + 1):
        b = copy.deepcopy(boot)
        for ev in b["events"]:
            ev["is_current"] = ev["id"] == g - 1; ev["is_next"] = ev["id"] == g
        for e in b["elements"]:
            e["status"] = "a"; e["chance_of_playing_next_round"] = None; e["news"] = ""
        fx = [dict(f, finished=False if (f["event"] or 0) >= g else f["finished"]) for f in fixtures]
        h = {pid: [r for r in rows if r["round"] < g] for pid, rows in hist.items()}
        h = {pid: rows for pid, rows in h.items() if rows}
        xg = fm.team_match_table(b, fx, h)
        model = fm.fit_teams(b, fx, xg, g - 1, prefs)
        players, _, _ = fm.project(b, fx, h, model, prefs, horizon_gws=3)
        for p in players:
            _cut_stats(p, h[p["id"]])
        players = fm.perf_scores(players, prefs.get("perf_min_minutes", 270))
        horizon = [x for x in range(g, g + 3) if x <= last]
        for p in players:
            p["xp1"] = sum(f[3] for f in p["fx"] if f[0] == g)
            p["xp3"] = sum(f[3] for f in p["fx"] if f[0] in horizon)
            rows = h[p["id"]]; pts = [r["total_points"] for r in rows]
            p["form3"] = float(np.mean(pts[-3:])); p["avg"] = float(np.mean(pts))
            p["y1"] = actual[p["id"]].get(g, 0.0)
            p["y3"] = sum(actual[p["id"]].get(x, 0.0) for x in horizon)
            p["price_now"] = price[p["id"]]
            p["reg"] = p["pm1"] >= 0.25 and len(p["fx"]) > 0
        # ---- accuracy records (players expected to play)
        for p in players:
            if p["fx"] and p["pm1"] >= 0.5:
                recs.append({"gw": g, "pos": p["p"], "xp1": p["xp1"], "form3": p["form3"], "avg": p["avg"], "y1": p["y1"]})
    gws = list(range(FIRST_GW, last + 1))

    acc = {}
    for pt, nm in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD")):
        r = [x for x in recs if x["pos"] == pt]
        if len(r) < 30:
            continue
        y = [x["y1"] for x in r]
        sp = lambda key: round(float(spearmanr([x[key] for x in r], y)[0]), 3)         # noqa: E731
        mae = lambda key: round(float(np.mean([abs(x[key] - x["y1"]) for x in r])), 2)  # noqa: E731
        acc[nm] = {"n": len(r), "rho": sp("xp1"), "rho_form": sp("form3"), "rho_avg": sp("avg"), "mae": mae("xp1"), "mae_avg": mae("avg")}
    per_gw = []
    for g in gws:
        r = [x for x in recs if x["gw"] == g]
        if len(r) > 30:
            per_gw.append({"gw": g, "rho": round(float(spearmanr([x["xp1"] for x in r], [x["y1"] for x in r])[0]), 3)})
    bias = round(float(np.mean([x["xp1"] - x["y1"] for x in recs])), 2) if recs else 0.0
    return {"key": model_key(last, prefs), "finished": last, "gws": [gws[0], gws[-1]], "computed": time.strftime("%d %b %H:%M"),
            "n": len(recs), "fdr": _fdr_run(boot, fixtures, xg_all, prefs, last), "cs": _cs_calibration(boot, fixtures, xg_all, prefs, last), "accuracy": acc, "per_gw": per_gw, "bias": bias}


def get(boot, fixtures, hist, prefs, force=False):
    """Cached result; recomputed only when a new gameweek has finished or the model/knobs changed."""
    last = max((e["id"] for e in boot["events"] if e["finished"]), default=0)
    try:
        c = json.load(open(FILE, encoding="utf-8"))
        if not force and c.get("key") == model_key(last, prefs):
            return c
    except Exception:                                   # noqa: BLE001
        pass
    r = compute(boot, fixtures, hist, prefs)
    if r:
        os.makedirs(DATA, exist_ok=True)
        json.dump(r, open(FILE, "w", encoding="utf-8"))
    return r


def main(root, prefs):
    boot, fixtures = fd.load_core()
    hist = fd.load_histories(boot)
    r = get(boot, fixtures, hist, prefs, force=True)
    if not r:
        print("Not enough finished gameweeks to backtest yet.")
        return
    print(f"Backtest over GW{r['gws'][0]}-GW{r['gws'][1]}: {r['n']} player-gameweeks\n")
    print("1) Single-gameweek projection quality (Spearman rank correlation with actual points; higher is better)")
    print(f"   {'pos':4} {'model':>7} {'form3':>7} {'avg':>7}   MAE model  MAE avg")
    for nm, a in r["accuracy"].items():
        print(f"   {nm:4} {a['rho']:7.3f} {a['rho_form']:7.3f} {a['rho_avg']:7.3f}   {a['mae']:8.2f} {a['mae_avg']:8.2f}")
    print(f"   mean bias (projected - actual): {r['bias']:+.2f} pts/player/GW\n")
    fd_ = r.get("fdr")
    if fd_:
        print(f"2) Fixture difficulty: model vs FPL's official FDR ({fd_['n']} team-fixtures; Spearman of 'ease' vs result, higher is better)")
        for lab, t in fd_["targets"].items():
            print(f"   {lab:16} pooled: official {t['official']:+.3f}  model {t['model']:+.3f} | per-GW mean: official {t['gw_official']:+.3f}  model {t['gw_model']:+.3f}"
                  f"  diff {t['gw_diff']:+.3f} (se {t['gw_se']:.3f}), model better in {t['gw_model_wins']}/{t['gw_n']} GWs")
        print(f"   attack: model {fd_['attack_rho']:+.3f} vs official {fd_['official_gf_rho']:+.3f} (goals scored); defence: model {fd_['defence_rho']:+.3f} vs official {fd_['official_ga_rho']:+.3f} (goals conceded)")
        print("   Official FDR is today's value, not as-of-then; the model is strictly pre-gameweek.\n")
    cs = r.get("cs")
    if cs:
        print(f"2b) Clean-sheet calibration ({cs['n']} team-fixtures): predicted vs actual clean-sheet rate by fifth, lowest to highest chance")
        print("    " + "   ".join(f"{b['pred']:.2f}->{b['act']:.2f}" for b in cs["bins"]) + f"   | Brier {cs['brier']} vs {cs['brier_base']} always predicting the base rate ({cs['mean_act']})\n")
    meta = sets_get(boot, fixtures, hist, prefs, force=True)
    if meta:
        pg = sets_for_page(meta, prefs)
        print(f"3) Weight sets ({meta['n_sets']:,} = {meta['n_team']} team-model sets x {meta['n_mix']} score mixes; GW1-GW{meta['finished']}, took {meta['seconds']}s)")
        print(f"   x = [{', '.join(SET_KEYS)}]")
        names = {"ov": "overall (mean percentile)", "pts": "points of the top picks", "gd": "goal difference", "xgd": "xG difference", "ll": "goals likelihood"}
        for v in VIEWS:
            top = sorted([row for row in pg["rows"] if v in row["top"]], key=lambda row: -(row["ov"] if v == "ov" else row["v"][MEAS.index(v)]))[:3]
            print(f"   best by {names[v]}:")
            for row in top:
                val = row["ov"] if v == "ov" else row["v"][MEAS.index(v)]
                print(f"     {row['x']}  {val}")
        cur = next((row for row in pg["rows"] if row["cur"]), None)
        if cur:
            print(f"   your current set {cur['x']}: overall {cur['ov']} (rank {cur['rk'][0]:,} of {meta['n_sets']:,})")
        for k, v in meta["official"].items():
            print(f"   official FDR on {names[k]}: {v['v']:+.4f}")
        print(f"   odd/even gameweek stability: {meta['stability']}")
        print(f"   auto-pick would choose the team model {meta['best_team']}")
        wf = meta.get("walkforward")
        if wf:
            t, m = wf["team"], wf["mix"]
            print(f"\n4) Walk-forward: pick using only earlier gameweeks, score on the next (GW{wf['gws'][0]}-GW{wf['gws'][-1]}, {len(wf['gws'])} test gameweeks)")
            print(f"   team model, picked vs default vs hindsight-best:  " + "  ".join(f"{k} {t['picked'][k]:+.3f}/{t['default'][k]:+.3f}/{t['oracle'][k]:+.3f}" for k in MEAS))
            print(f"   picked minus default: " + "  ".join(f"{k} {t['diff'][k]['mean']:+.3f} (se {t['diff'][k]['se']:.3f}, better in {t['diff'][k]['wins']}/{t['diff'][k]['n']})" for k in MEAS) + f"; changed pick {t['changes']} times")
            print(f"   Score mix by Points, picked vs default vs hindsight-best: {m['picked']:.2f} / {m['default']:.2f} / {m['oracle']:.2f};  picked minus default {m['diff']['mean']:+.3f} (se {m['diff']['se']:.3f}, better in {m['diff']['wins']}/{m['diff']['n']}); value weights picked: {m['value_weights']}")
    print("\nCaveat: one season, few gameweeks, overlapping windows, today's prices; the best of thousands of sets is flattered by")
    print("being picked on the same data. Treat a tie as a tie.")
