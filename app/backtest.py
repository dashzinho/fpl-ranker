"""Backtest: how good are the projections, and which settings would have worked best?

For every already-played gameweek g the model is re-run using ONLY data from before g: team ratings fitted on matches
before g, player histories cut at g-1, and the season-performance inputs recomputed from those same cut histories
(nothing from the full season leaks in). Injury flags are neutral because past news is not stored. The real
`perf_scores` and the page's own Score formula are used.

Two outputs:
  * data/backtest.json   accuracy of the single-gameweek projections by position, against two naive baselines, plus the
                         comparison of the team model's fixture difficulty with FPL's official rating (python fpl.py --backtest)
  * data/backtest_sets.* ONE table of weight sets. A set = the four team-model weights (xG share, half-life, prior
                         strength, prior scale) + the Score weights (outlook / perf / value in steps of 5). Every
                         combination is scored on four measures, each refitted per gameweek on earlier data only:
                           pts  realised points over the next 3 GWs of the top picks per position (GW4 onward)
                           gd   rank correlation of the model's fixture ease with the real goal difference
                           xgd  the same with the xG difference
                           ll   Poisson log-likelihood of the goals each team scored
                         plus an overall score (mean percentile across the four) that auto-pick uses. Each measure has
                         a paired standard error versus its best set, so near-ties show as ties instead of a crown.

Honest limits: one season, few gameweeks, overlapping 3-GW windows (the standard errors are optimistic), prices are
today's, and picking the best of ~30k combinations on this data flatters the winner. Re-run as the season progresses.
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

def _fdr_records(fixtures, xg, model, g):
    """One record per (finished fixture in GW g, team): the model's pre-GW expectation, FPL's official difficulty
    and what happened. Official FDR is today's value (FPL rarely changes it mid-season), the model is as-of before g."""
    out = []
    for f in fixtures:
        if f["event"] != g or not f["finished"] or f["team_h_score"] is None:
            continue
        for t, o, home, gf, ga, d in ((f["team_h"], f["team_a"], True, f["team_h_score"], f["team_a_score"], f["team_h_difficulty"]),
                                      (f["team_a"], f["team_h"], False, f["team_a_score"], f["team_h_score"], f["team_a_difficulty"])):
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
TM_GRID = {"xg_weight": [0.0, 0.65, 1.0], "half_life_gw": [4, 12, 24, 100],
           "prior_strength": [1, 5, 10, 20], "prior_scale": [0.05, 0.10, 0.20]}
TM_KEYS = tuple(TM_GRID)
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
    sets = list(itertools.product(*TM_GRID.values()))
    nS, nM = len(sets), len(MIX)
    gws = list(range(1, last + 1))
    pgws = [g for g in gws if g >= FIRST_GW]
    Mx = np.array(MIX, float)
    T = {k: np.full((nS, len(gws)), np.nan) for k in ("gd", "xgd", "ll")}
    P = np.full((nS, nM, len(pgws)), np.nan)
    off = {"gd": np.full(len(gws), np.nan), "xgd": np.full(len(gws), np.nan)}
    lg = np.vectorize(math.lgamma)
    n_obs = 0
    log(f"  backtesting {nS * nM:,} weight sets ({nS} team-model sets x {nM} score mixes) over GW1-GW{last}...")
    for gi, g in enumerate(gws):
        rows = []                                      # (team, opp, home, gf, ga, xg diff, official difficulty)
        for f in fixtures:
            if f["event"] != g or not f["finished"] or f["team_h_score"] is None:
                continue
            xh, xa = xg_all.get((f["id"], f["team_h"])), xg_all.get((f["id"], f["team_a"]))
            ok = xh is not None and xa is not None
            for t, o, home, gf, ga, d in ((f["team_h"], f["team_a"], True, f["team_h_score"], f["team_a_score"], f["team_h_difficulty"]),
                                          (f["team_a"], f["team_h"], False, f["team_a_score"], f["team_h_score"], f["team_a_difficulty"])):
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
            SE[k] = _se(P[si, mi][None, None, :] - P, 2)
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
    meta = {"key": _sets_key(last, prefs), "finished": last, "gws": [1, last], "pgws": [pgws[0], last], "computed": time.strftime("%d %b %H:%M"),
            "seconds": round(time.time() - t0), "n_sets": nS * nM, "n_team": nS, "n_mix": nM, "n_obs": n_obs, "grid": TM_GRID, "keys": list(SET_KEYS),
            "best": {k: x_of(*v) for k, v in best.items()}, "stability": stab, "official": official,
            "n_tied": {name: int(TIE[k].sum()) for k, name in enumerate(MEAS)}}
    return meta, {"A": A.astype(np.float32), "GAP": GAP.astype(np.float32), "SE": SE.astype(np.float32), "TIE": TIE, "OV": OV.astype(np.float32)}


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
    """Index (set, mix) of a 7-value set on the tested grid, or None."""
    g = meta["grid"]
    try:
        si = 0
        for k, key in enumerate(TM_KEYS):
            vals = g[key]
            j = next(i for i, v in enumerate(vals) if abs(v - x[k]) < 1e-9)
            si = si * len(vals) + j
        mi = MIX.index((int(x[4]), int(x[5]), int(x[6])))
        return si, mi
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
    sets = list(itertools.product(*TM_GRID.values()))
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
        add(bo, nm=chosen.get(bo, {}).get("nm") or "Best overall on this data", ds=chosen.get(bo, {}).get("ds") or "Highest mean percentile across the four measures: what auto-pick chooses.")
    out = []
    for (si, mi), e in chosen.items():
        out.append({"x": [*[float(v) for v in sets[si]], *MIX[mi]], "v": [round(float(A[k][si, mi]), 4) for k in range(4)],
                    "gap": [round(float(GAP[k][si, mi]), 4) for k in range(4)], "se": [round(float(SE[k][si, mi]), 4) for k in range(4)],
                    "tie": [int(TIE[k][si, mi]) for k in range(4)], "ov": round(float(OV[si, mi]), 1),
                    "rk": [int((OV > OV[si, mi]).sum()) + 1, int((A[0] > A[0][si, mi]).sum()) + 1]
                          + [int((A[k][:, 0] > A[k][si, 0]).sum()) + 1 for k in (1, 2, 3)],       # team-only measures: rank among team sets
                    "cur": e["cur"], "pre": e["pre"], "nm": e["nm"], "ds": e["ds"], "top": e["top"]})
    return {**{k: meta[k] for k in ("finished", "gws", "pgws", "computed", "n_sets", "n_team", "n_mix", "n_obs", "grid", "best", "stability", "official")},
            "n_tied": {k: (v if k == "pts" else v // meta["n_mix"]) for k, v in meta["n_tied"].items()},        # team-only measures count team sets
            "ov_best": round(float(OV.max()), 1), "rows": out, "on_grid": cur is not None}


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
            "n": len(recs), "fdr": _fdr_run(boot, fixtures, xg_all, prefs, last), "accuracy": acc, "per_gw": per_gw, "bias": bias}


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
    print("\nCaveat: one season, few gameweeks, overlapping windows, today's prices; the best of thousands of sets is flattered by")
    print("being picked on the same data. Treat a tie as a tie.")
