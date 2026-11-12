"""Backtest: how good are the projections, and which Score weights would have picked better players?

For every already-played gameweek g (from GW4 on) the model is re-run using ONLY data from before g: team ratings
fitted on matches before g, player histories cut at g-1, and the season-performance inputs (minutes, points, xGI,
bonus) recomputed from those same cut histories (nothing from the full season leaks in). Injury flags are neutral
because past news is not stored. The real `perf_scores` and the page's own Score formula are used, so the weights
tested are the weights the Rankings page applies.

Outputs (data/backtest.json, shown on the page's Model tab):
  * accuracy of the single-gameweek projections by position, against two naive baselines
  * for every weight mix (outlook/perf/value in steps of 5 summing to 100): the realised points over the next 3 GWs of
    the top picks per position (the same picks the Overview shows), and the same per GBP million, each with a paired
    standard error versus the best mix, so near-ties are visible instead of a single "optimal" crown
  * stability: do the odd and even gameweeks rank the mixes alike? (low = the ranking is mostly noise)

Honest limits: one season, few gameweeks, overlapping 3-GW windows (the standard errors are optimistic), prices are
today's, and picking the best of ~200 mixes on this data flatters the winner. Re-run as the season progresses.
Run:  python fpl.py --backtest      (the page also refreshes it automatically when a gameweek finishes)
"""
import copy, hashlib, json, os, time
import numpy as np
from scipy.stats import spearmanr
import fpl_data as fd
import fpl_model as fm
from paths import DATA

FILE = os.path.join(DATA, "backtest.json")
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


def model_key(last, prefs):
    knobs = {k: prefs.get(k) for k in ("prior_strength", "prior_scale", "half_life_gw", "perf_min_minutes")}
    code = hashlib.md5(open(fm.__file__, "rb").read() + open(__file__, "rb").read()).hexdigest()[:8]
    return f"{last}|{json.dumps(knobs, sort_keys=True)}|{code}"


def compute(boot, fixtures, hist, prefs):
    last = max((e["id"] for e in boot["events"] if e["finished"]), default=0)
    if last < FIRST_GW + 1:
        return None
    actual = {pid: {r["round"]: r["total_points"] for r in rows} for pid, rows in hist.items()}
    price = {e["id"]: e["now_cost"] / 10.0 for e in boot["elements"]}
    recs, scores = [], {}                                    # scores[(wo,wp,wv)][gw] = {pos: (pts, cost)}
    mix_of = {m: {} for m in MIX}
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
        # ---- weight mixes, evaluated with the page's own Score formula
        for pt, k in TOPN.items():
            grp = [p for p in players if p["p"] == pt]
            if len(grp) < k * 3:
                continue
            mask = [p["reg"] for p in grp]
            po = _pct([p["xp3"] for p in grp], mask)
            pv = _pct([p["xp3"] / p["price_now"] for p in grp], mask)
            pf = np.array([p["perf"] for p in grp], float)
            y3 = np.array([p["y3"] for p in grp]); pr = np.array([p["price_now"] for p in grp])
            mk = np.array(mask)
            for wo, wp, wv in MIX:
                sc = np.where(mk, (wo * po + wp * pf + wv * pv) / 100.0, 0.0)
                top = np.argsort(-sc, kind="stable")[:k]
                mix_of[(wo, wp, wv)].setdefault(g, {})[pt] = (float(y3[top].mean()), float(y3[top].sum() / max(pr[top].sum(), 0.1)))
    gws = list(range(FIRST_GW, last + 1))

    def series(mx, idx):
        out = []
        for g in gws:
            d = mix_of[mx].get(g, {})
            out.append(float(np.mean([v[idx] for v in d.values()])) if d else np.nan)
        return np.array(out)

    res = {}
    for target, idx in (("pts", 0), ("per_m", 1)):
        S = {mx: series(mx, idx) for mx in MIX}
        avg = {mx: float(np.nanmean(v)) for mx, v in S.items()}
        best = max(avg, key=avg.get)
        rows = []
        for mx in MIX:
            diff = S[best] - S[mx]
            n = int(np.sum(~np.isnan(diff)))
            se = float(np.nanstd(diff, ddof=1) / np.sqrt(n)) if n > 2 else 0.0
            gap = avg[best] - avg[mx]
            rows.append({"wo": mx[0], "wp": mx[1], "wv": mx[2], "v": round(avg[mx], 3), "gap": round(gap, 3), "se": round(se, 3),
                         "tie": bool(gap <= 2 * se + 1e-9)})
        rows.sort(key=lambda r: -r["v"])
        odd = np.array([np.nanmean(S[mx][0::2]) for mx in MIX]); even = np.array([np.nanmean(S[mx][1::2]) for mx in MIX])
        rho = float(spearmanr(odd, even)[0]) if len(set(odd)) > 1 and len(set(even)) > 1 else 0.0
        res[target] = {"mixes": rows, "best": {"wo": best[0], "wp": best[1], "wv": best[2]}, "n_tied": sum(r["tie"] for r in rows),
                       "stability": round(rho, 2)}
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
            "n": len(recs), "accuracy": acc, "per_gw": per_gw, "bias": bias, "topn": TOPN, **res}


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
    cur = prefs["weights"]
    for target, label in (("pts", "points of the top picks over the next 3 GWs"), ("per_m", "points per GBP million of the top picks")):
        t = r[target]
        print(f"2) Weights, by {label}")
        for m in t["mixes"][:6]:
            print(f"   outlook {m['wo']:3d} / perf {m['wp']:3d} / value {m['wv']:3d}  ->  {m['v']:6.2f}   gap {m['gap']:.2f} (se {m['se']:.2f}){'  ~tied with best' if m['tie'] else ''}")
        mine = next((m for m in t["mixes"] if (m["wo"], m["wp"], m["wv"]) == (cur["outlook"], cur["perf"], cur["value"])), None)
        if mine:
            print(f"   yours {cur['outlook']}/{cur['perf']}/{cur['value']}: {mine['v']:.2f} (rank {t['mixes'].index(mine) + 1} of {len(t['mixes'])}, {'tied with the best' if mine['tie'] else 'behind the best'})")
        print(f"   {t['n_tied']} of {len(t['mixes'])} mixes are statistically tied with the best; odd-vs-even gameweek stability {t['stability']:+.2f}\n")
    print("Caveat: one season, few gameweeks, overlapping windows, today's prices; the best of ~200 mixes is flattered by")
    print("being picked on the same data. Treat a tie as a tie.")
