"""Backtest: how good are the projections, and which ranking weights would have picked better players?

For every already-played gameweek g (from GW4 on) the model is re-run using ONLY data from before g (team ratings
fitted on matches before g, player histories cut at g-1, injury flags neutral because past news isn't stored),
then compared with what actually happened. It reports:

  * rank correlation + mean absolute error of the projected points for GW g (vs. simple baselines)
  * a grid search over the three ranking weights (outlook / season performance / value): which mix would have picked
    the players who scored most over the next 3 GWs

Caveat printed with the results: one season, a handful of gameweeks - treat as a sanity check, not proof.
Run:  python fpl.py --backtest
"""
import copy, itertools, os
import numpy as np
from scipy.stats import spearmanr
import fpl_data as fd
import fpl_model as fm

TOPN = {1: 3, 2: 8, 3: 8, 4: 5}


def _pct(vals):
    a = np.array(vals, float)
    return a.argsort().argsort() / max(1, len(a) - 1) * 100


def main(root, prefs):
    boot, fixtures = fd.load_core()
    hist = fd.load_histories(boot)
    last = max((e["id"] for e in boot["events"] if e["finished"]), default=0)
    if last < 5:
        print("Not enough finished gameweeks to backtest yet (need at least 5).")
        return
    pos_of = {e["id"]: e["element_type"] for e in boot["elements"]}
    price = {e["id"]: e["now_cost"] / 10.0 for e in boot["elements"]}
    actual = {pid: {r["round"]: r["total_points"] for r in rows} for pid, rows in hist.items()}
    recs = []                                                       # one dict per (gw, player)
    for g in range(4, last + 1):
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
        players, _, gws = fm.project(b, fx, h, model, prefs, horizon_gws=3)
        horizon = [x for x in range(g, g + 3) if x <= last]
        for p in players:
            if not p["fx"] or p["pm1"] < 0.5:
                continue
            xp1 = sum(f[3] for f in p["fx"] if f[0] == g)
            xp3 = sum(f[3] for f in p["fx"] if f[0] in horizon)
            rows = h[p["id"]]
            games = [r["total_points"] for r in rows]
            mins = sum(r["minutes"] for r in rows)
            recs.append({"gw": g, "id": p["id"], "pos": p["p"], "price": p["pr"], "xp1": xp1, "xp3": xp3,
                         "form3": float(np.mean(games[-3:])), "avg": float(np.mean(games)),
                         "p90": sum(games) / (mins / 90.0) if mins else 0.0, "mins": mins,
                         "y1": actual[p["id"]].get(g, 0.0), "y3": sum(actual[p["id"]].get(x, 0.0) for x in horizon)})
    print(f"Backtest over GW4-GW{last}: {len(recs)} player-gameweeks\n")

    # ---- 1) point-forecast quality (GW g only), players expected to play
    print("1) Single-gameweek projection quality (Spearman rank correlation with actual points; higher is better)")
    print(f"   {'pos':4} {'model':>7} {'form3':>7} {'avg':>7}   MAE model  MAE avg")
    for pt, nm in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD")):
        r = [x for x in recs if x["pos"] == pt]
        if len(r) < 30:
            continue
        y = [x["y1"] for x in r]
        sp = lambda k: spearmanr([x[k] for x in r], y)[0]
        mae = lambda k: float(np.mean([abs(x[k] - x["y1"]) for x in r]))
        print(f"   {nm:4} {sp('xp1'):7.3f} {sp('form3'):7.3f} {sp('avg'):7.3f}   {mae('xp1'):8.2f} {mae('avg'):8.2f}")
    bias = float(np.mean([x["xp1"] - x["y1"] for x in recs]))
    print(f"   mean bias (projected - actual): {bias:+.2f} pts/player/GW\n")

    # ---- 2) weight grid: realised points (next 3 GWs) of the top picks per position, averaged over gameweeks
    grid = []
    cur = (prefs["weights"]["outlook"], prefs["weights"]["perf"], prefs["weights"]["value"])
    combos = [(wo, wp) for wo, wp in itertools.product(range(0, 101, 10), repeat=2) if wo + wp <= 100] + [(cur[0], cur[1])]
    for wo, wp in combos:
        wv = 100 - wo - wp
        tot, n = 0.0, 0
        for g in sorted({x["gw"] for x in recs}):
            for pt, k in TOPN.items():
                r = [x for x in recs if x["gw"] == g and x["pos"] == pt]
                if len(r) < k * 3:
                    continue
                po = _pct([x["xp3"] for x in r]); pv = _pct([x["xp3"] / x["price"] for x in r])
                reg = [x["mins"] >= 270 for x in r]
                pf = _pct([x["p90"] if x["mins"] >= 270 else 0 for x in r])
                score = [(wo * a + wp * bb + wv * c) for a, bb, c in zip(po, pf, pv)]
                top = np.argsort(score)[::-1][:k]
                tot += float(np.mean([r[i]["y3"] for i in top])); n += 1
        grid.append((tot / n if n else 0, wo, wp, wv))
    grid.sort(reverse=True)
    print("2) Ranking weights: average realised points over the next 3 GWs of the top picks per position")
    print("   (outlook = projected points, perf = points per 90 so far, value = projected points per GBP million)")
    for sc, wo, wp, wv in grid[:6]:
        print(f"   outlook {wo:3d} / perf {wp:3d} / value {wv:3d}  ->  {sc:6.2f}")
    mine = next((x for x in grid if (x[1], x[2], x[3]) == cur), None)
    if mine:
        print(f"   your current {cur[0]}/{cur[1]}/{cur[2]}: {mine[0]:.2f}  (rank {grid.index(mine) + 1} of {len(grid)})")
    top_only = next((x for x in grid if (x[1], x[2], x[3]) == (100, 0, 0)), None)
    if top_only:
        print(f"   projections only 100/0/0: {top_only[0]:.2f}")
    print("\nCaveat: one season so far, few gameweeks, top-N picks overlap heavily between weightings - differences of")
    print("~0.1-0.2 pts are noise. Past injury news is not stored, so availability is assumed 100% (flatters nobody, but")
    print("is slightly harsher on the model than real use). Re-run as the season progresses.")
