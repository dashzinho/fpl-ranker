"""Does the choice of team-model settings hold up on MORE data than this season?

The weight-set backtest on the Model tab only has the current season (about ten gameweeks), so most sets are
statistically tied. This script re-scores the same team-model sets on past seasons from the community archive
vaastav/Fantasy-Premier-League (public, no key) and reports how stable the ordering is across seasons.

Only the three TEAM-MODEL measures can be compared: goal difference, xG difference and goals likelihood. Points cannot,
because past seasons have no defensive_contribution points (introduced in 2025-26) and a different squad shape.
Seasons without xG (2021-22 and earlier) are skipped.

Run:  python tools/multi_season.py            (downloads about 5 MB per season into data/history/, then prints a report)
"""
import csv, io, math, os, sys, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import numpy as np
from scipy.stats import rankdata, spearmanr
import fpl_model as fm
import backtest as bt

SEASONS = ["2022-23", "2023-24", "2024-25"]
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/{f}"
CACHE = os.path.join(ROOT, "data", "history")


def fetch(season, name):
    path = os.path.join(CACHE, season, name.replace("/", "_"))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(BASE.format(s=season, f=name), timeout=60) as r:
            open(path, "wb").write(r.read())
    return list(csv.DictReader(open(path, encoding="utf-8")))


def load_season(season):
    teams = fetch(season, "teams.csv")
    fixtures = fetch(season, "fixtures.csv")
    gws = fetch(season, "gws/merged_gw.csv")
    if "expected_goals" not in gws[0]:
        return None
    name_id = {t["name"]: int(t["id"]) for t in teams}
    # past seasons publish strength_overall_* on a ~1000+ scale; the model expects FPL's 1-5 rating, which is `strength`
    boot = {"teams": [{"id": int(t["id"]), "strength_overall_home": int(t["strength"]), "strength_overall_away": int(t["strength"])} for t in teams]}
    assert all(1 <= x["strength_overall_home"] <= 5 for x in boot["teams"]), "strength is not on the 1-5 scale"
    fx = []
    for f in fixtures:
        if not f.get("event"):
            continue
        done = f["finished"] == "True" and f["team_h_score"] not in ("", None)
        fx.append({"id": int(f["id"]), "event": int(f["event"]), "finished": done, "team_h": int(f["team_h"]), "team_a": int(f["team_a"]),
                   "team_h_score": int(float(f["team_h_score"])) if done else None, "team_a_score": int(float(f["team_a_score"])) if done else None,
                   "team_h_difficulty": int(f["team_h_difficulty"]), "team_a_difficulty": int(f["team_a_difficulty"])})
    xg = {}
    for r in gws:
        tid = name_id.get(r["team"])
        if tid is None:
            continue
        k = (int(r["fixture"]), tid)
        xg[k] = xg.get(k, 0.0) + float(r["expected_goals"] or 0)
    return boot, fx, xg


def season_series(boot, fx, xg):
    last = max((f["event"] for f in fx if f["finished"]), default=0)
    sets = list(bt.TM_SETS)
    T = {k: np.full((len(sets), last), np.nan) for k in ("gd", "xgd", "ll")}
    off = {"gd": np.full(last, np.nan), "xgd": np.full(last, np.nan)}
    lg = np.vectorize(math.lgamma)
    for g in range(1, last + 1):
        rows = []
        for f in fx:
            if f["event"] != g or not f["finished"]:
                continue
            xh, xa = xg.get((f["id"], f["team_h"])), xg.get((f["id"], f["team_a"]))
            ok = xh is not None and xa is not None
            for t, o, home, gf, ga, d in ((f["team_h"], f["team_a"], True, f["team_h_score"], f["team_a_score"], f["team_h_difficulty"]),
                                          (f["team_a"], f["team_h"], False, f["team_a_score"], f["team_h_score"], f["team_a_difficulty"])):
                x = (xh - xa if home else xa - xh) if ok else float(gf - ga)
                rows.append((t, o, home, float(gf), float(ga), x, float(d)))
        if len(rows) < 10:
            continue
        gf_ = np.array([r[3] for r in rows]); gd_ = gf_ - np.array([r[4] for r in rows]); xd_ = np.array([r[5] for r in rows])
        off["gd"][g - 1] = bt._rho(-np.array([r[6] for r in rows]), gd_); off["xgd"][g - 1] = bt._rho(-np.array([r[6] for r in rows]), xd_)
        masked = [dict(f, finished=False if f["event"] >= g else f["finished"]) for f in fx]
        for si, vals in enumerate(sets):
            m = fm.fit_teams(boot, masked, xg, g - 1, dict(zip(bt.TM_KEYS, vals)))
            lt = np.array([m["lam"](r[0], r[1], r[2]) for r in rows]); lo = np.array([m["lam"](r[1], r[0], not r[2]) for r in rows])
            ease = np.log(lt / lo)
            T["gd"][si, g - 1] = bt._rho(ease, gd_); T["xgd"][si, g - 1] = bt._rho(ease, xd_)
            T["ll"][si, g - 1] = float(np.mean(gf_ * np.log(lt) - lt - lg(gf_ + 1)))
    return T, off, last


def main():
    per, offs = {}, {}
    for s in SEASONS:
        print(f"{s}: loading...", flush=True)
        try:
            d = load_season(s)
        except Exception as e:                           # noqa: BLE001
            print(f"  skipped ({e})"); continue
        if d is None:
            print("  skipped (no xG in this season)"); continue
        per[s], offs[s], _ = season_series(*d)
    if not per:
        sys.exit("no seasons could be loaded")
    seasons = list(per)
    nS = len(bt.TM_SETS); d0 = bt.TM_SETS.index(tuple(bt.TM_DEFAULT[k] for k in bt.TM_KEYS))
    print(f"\nTeam-model sets scored on {len(seasons)} past seasons ({', '.join(seasons)}); {sum(int((~np.isnan(per[s]['gd'][0])).sum()) for s in seasons)} gameweeks")
    allT = {k: np.concatenate([per[s][k] for s in seasons], axis=1) for k in ("gd", "xgd", "ll")}
    names = {"gd": "goal difference (rank corr.)", "xgd": "xG difference (rank corr.)", "ll": "goals likelihood (log-lik.)"}
    for k in ("gd", "xgd", "ll"):
        v = bt._nanmean(allT[k], 1); best = int(np.argmax(v))
        diff = allT[k][best] - allT[k]
        se = bt._se(diff, 1)
        print(f"\n{names[k]}")
        for si in np.argsort(-v)[:4]:
            print(f"   {bt.TM_SETS[si]}  {v[si]:+.4f}  gap {v[best] - v[si]:.4f} (se {se[si]:.4f})")
        print(f"   default {bt.TM_SETS[d0]}  {v[d0]:+.4f}  rank {int((v > v[d0]).sum()) + 1} of {nS}; gap to best {v[best] - v[d0]:.4f} (se {se[d0]:.4f}); tied sets: {int((v[best] - v <= 2 * se + 1e-12).sum())} of {nS}")
        if k in ("gd", "xgd"):
            ov = np.nanmean(np.concatenate([offs[s][k] for s in seasons]))
            offall = np.concatenate([offs[s_][k] for s_ in seasons]); dd = allT[k][d0] - offall
            print(f"   official FDR  {ov:+.4f}  (best set {v[best]:+.4f}, default {v[d0]:+.4f}); default minus FDR {np.nanmean(dd):+.4f} (se {float(bt._se(dd, 0)):.4f}), default better in {int((dd > 0).sum())} of {int((~np.isnan(dd)).sum())} gameweeks")
    print(f"\nIs the ordering of sets stable from one season to the next? (rank correlation of the {nS} set scores)")
    for k in ("gd", "xgd", "ll"):
        for a in range(len(seasons)):
            for b in range(a + 1, len(seasons)):
                ra, rb = bt._nanmean(per[seasons[a]][k], 1), bt._nanmean(per[seasons[b]][k], 1)
                print(f"   {names[k]:30} {seasons[a]} vs {seasons[b]}: {spearmanr(ra, rb)[0]:+.2f}")
    mean_pct = sum(rankdata(bt._nanmean(allT[k], 1)) for k in ("gd", "xgd", "ll")) / 3
    print(f"\nBest by mean rank of the three measures over all seasons: {bt.TM_SETS[int(np.argmax(mean_pct))]}   (this season's auto-pick: see the Model tab)")


if __name__ == "__main__":
    main()
