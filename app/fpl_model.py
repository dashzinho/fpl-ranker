"""Team-strength model + expected-points model.

1) Teams: Poisson attack/defence ratings fitted on finished matches (target = xg_weight xG + the rest goals, 65/35 by default, recent
   matches weigh more). FPL's coarse 2-5 overall strength is only a *prior* that fades as matches are played,
   so a side that's expected to be bad but is performing well stops being treated as an easy fixture.
2) Players: expected points per upcoming fixture from blended xG/xA/goal/assist rates (shrunk toward the
   position average), bonus, clean sheets, goals conceded, saves, defensive contributions, cards and
   a minutes/availability model. Fixture effects come from (1).
"""
import math
import numpy as np
from scipy.optimize import minimize, milp, LinearConstraint, Bounds

POS_SHORT = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DC_THR = {2: 10, 3: 12, 4: 12}       # defensive-contribution thresholds (CBIT / CBIRT)
# Premier League yellow-card suspensions: (yellow cards, reached by this league match -> ban length in matches)
YC_RULES = ((5, 19, 1), (10, 32, 2))


# ------------------------------------------------------------------ teams
def team_match_table(boot, fixtures, hist):
    """xG for/against per (fixture, team) from summed player xG."""
    team_of = {e["id"]: e["team"] for e in boot["elements"]}
    xg = {}
    for pid, rows in hist.items():
        t = team_of.get(pid)
        for r in rows:
            k = (r["fixture"], t)
            xg[k] = xg.get(k, 0.0) + float(r.get("expected_goals") or 0)
    return xg


def fit_teams(boot, fixtures, xg, gw_now, prefs):
    ids = sorted(t["id"] for t in boot["teams"])
    idx = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    tm = {t["id"]: t for t in boot["teams"]}
    obs = []
    wx = prefs.get("xg_weight", 0.65)                 # share of xG (vs actual goals) in the fit target
    for f in fixtures:
        if not f["finished"] or f["team_h_score"] is None or f["event"] is None:
            continue
        h, a = f["team_h"], f["team_a"]
        xh, xa = xg.get((f["id"], h)), xg.get((f["id"], a))
        yh = wx * xh + (1 - wx) * f["team_h_score"] if xh else float(f["team_h_score"])
        ya = wx * xa + (1 - wx) * f["team_a_score"] if xa else float(f["team_a_score"])
        w = 0.5 ** ((gw_now - f["event"]) / prefs.get("half_life_gw", 12))
        obs.append((idx[h], idx[a], 1.0, yh, w))
        obs.append((idx[a], idx[h], -1.0, ya, w))
    s_overall = np.array([(tm[t]["strength_overall_home"] + tm[t]["strength_overall_away"]) / 2 for t in ids], float)
    k = prefs.get("prior_scale", 0.10)
    a0 = k * (s_overall - 3.2)
    d0 = -k * (s_overall - 3.2)
    if obs:
        I, J, S, Y, W = (np.array(c) for c in zip(*obs))
        I, J = I.astype(int), J.astype(int)
        mu = float((W * Y).sum() / W.sum())
    else:
        I = J = np.array([], int); S = Y = W = np.array([]); mu = 1.4
    ridge = 0.7 * prefs.get("prior_strength", 5.0)

    def unpack(th): return th[:n], th[n:2 * n], th[2 * n]

    def loss(th):
        a, d, h = unpack(th)
        g = np.zeros_like(th)
        val = 0.0
        if len(I):
            lam = mu * np.exp(a[I] + d[J] + h * S / 2)
            val += float((W * (lam - Y * np.log(lam))).sum())
            r = W * (lam - Y)
            ga = np.zeros(n); gd = np.zeros(n)
            np.add.at(ga, I, r); np.add.at(gd, J, r)
            g[:n] += ga; g[n:2 * n] += gd; g[2 * n] += float((r * S / 2).sum())
        val += ridge * float(((a - a0) ** 2).sum() + ((d - d0) ** 2).sum()) + 5 * (h - 0.2) ** 2
        g[:n] += 2 * ridge * (a - a0); g[n:2 * n] += 2 * ridge * (d - d0); g[2 * n] += 10 * (h - 0.2)
        val += 20 * (a.mean() ** 2 + d.mean() ** 2)
        g[:n] += 40 * a.mean() / n; g[n:2 * n] += 40 * d.mean() / n
        return val, g

    th0 = np.concatenate([a0, d0, [0.2]])
    res = minimize(loss, th0, jac=True, method="L-BFGS-B")
    a, d, h = unpack(res.x)

    def lam(team, opp, home):
        return float(mu * math.exp(a[idx[team]] + d[idx[opp]] + (h / 2 if home else -h / 2)))

    return {"ids": ids, "idx": idx, "a": a, "d": d, "h": float(h), "mu": mu, "a0": a0, "d0": d0, "lam": lam,
            "n_obs": len(I) // 2}


def team_summary(boot, fixtures, xg, model):
    rows = {t["id"]: dict(id=t["id"], short=t["short_name"], name=t["name"], p=0, w=0, d=0, l=0, gf=0, ga=0, xgf=0.0, xga=0.0, form=[])
            for t in boot["teams"]}
    for f in sorted([f for f in fixtures if f["finished"] and f["event"]], key=lambda x: (x["event"], x["kickoff_time"] or "")):
        for me, op, gf, ga in ((f["team_h"], f["team_a"], f["team_h_score"], f["team_a_score"]),
                               (f["team_a"], f["team_h"], f["team_a_score"], f["team_h_score"])):
            r = rows[me]; r["p"] += 1; r["gf"] += gf; r["ga"] += ga
            r["xgf"] += xg.get((f["id"], me), 0.0); r["xga"] += xg.get((f["id"], op), 0.0)
            res = "W" if gf > ga else "D" if gf == ga else "L"
            r["w" if res == "W" else "d" if res == "D" else "l"] += 1; r["form"].append(res)
    idx = model["idx"]
    out = []
    for tid, r in rows.items():
        i = idx[tid]
        r["att"] = round(float(math.exp(model["a"][i])), 3); r["def"] = round(float(math.exp(model["d"][i])), 3)   # >1 = concedes more
        r["att0"] = round(float(math.exp(model["a0"][i])), 3); r["def0"] = round(float(math.exp(model["d0"][i])), 3)
        r["xgf"] = round(r["xgf"], 1); r["xga"] = round(r["xga"], 1); r["form"] = r["form"][-6:]
        out.append(r)
    # overall rating = log(att/def): higher is stronger. prior vs now
    for r in out:
        r["ovr"] = round(math.log(r["att"] / r["def"]), 3); r["ovr0"] = round(math.log(r["att0"] / r["def0"]), 3)
    for key, rk in (("ovr", "rank_now"), ("ovr0", "rank_prior")):
        for pos, r in enumerate(sorted(out, key=lambda x: -x[key]), 1):
            r[rk] = pos
    return out


# ------------------------------------------------------------------ players
def _poisson_floor_half(lam, cap=12):
    pm = math.exp(-lam); tot = 0.0
    for k in range(cap):
        tot += pm * (k // 2)
        pm *= lam / (k + 1)
    return tot


def avail_factor(e, k):
    st, ch = e["status"], e["chance_of_playing_next_round"]
    if st in ("u", "n"):
        return 0.0
    c = (25 if st == "i" else 100) if ch is None else ch
    f = c / 100.0
    if f >= 1:
        return 1.0
    if f <= 0:
        return 0.0 if k < 2 else 0.5 if k < 4 else 0.85
    return f + (1 - f) * min(1.0, k / 3.0)


def ban_probs(y, yc90, mm, seq):
    """Chance of serving a yellow-card ban in each upcoming match of `seq` [(gw, fixture, availability), ...].
    A player on y yellows who picks up enough more before the cut-off match misses the following match(es).
    Per-match chance of a yellow = 1 - exp(-rate per 90 * expected 90s); at most one yellow per match.
    Returns (ban chance per match, chance of being banned for at least one match inside the window)."""
    n = len(seq); ban = [0.0] * n; risk = 0.0
    for thr, last_gw, length in YC_RULES:
        left = thr - y
        if left <= 0 or left > 3:
            continue
        dist = [1.0] + [0.0] * (left - 1)             # P(extra yellows so far = index), absorbed once the ban is triggered
        for k, (gw, _, av) in enumerate(seq):
            if gw > last_gw:
                break
            q = 1.0 - math.exp(-yc90 * mm / 90.0 * av)
            reach = dist[left - 1] * q
            new = [0.0] * left
            for i in range(left):
                new[i] += dist[i] * (1 - q)
                if i:
                    new[i] += dist[i - 1] * q
            dist = new
            for j in range(k + 1, min(n, k + 1 + length)):
                ban[j] += reach
            if k + 1 < n:
                risk += reach
    return [min(1.0, b) for b in ban], min(1.0, risk)


def position_priors(boot, hist):
    pri = {}
    team_of = {e["id"]: e for e in boot["elements"]}
    for pt in (1, 2, 3, 4):
        acc = {k: [0.0, 0.0] for k in ("xg", "g", "xa", "a", "bn", "sv", "yc")}
        dcn = dcd = 0.0
        for pid, rows in hist.items():
            e = team_of[pid]
            if e["element_type"] != pt:
                continue
            for r in rows:
                m = r["minutes"]
                if m <= 0:
                    continue
                for key, col in (("xg", "expected_goals"), ("g", "goals_scored"), ("xa", "expected_assists"), ("a", "assists"),
                                 ("bn", "bonus"), ("sv", "saves"), ("yc", "yellow_cards")):
                    acc[key][0] += float(r[col] or 0); acc[key][1] += m / 90
                if m >= 45 and pt in DC_THR:
                    dcn += 1 if (r.get("defensive_contribution") or 0) >= DC_THR[pt] else 0; dcd += 1
        pri[pt] = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in acc.items()}
        pri[pt]["dc"] = dcn / dcd if dcd else 0.0
    return pri


def profile(rows, pt, pri):
    n = len(rows)
    age = np.arange(n - 1, -1, -1)
    w = 0.93 ** age
    wm = 0.85 ** age
    m = np.array([r["minutes"] for r in rows], float)
    m90 = float((w * m / 90).sum())
    K = 4.0

    def rate(col, prior):
        s = float((w * np.array([float(r.get(col) or 0) for r in rows])).sum())
        return (s + K * prior) / (m90 + K)

    p = pri[pt]
    out = {"xg": rate("expected_goals", p["xg"]), "g": rate("goals_scored", p["g"]),
           "xa": rate("expected_assists", p["xa"]), "a": rate("assists", p["a"]),
           "bn": rate("bonus", p["bn"]), "sv": rate("saves", p["sv"]), "yc": rate("yellow_cards", p["yc"])}
    sw = float(wm.sum()) or 1.0
    out["pm1"] = float((wm * (m > 0)).sum() / sw)
    out["p60"] = float((wm * (m >= 60)).sum() / sw)
    out["mm"] = float((wm * m).sum() / sw)
    if pt in DC_THR:
        sel = [(wmi, 1.0 if (r.get("defensive_contribution") or 0) >= DC_THR[pt] else 0.0)
               for wmi, r, mi in zip(wm, rows, m) if mi >= 45]
        sw2 = sum(x for x, _ in sel)
        out["dc"] = (sum(x * y for x, y in sel) + K * p["dc"]) / (sw2 + K)
    else:
        out["dc"] = 0.0
    return out, m90


def project(boot, fixtures, hist, model, prefs, horizon_gws=8):
    """Returns (players list for the app, team fixture grid)."""
    sc = boot["game_config"]["scoring"]
    cur = next((e["id"] for e in boot["events"] if e["is_current"]), 0)
    nxt = next((e["id"] for e in boot["events"] if e["is_next"]), cur + 1)
    gws = list(range(nxt, min(38, nxt + horizon_gws - 1) + 1))
    teams = {t["id"]: t for t in boot["teams"]}
    up = {t: [] for t in teams}
    for f in fixtures:
        if f["finished"] or not f["event"] or f["event"] not in gws:
            continue
        up[f["team_h"]].append({"gw": f["event"], "opp": f["team_a"], "home": True, "fid": f["id"]})
        up[f["team_a"]].append({"gw": f["event"], "opp": f["team_h"], "home": False, "fid": f["id"]})
    lam = model["lam"]
    for t, lst in up.items():
        for x in lst:
            x["lt"] = lam(t, x["opp"], x["home"]); x["lo"] = lam(x["opp"], t, not x["home"])
            x["pcs"] = math.exp(-x["lo"])
        lst.sort(key=lambda x: x["gw"])
    # reference level per team = its own average over matches already played (so a rate isn't double counted)
    ref_att, ref_def = {}, {}
    for t in teams:
        vals_a, vals_d = [], []
        for f in fixtures:
            if f["finished"] and f["event"] and t in (f["team_h"], f["team_a"]):
                home = f["team_h"] == t
                opp = f["team_a"] if home else f["team_h"]
                vals_a.append(lam(t, opp, home)); vals_d.append(lam(opp, t, not home))
        ref_att[t] = float(np.mean(vals_a)) if vals_a else model["mu"]
        ref_def[t] = float(np.mean(vals_d)) if vals_d else model["mu"]

    # difficulty buckets 1 (easy) .. 5 (hard) by quantile across all upcoming fixtures
    allx = [x for lst in up.values() for x in lst]
    def bucket(vals, higher_is_easier):
        arr = np.array(vals, float)
        order = arr.argsort().argsort() / max(1, len(arr) - 1)       # percentile 0..1
        if higher_is_easier:
            return [int(5 - min(4, int(p * 5))) for p in order]
        return [int(1 + min(4, int(p * 5))) for p in order]
    if allx:
        dA = bucket([x["lt"] for x in allx], True)
        dD = bucket([x["lo"] for x in allx], False)
        dO = bucket([math.log(x["lt"] / x["lo"]) for x in allx], True)
        for x, a, d, o in zip(allx, dA, dD, dO):
            x["dA"], x["dD"], x["dO"] = a, d, o

    pri = position_priors(boot, hist)
    players = []
    for e in boot["elements"]:
        rows = hist.get(e["id"])
        pt = e["element_type"]
        if not rows or e["minutes"] <= 0:
            continue
        pr, m90 = profile(rows, pt, pri)
        t = e["team"]
        goal_pts = sc["goals_scored"][POS_SHORT[pt]]
        cs_pts = sc["clean_sheets"][POS_SHORT[pt]]
        gc_pts = sc["goals_conceded"][POS_SHORT[pt]]
        dc_pts = sc["defensive_contribution"][POS_SHORT[pt]]
        seq = []
        for k_gw, gw in enumerate(gws):
            av = avail_factor(e, k_gw)
            for x in [x for x in up[t] if x["gw"] == gw]:
                seq.append((gw, x, av))
        ban, ban_risk = ban_probs(e.get("yellow_cards", 0), pr["yc"], pr["mm"], seq)
        fx = []
        for si, (gw, x, av) in enumerate(seq):
            if True:
                sT = x["lt"] / ref_att[t]; sO = x["lo"] / ref_def[t]
                exp90 = pr["mm"] / 90.0 * av
                p60 = pr["p60"] * av; p1 = pr["pm1"] * av
                ap = sc["short_play"] * max(0.0, p1 - p60) + sc["long_play"] * p60
                gl = (0.7 * pr["xg"] + 0.3 * pr["g"]) * exp90 * sT * goal_pts if pt != 1 else 0.0
                asst = (0.7 * pr["xa"] + 0.3 * pr["a"]) * exp90 * sT * sc["assists"] if pt != 1 else 0.0
                cs = cs_pts * p60 * x["pcs"]
                expo = min(1.0, (pr["mm"] / max(pr["pm1"], 0.05)) / 90.0)
                gc = gc_pts * p1 * _poisson_floor_half(x["lo"] * expo) if gc_pts else 0.0
                sv = (pr["sv"] * exp90 * (sO ** 0.9) * sc["saves"] / 3.0) if pt == 1 else 0.0
                bn = pr["bn"] * exp90 * (sT ** 0.4) * (sO ** -0.2) * sc["bonus"]
                dc = dc_pts * pr["dc"] * p60 * (sO ** 0.2) if dc_pts else 0.0
                cd = -pr["yc"] * exp90 * 1.0
                raw = ap + gl + asst + cs + gc + sv + bn + dc + cd
                tot = raw * (1.0 - ban[si])                      # banned matches score nothing
                fx.append([gw, x["opp"], int(x["home"]), round(tot, 2), round(ap, 2), round(gl, 2), round(asst, 2), round(cs, 2),
                           round(gc, 2), round(sv, 2), round(bn, 2), round(dc, 2), round(cd, 2),
                           x.get("dA", 3), x.get("dD", 3), x.get("dO", 3), round(x["lt"], 2), round(x["lo"], 2),
                           round(-ban[si] * raw, 2)])

        mins_played = e["minutes"]
        players.append({
            "id": e["id"], "n": e["web_name"], "full": f"{e['first_name']} {e['second_name']}", "t": t, "p": pt,
            "pr": e["now_cost"] / 10.0, "own": float(e["selected_by_percent"] or 0), "form": float(e["form"] or 0),
            "pts": e["total_points"], "mins": mins_played,
            "p90": round(e["total_points"] / (mins_played / 90.0), 2) if mins_played else 0.0,
            "xgi90": round(float(e.get("expected_goal_involvements_per_90") or 0), 2),
            "bn90": round(e["bonus"] / (mins_played / 90.0), 2) if mins_played else 0.0,
            "dcr": round(pr["dc"], 2), "p60": round(pr["p60"], 2), "pm1": round(pr["pm1"], 2),
            "news": e["news"] or "", "chance": e["chance_of_playing_next_round"], "st": e["status"],
            "pen": e.get("penalties_order"), "ep": float(e["ep_next"] or 0),
            "yc": e.get("yellow_cards", 0), "ycleft": next((thr - e.get("yellow_cards", 0) for thr, last, _ in YC_RULES
                                                           if 0 < thr - e.get("yellow_cards", 0) <= 2 and nxt <= last), None),
            "ban": round(ban_risk, 2),
            "tin": e.get("transfers_in_event", 0), "tout": e.get("transfers_out_event", 0), "cc": e.get("cost_change_event", 0),
            "tp": round(100.0 * (e.get("transfers_in_event", 0) - e.get("transfers_out_event", 0))
                        / max(1.0, float(e["selected_by_percent"] or 0) / 100.0 * boot.get("total_players", 1e7)), 2),
            "hist": [r["total_points"] for r in rows[-10:]], "histm": [r["minutes"] for r in rows[-10:]],
            "fx": fx,
        })
    grid = {t: [[x["gw"], x["opp"], int(x["home"]), round(x["lt"], 2), round(x["lo"], 2), round(x["pcs"], 2),
                 x.get("dA", 3), x.get("dD", 3), x.get("dO", 3)] for x in lst] for t, lst in up.items()}
    return players, grid, gws


def perf_scores(players, min_minutes=270):
    """Season performance index per position (percentiles of production stats)."""
    weights = {1: {"p90": .5, "bn90": .15, "sv": .15, "cs": .2}, 2: {"p90": .4, "bn90": .1, "xgi90": .15, "dcr": .15, "cs": .2},
               3: {"p90": .4, "bn90": .1, "xgi90": .35, "dcr": .15}, 4: {"p90": .4, "bn90": .1, "xgi90": .5}}
    out = {}
    for pt in (1, 2, 3, 4):
        grp = [p for p in players if p["p"] == pt and p["mins"] >= min_minutes]
        if not grp:
            continue
        for key in weights[pt]:
            if key in ("sv", "cs"):
                continue
            vals = np.array([p[key] for p in grp], float)
            order = vals.argsort().argsort() / max(1, len(vals) - 1) * 100
            for p, v in zip(grp, order):
                p.setdefault("_pct", {})[key] = float(v)
        for p in grp:
            pc = p["_pct"]; wsum = sum(weights[pt][k] for k in pc)
            out[p["id"]] = sum(weights[pt][k] * pc[k] for k in pc) / wsum if wsum else 50.0
    for p in players:
        p.pop("_pct", None)
        raw = out.get(p["id"])
        shrink = p["mins"] / (p["mins"] + min_minutes)
        p["perf"] = round(50 + ((raw if raw is not None else 50) - 50) * shrink, 1)
    return players


# ------------------------------------------------------------------ squad optimiser
def best_squad(players, horizon, budget=100.0, per_team=3, only_gws=None):
    """Maximise projected points of the starting XI (+ small bench weight) via integer programming.
    only_gws: optimise for exactly those gameweeks (e.g. one GW, for Free Hit) instead of the first `horizon`."""
    gws = list(only_gws) if only_gws else sorted({fx[0] for p in players for fx in p["fx"]})[:horizon]
    pool = [p for p in players if p["fx"]]
    xp = np.array([sum(fx[3] for fx in p["fx"] if fx[0] in gws) for p in pool])
    n = len(pool)
    c = -np.concatenate([0.05 * xp, 0.95 * xp])             # variables: x (in squad), y (starts)
    A, lo, hi = [], [], []
    def add(row, l, u):
        A.append(row); lo.append(l); hi.append(u)
    z = np.zeros(n)
    add(np.concatenate([np.ones(n), z]), 15, 15)
    add(np.concatenate([z, np.ones(n)]), 11, 11)
    add(np.concatenate([np.array([p["pr"] for p in pool]), z]), 0, budget)
    for pt, (sq, mn, mx) in {1: (2, 1, 1), 2: (5, 3, 5), 3: (5, 2, 5), 4: (3, 1, 3)}.items():
        ind = np.array([1.0 if p["p"] == pt else 0.0 for p in pool])
        add(np.concatenate([ind, z]), sq, sq)
        add(np.concatenate([z, ind]), mn, mx)
    for t in {p["t"] for p in pool}:
        ind = np.array([1.0 if p["t"] == t else 0.0 for p in pool])
        add(np.concatenate([ind, z]), 0, per_team)
    for i in range(n):                                        # y_i <= x_i
        row = np.zeros(2 * n); row[i] = -1; row[n + i] = 1
        add(row, -np.inf, 0)
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(2 * n), bounds=Bounds(0, 1))
    if res.x is None:
        return None
    x = np.round(res.x[:n]).astype(int); y = np.round(res.x[n:]).astype(int)
    squad = [pool[i]["id"] for i in range(n) if x[i]]
    xi = [pool[i]["id"] for i in range(n) if y[i]]
    cost = round(sum(pool[i]["pr"] for i in range(n) if x[i]), 1)
    xi_pts = round(sum(xp[i] for i in range(n) if y[i]), 1)
    return {"squad": squad, "xi": xi, "cost": cost, "xpts": xi_pts, "horizon": horizon}
