"""Assemble all model output into one JSON blob and inject it into the HTML template."""
import datetime, json, os
import fpl_data as fd
import fpl_model as fm
from paths import APP


def build_data(boot, fixtures, hist, prefs, my_team=None):
    cur = next((e["id"] for e in boot["events"] if e["is_current"]), 0)
    nxt = next((e["id"] for e in boot["events"] if e["is_next"]), cur + 1)
    xg = fm.team_match_table(boot, fixtures, hist)
    model = fm.fit_teams(boot, fixtures, xg, cur, prefs)
    teams = fm.team_summary(boot, fixtures, xg, model)
    players, grid, gws = fm.project(boot, fixtures, hist, model, prefs, horizon_gws=prefs.get("horizon_max", 8))
    players = fm.perf_scores(players, prefs.get("perf_min_minutes", 270))
    keep = [p for p in players if p["fx"] or p["own"] >= 0.5]
    squads = {}
    for h in (1, 3, 5, 8):
        try:
            s = fm.best_squad(keep, h, prefs.get("budget", 100.0), 3)
            if s:
                squads[str(h)] = s
        except Exception as e:                       # noqa: BLE001
            squads[str(h)] = {"error": str(e)}
    mine = None
    if my_team and "error" not in my_team:
        mine = {"name": my_team.get("name"), "gw": my_team.get("gw"), "bank": my_team["bank"], "value": my_team["value"],
                "picks": [{"id": p["element"], "pos": p["position"], "cap": p["is_captain"], "vc": p["is_vice_captain"], "mult": p["multiplier"]}
                          for p in my_team["picks"]]}
    elif my_team:
        mine = {"error": my_team["error"]}
    return {
        "meta": {"gw_now": cur, "gw_next": nxt, "gws": gws, "generated": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
                 "season": boot["events"][0]["deadline_time"][:4], "n_players": len(keep), "n_matches": model["n_obs"],
                 "home_adv": round(model["h"], 3), "mu": round(model["mu"], 3), "prior_strength": prefs.get("prior_strength", 5.0),
                 "horizon": prefs.get("default_horizon", 5), "budget": prefs.get("budget", 100.0)},
        "weights": prefs.get("weights", {"outlook": 60, "perf": 25, "value": 15}),
        "teams": teams, "grid": {str(k): v for k, v in grid.items()},
        "players": keep, "squads": squads, "mine": mine,
    }


def write_html(data, out_path):
    tpl = open(os.path.join(APP, "board_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open(out_path, "w", encoding="utf-8").write(html)
    return os.path.getsize(out_path)
