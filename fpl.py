#!/usr/bin/env python
"""FPL Rankings - one command.

    python fpl.py            first run: sets everything up and builds the rankings page.
                             later runs: update mode (only fetches/recomputes what changed).
    python fpl.py --open     also open the page in your browser
    python fpl.py --force    re-fetch and rebuild even if nothing changed
    python fpl.py --offline  rebuild from cached data only (no network)

Layout (created on first run):
    prefs.json     preferences (look-ahead, weights, budget, optional fpl_team_id, model knobs)
    app/           code + HTML template
    data/          cached API data (bootstrap, fixtures, one file per player) + state.json
    output/        fpl_rankings.html (all data embedded, opens offline)

Update logic (keeps it cheap): bootstrap + fixtures are ~2 MB; per-player histories are only re-fetched for
players whose minutes/points changed; if the data fingerprint is unchanged nothing is rebuilt; if the last
check was < min_refresh_minutes ago no network call is made at all.
"""
import argparse, hashlib, json, os, subprocess, sys, time, webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
APP, DATA, OUT = (os.path.join(ROOT, d) for d in ("app", "data", "output"))
STATE = os.path.join(DATA, "state.json")
HTML = os.path.join(OUT, "fpl_rankings.html")
DEPS = {"numpy": "numpy", "scipy": "scipy"}
DEFAULT_PREFS = {
    "default_horizon": 5, "horizon_max": 8, "weights": {"outlook": 60, "perf": 25, "value": 15},
    "budget": 100.0, "fpl_team_id": None, "min_refresh_minutes": 30,
    "prior_strength": 5.0, "prior_scale": 0.10, "half_life_gw": 12, "perf_min_minutes": 270,
}


def log(m): print(m, flush=True)


def sig_code():
    h = hashlib.md5()
    for base, _, files in os.walk(APP):
        for f in sorted(files):
            if f.endswith((".py", ".html")):
                h.update(open(os.path.join(base, f), "rb").read())
    return h.hexdigest()[:12]


def sig_prefs():
    p = os.path.join(ROOT, "prefs.json")
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:12] if os.path.exists(p) else ""


def load_state():
    try: return json.load(open(STATE, encoding="utf-8"))
    except Exception: return {}


def save_state(s):
    json.dump(s, open(STATE, "w", encoding="utf-8"), indent=2)


def setup():
    log("First run: setting up the FPL rankings folder...")
    for d in (APP, DATA, os.path.join(DATA, "players"), OUT): os.makedirs(d, exist_ok=True)
    missing = []
    for mod, pkg in DEPS.items():
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        log(f"  installing: {', '.join(missing)}")
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing, "--disable-pip-version-check"]).returncode:
            sys.exit("pip install failed; run: pip install " + " ".join(missing))
    pp = os.path.join(ROOT, "prefs.json")
    if not os.path.exists(pp):
        json.dump(DEFAULT_PREFS, open(pp, "w", encoding="utf-8"), indent=2)
        log("  wrote default prefs.json")
    log("  setup complete")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    state = load_state()
    first = not state.get("setup_done") or not os.path.exists(os.path.join(ROOT, "prefs.json"))
    if first: setup()
    sys.path.insert(0, APP)
    import fpl_data as fd, fpl_report as fr

    prefs = dict(DEFAULT_PREFS); prefs.update(json.load(open(os.path.join(ROOT, "prefs.json"), encoding="utf-8")))
    s_code, s_prefs = sig_code(), sig_prefs()
    now = time.time()
    unchanged_code = state.get("code") == s_code and state.get("prefs") == s_prefs and os.path.exists(HTML)

    def done(msg):
        log(msg); log(f"Page: {HTML}")
        if a.open: webbrowser.open("file:///" + HTML.replace("\\", "/"))

    # 1) very recent check -> no network at all
    recent = now - state.get("checked_ts", 0) < prefs["min_refresh_minutes"] * 60
    if unchanged_code and recent and not a.force and not first:
        return done(f"Up to date (checked {state.get('checked_at')}, GW{state.get('gw_next')} next). No network used.")

    # 2) fetch the two core files (or reuse cache when offline)
    if a.offline:
        boot, fixtures = fd.load_core()
    else:
        log("Update mode: checking the FPL API..." if not first else "Fetching FPL data...")
        boot, fixtures = fd.fetch_core()
    sig = fd.data_signature(boot, fixtures)
    if unchanged_code and state.get("data_sig") == sig and not a.force and not first:
        state.update({"checked_ts": now, "checked_at": time.strftime("%Y-%m-%d %H:%M")}); save_state(state)
        return done("Up to date: no new results, prices or injury news since the last build.")

    # 3) fetch only changed player histories, then rebuild
    t0 = time.time()
    if not a.offline:
        fd.sync_players(boot, log=log)
    hist = fd.load_histories(boot)
    my = None
    if prefs.get("fpl_team_id") and not a.offline:
        cur = next((e["id"] for e in boot["events"] if e["is_current"]), 1)
        my = fd.fetch_my_team(prefs["fpl_team_id"], cur)
    data = fr.build_data(boot, fixtures, hist, prefs, my)
    size = fr.write_html(data, HTML)
    nxt = data["meta"]["gw_next"]
    state.update({"setup_done": True, "code": s_code, "prefs": s_prefs, "data_sig": sig, "gw_next": nxt,
                  "checked_ts": now, "checked_at": time.strftime("%Y-%m-%d %H:%M"), "built_at": time.strftime("%Y-%m-%d %H:%M"),
                  "players": len(hist), "seconds": round(time.time() - t0, 1)})
    save_state(state)
    done(f"Built {HTML} ({size/1e3:.0f} KB, {len(hist)} players, GW{nxt} next) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
