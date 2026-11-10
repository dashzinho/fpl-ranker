"""Fetching + caching of the official FPL API (free, no key).

bootstrap-static and fixtures are fetched every update (about 2 MB). Per-player histories
(/element-summary/{id}/) are only re-fetched for players whose minutes/points changed.
"""
import concurrent.futures as cf
import hashlib, json, os, time, urllib.request
from paths import DATA, PLAYERS

BASE = "https://fantasy.premierleague.com/api"
UA = {"User-Agent": "Mozilla/5.0 (personal fpl ranking tool)"}


def get(path, retries=3, timeout=40):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(BASE + path, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"FPL API request failed for {path}: {last}")


def fetch_core():
    boot = get("/bootstrap-static/")
    fixtures = get("/fixtures/")
    json.dump(boot, open(os.path.join(DATA, "bootstrap.json"), "w", encoding="utf-8"))
    json.dump(fixtures, open(os.path.join(DATA, "fixtures.json"), "w", encoding="utf-8"))
    return boot, fixtures


def load_core():
    boot = json.load(open(os.path.join(DATA, "bootstrap.json"), encoding="utf-8"))
    fixtures = json.load(open(os.path.join(DATA, "fixtures.json"), encoding="utf-8"))
    return boot, fixtures


def data_signature(boot, fixtures):
    """Fingerprint of everything that changes the rankings (not ownership %, which drifts daily)."""
    cur = next((e["id"] for e in boot["events"] if e["is_current"]), 0)
    nxt = next((e["id"] for e in boot["events"] if e["is_next"]), 0)
    h = hashlib.md5()
    h.update(f"{cur}|{nxt}".encode())
    for e in sorted(boot["elements"], key=lambda x: x["id"]):
        h.update(f"{e['id']},{e['minutes']},{e['total_points']},{e['now_cost']},{e['status']},{e['chance_of_playing_next_round']},{e['news']}|".encode())
    for f in fixtures:
        h.update(f"{f['id']},{f['finished']},{f['team_h_score']},{f['team_a_score']},{f['event']},{f['kickoff_time']}|".encode())
    return h.hexdigest()[:16]


def _cache_path(pid):
    return os.path.join(PLAYERS, f"{pid}.json")


def load_histories(boot):
    """All cached histories {player_id: [rows]} (no network)."""
    out = {}
    for e in boot["elements"]:
        p = _cache_path(e["id"])
        if os.path.exists(p):
            try:
                out[e["id"]] = json.load(open(p, encoding="utf-8"))["history"]
            except Exception:                        # noqa: BLE001
                pass
    return out


def sync_players(boot, workers=6, log=print):
    """Fetch histories only for players whose (minutes, total_points) differ from the cached copy."""
    todo = []
    for e in boot["elements"]:
        if e["minutes"] <= 0:
            continue
        sig = [e["minutes"], e["total_points"]]
        p = _cache_path(e["id"])
        if os.path.exists(p):
            try:
                if json.load(open(p, encoding="utf-8")).get("sig") == sig:
                    continue
            except Exception:                        # noqa: BLE001
                pass
        todo.append((e["id"], sig))
    if not todo:
        return 0

    def work(item):
        pid, sig = item
        d = get(f"/element-summary/{pid}/")
        json.dump({"sig": sig, "history": d.get("history", []), "history_past": d.get("history_past", [])},
                  open(_cache_path(pid), "w", encoding="utf-8"))
        return pid

    log(f"  fetching history for {len(todo)} player(s) (others unchanged, cached)...")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    return len(todo)


def fetch_my_team(team_id, gw):
    """Public squad of a manager for a gameweek (ids only; selling prices need a login, so now_cost is used)."""
    try:
        picks = get(f"/entry/{team_id}/event/{gw}/picks/")
        entry = get(f"/entry/{team_id}/")
        return {"picks": picks["picks"], "bank": picks["entry_history"]["bank"] / 10.0,
                "value": picks["entry_history"]["value"] / 10.0, "name": entry.get("name"), "gw": gw,
                "free_transfers_note": "Free transfers aren't exposed publicly; assume 1 unless you banked one."}
    except Exception as e:                           # noqa: BLE001
        return {"error": str(e)}
