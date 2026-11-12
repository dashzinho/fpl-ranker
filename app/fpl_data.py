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


def _free_transfers(hist_current, next_gw):
    """Replay the 1/wk, bank-up-to-5 rule from public history (chip weeks: wildcard/free hit don't use or reset FTs)."""
    ft = 1
    for gw in range(2, next_gw + 1):
        ft = min(ft + 1, 5) if gw > 2 else 1
        if gw == next_gw:
            break
        r = next((x for x in hist_current if x["event"] == gw), None)
        if r:
            ft = max(ft - r["event_transfers"], 0) if r["event_transfers_cost"] == 0 else 0
    return ft


def _sell_price(now_cost, buy, bought):
    """FPL rule: sell = buy + floor(half the rise). Falls back to current price if the buy price is unknown."""
    if buy is None or not bought or now_cost <= buy:
        return now_cost                                  # never bought by you, or the price has fallen: sells at today's price
    return buy + (now_cost - buy) // 2                   # a rise: you keep half, rounded down to 0.1


def fetch_my_team(team_id, gw, boot=None):
    """Manager's squad plus chips, transfers and selling prices.
    Public API gives history/chips/transfers. If data/my_team_login.json exists (captured from a logged-in browser
    session via /api/my-team/) it overrides squad, exact selling prices, chip availability and pending transfers."""
    try:
        picks = get(f"/entry/{team_id}/event/{gw}/picks/")
        entry = get(f"/entry/{team_id}/")
        hist = get(f"/entry/{team_id}/history/")
        trs = get(f"/entry/{team_id}/transfers/")
        eh = picks["entry_history"]
        out = {"picks": picks["picks"], "bank": eh["bank"] / 10.0, "value": eh["value"] / 10.0, "name": entry.get("name"),
               "gw": gw, "source": "public",
               "chips_used": [{"name": c["name"], "gw": c["event"]} for c in hist.get("chips", [])],
               "history": [{"gw": x["event"], "pts": x["points"], "transfers": x["event_transfers"], "cost": x["event_transfers_cost"]}
                           for x in hist["current"]],
               "free_transfers": _free_transfers(hist["current"], gw + 1), "free_transfers_exact": False,
               "pending": None, "chips_left": None, "chip_window_ends_gw": None, "captured": None, "free_total": None}
        # selling prices from transfer history (bought price) vs current price
        if boot:
            cost = {e["id"]: e["now_cost"] for e in boot["elements"]}
            bought = {}
            for t in sorted(trs, key=lambda x: x["time"]):
                bought[t["element_in"]] = t["element_in_cost"]
            for p in out["picks"]:
                nc = cost.get(p["element"])
                p["selling_price"] = _sell_price(nc, bought.get(p["element"]), p["element"] in bought) / 10.0 if nc else None
        sn = load_snapshot(team_id)
        if sn:
            t = sn["transfers"]
            made, limit, per = t.get("made", 0), t.get("limit"), t.get("cost", 4)
            free_left = 99 if limit is None else max(limit - made, 0)
            hit = 0 if limit is None else max(made - limit, 0) * per
            out["picks"] = sn["picks"]
            out.update({"source": "login snapshot", "captured": sn.get("captured"), "bank": t["bank"], "value": t["value"],
                        "gw": sn["for_gw"], "pending": {"made": made, "hit": hit}, "free_transfers": free_left,
                        "free_total": limit, "free_transfers_exact": True,
                        "chips_left": [c["name"] for c in sn["chips"] if c["status"] == "available"],
                        "chip_window_ends_gw": sn.get("chip_window_ends_gw")})
        return out
    except Exception as e:                           # noqa: BLE001
        return {"error": str(e)}


# ------------------------------------------------------------------ login snapshot (exact squad/chips/transfers)
SNAP = os.path.join(DATA, "my_team_login.json")


def load_snapshot(team_id=None):
    try:
        sn = json.load(open(SNAP, encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return None
    return sn if (team_id is None or sn.get("team_id") == team_id) else None


def snapshot_from_raw(team_id, raw, for_gw):
    """Convert the logged-in /api/my-team/{id}/ response into the snapshot the app reads.
    In that response transfers.limit = free transfers available, made = made this GW, cost = points per EXTRA transfer."""
    t = raw["transfers"]
    chips = []
    for c in raw.get("chips", []):
        played = (c.get("played_by_entry") or [None])[0]
        chips.append({"name": c["name"], "status": c.get("status_for_entry", "available"), "gw": played, "stop_event": c.get("stop_event")})
    stops = [c["stop_event"] for c in chips if c["status"] == "available" and c.get("stop_event")]
    return {"team_id": team_id, "for_gw": for_gw, "captured": time.strftime("%Y-%m-%d %H:%M"),
            "picks_last_updated": raw.get("picks_last_updated"),
            "picks": [{"element": p["element"], "position": p["position"], "selling_price": p["selling_price"] / 10.0,
                       "purchase_price": p["purchase_price"] / 10.0, "multiplier": p["multiplier"],
                       "is_captain": p["is_captain"], "is_vice_captain": p["is_vice_captain"]} for p in raw["picks"]],
            "chips": chips, "chip_window_ends_gw": min(stops) if stops else None,
            "transfers": {"cost": t.get("cost", 4), "limit": t.get("limit"), "made": t.get("made", 0),
                          "bank": t["bank"] / 10.0, "value": t["value"] / 10.0, "status": t.get("status")}}


def write_snapshot(team_id, raw, for_gw):
    sn = snapshot_from_raw(team_id, raw, for_gw)
    os.makedirs(DATA, exist_ok=True)
    json.dump(sn, open(SNAP, "w", encoding="utf-8"), indent=1)
    return sn
