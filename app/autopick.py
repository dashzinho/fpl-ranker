"""Auto-pick: after a gameweek finishes, adopt the best-ranked weight set from the backtest as the new default.

One group: a set = the four team-model weights + the Score weights (outlook / perf / value). "Best" = highest overall
score, the mean percentile across the four measures (points of the top picks, goal difference, xG difference, goals
likelihood), so no single measure decides it. It runs at most once per finished gameweek (record in data/auto_pick.json).

prefs.local.json -> "auto_pick": true/false (default true). A manual choice on the Model tab ("Use" / "Save my sliders
as default") turns it off so your choice is never overwritten; the switch on the page turns it back on and picks again
at the next run. "Pick best now" adopts the best once, right away, even with auto-pick off, and leaves the switch alone.
Caveat shown on the page: the best of thousands of sets on a few gameweeks is partly luck, so the pick can move around.
"""
import json, os, time
from paths import DATA

FILE = os.path.join(DATA, "auto_pick.json")
TEAM_KEYS = ("xg_weight", "half_life_gw", "prior_strength", "prior_scale")


def enabled(prefs):
    return bool(prefs.get("auto_pick", True))


def _read(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)


def merge_local(root, updates):
    """Merge top-level keys into prefs.local.json atomically."""
    path = os.path.join(root, "prefs.local.json")
    cur = _read(path)
    cur.update(updates)
    _write(path, cur)


def current(prefs):
    w = prefs.get("weights") or {"outlook": 60, "perf": 25, "value": 15}
    d = {"xg_weight": 0.65, "half_life_gw": 12, "prior_strength": 5.0, "prior_scale": 0.10}
    return [float(prefs.get(k, d[k])) for k in TEAM_KEYS] + [float(w["outlook"]), float(w["perf"]), float(w["value"])]


def apply(root, prefs, meta, log=print):
    """Adopt the best set once per finished gameweek. Updates prefs in place; returns True if the set changed."""
    if not meta:
        return False
    st = _read(FILE)
    forced = bool(st.get("force"))
    if not forced and not (enabled(prefs) and st.get("gw", 0) < meta["finished"]):
        return False
    best = [float(v) for v in meta["best"]["ov"]]
    cur = current(prefs)
    changed = best != cur
    if changed:
        team = {k: (int(v) if k == "half_life_gw" else float(v)) for k, v in zip(TEAM_KEYS, best[:4])}
        weights = {"outlook": int(best[4]), "perf": int(best[5]), "value": int(best[6])}
        merge_local(root, {**team, "weights": weights})
        prefs.update(team); prefs["weights"] = weights
        log(f"  auto-pick (after GW{meta['finished']}): {', '.join(f'{k}={v}' for k, v in team.items())}; "
            f"Score mix {weights['outlook']}/{weights['perf']}/{weights['value']}")
    st = _read(FILE)
    st.update({"gw": meta["finished"], "chosen": best, "was": cur, "changed": changed, "at": time.strftime("%d %b %H:%M")})
    st.pop("force", None)
    _write(FILE, st)
    return changed


def set_flag(root, on):
    """Turn auto-pick on/off from the page. Turning on forgets the last pick so the next run picks again."""
    merge_local(root, {"auto_pick": bool(on)})
    if on:
        st = _read(FILE)
        st.pop("gw", None)
        _write(FILE, st)
    return bool(on)


def pick_now(root):
    """Manual button: adopt the best on the next rebuild, once. Leaves the on/off switch alone."""
    st = _read(FILE)
    st["force"] = True
    _write(FILE, st)


def status(prefs):
    st = _read(FILE)
    return {"on": enabled(prefs), **{k: st.get(k) for k in ("gw", "changed", "at", "was")}}
