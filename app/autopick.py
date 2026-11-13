"""Auto-pick: after a gameweek finishes, adopt the best-ranked TEAM MODEL set from the backtest as the new default.

What it changes: the four team-model settings (xg_weight, half_life_gw, prior_strength, prior_scale) and nothing else.
Your Score weights (outlook / perf / value) are never changed automatically: the Points measure scores the top picks
with no budget, so it always rewards ignoring price, and a walk-forward test (pick on earlier gameweeks, score on the
next) showed picking the mix did no better than keeping 60/25/15. The team model is different: it improved the
fixture-ranking measures out of sample and did not hurt Points.

"Best" = highest mean percentile of the four measures among the team-model sets, with Points averaged over every Score
mix so the answer does not depend on your weights (computed in backtest.py as meta["best_team"]).
It runs at most once per finished gameweek (data/auto_pick.json). Every pick is logged with the values it replaced, and
`revert` puts the previous values back and turns auto-pick off.

prefs.local.json -> "auto_pick": true/false (default true). A manual choice on the Model tab ("Use" / "Save my sliders
as default") turns it off so your choice is never overwritten; the switch on the page turns it back on and picks again at
the next run. "Pick best now" adopts the best once, right away, even with auto-pick off, and leaves the switch alone.
"""
import json, os, time
from paths import DATA

FILE = os.path.join(DATA, "auto_pick.json")
TEAM_KEYS = ("xg_weight", "half_life_gw", "prior_strength", "prior_scale")
TEAM_DEFAULT = {"xg_weight": 0.65, "half_life_gw": 12, "prior_strength": 5.0, "prior_scale": 0.10}
HISTORY_MAX = 20


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
    return [float(prefs.get(k, TEAM_DEFAULT[k])) for k in TEAM_KEYS]


def _as_prefs(vals):
    return {k: (int(v) if k == "half_life_gw" else float(v)) for k, v in zip(TEAM_KEYS, vals)}


def _log(st, entry):
    st["history"] = ((st.get("history") or []) + [entry])[-HISTORY_MAX:]


def apply(root, prefs, meta, log=print):
    """Adopt the best team-model set once per finished gameweek. Updates prefs in place; returns True if it changed."""
    if not meta or not meta.get("best_team"):
        return False
    st = _read(FILE)
    forced = bool(st.get("force"))
    if not forced and not (enabled(prefs) and st.get("gw", 0) < meta["finished"]):
        return False
    best = [float(v) for v in meta["best_team"]]
    cur = current(prefs)
    changed = best != cur
    if changed:
        new = _as_prefs(best)
        merge_local(root, new)
        prefs.update(new)
        log(f"  auto-pick (after GW{meta['finished']}): {', '.join(f'{k}={v}' for k, v in new.items())}")
    st = _read(FILE)
    st.update({"gw": meta["finished"], "chosen": best, "was": cur, "changed": changed, "at": time.strftime("%d %b %H:%M")})
    st.pop("force", None)
    _log(st, {"gw": meta["finished"], "at": st["at"], "was": cur, "chosen": best, "changed": changed, "manual": forced, "reverted": False})
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


def _revertable(st, prefs_vals):
    """The latest history entry, if it is a real change that has not been undone AND the settings still equal what it
    chose. Anything you changed by hand since is never overwritten."""
    hist = st.get("history") or []
    if not hist:
        return None, "nothing to revert"
    last = hist[-1]
    if last.get("reverted") or not last.get("changed"):
        return None, "nothing to revert"
    if [float(v) for v in last["chosen"]] != [float(v) for v in prefs_vals]:
        return None, "the settings were changed after the last pick, so there is nothing safe to revert"
    return last, None


def revert(root):
    """Put back the team-model values that the latest pick replaced, and turn auto-pick off so it stays put."""
    st = _read(FILE)
    cur = _read(os.path.join(root, "prefs.local.json"))
    last, why = _revertable(st, current(cur))
    if not last:
        raise ValueError(why)
    merge_local(root, {**_as_prefs(last["was"]), "auto_pick": False})
    last["reverted"] = True
    _log(st, {"gw": last["gw"], "at": time.strftime("%d %b %H:%M"), "was": last["chosen"], "chosen": last["was"], "changed": True, "manual": True, "reverted": True})
    _write(FILE, st)
    return _as_prefs(last["was"])


def status(prefs):
    st = _read(FILE)
    hist = st.get("history") or []
    return {"on": enabled(prefs), **{k: st.get(k) for k in ("gw", "changed", "at", "was")}, "history": hist[-5:][::-1],
            "can_revert": _revertable(st, current(prefs))[0] is not None}
