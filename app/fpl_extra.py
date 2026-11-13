"""Extras: price-change history and the AI-insights layer (export context / validate result).

AI layer (see .claude/skills/fpl-ai-insights/SKILL.md):
  python builds data/ai_context.json  ->  a Claude Code skill reads it and writes data/ai_insights.json
  ->  load_insights() validates every field (ids exist, numbers clamped, evidence must cite a headline that is in the
  context, text sanitised) and the page embeds the validated result. Adjustments are opt-in in the page.
"""
import datetime, json, os, re, time
import fpl_data as fd
from paths import DATA

PRICE_FILE = os.path.join(DATA, "price_history.json")
CONTEXT_FILE = os.path.join(DATA, "ai_context.json")
INSIGHTS_FILE = os.path.join(DATA, "ai_insights.json")


# ------------------------------------------------------------------ price history (observed rises/falls)
def update_price_history(boot, keep_days=21):
    """Append a snapshot of prices and net transfers; return observed price changes and net-transfer momentum."""
    try:
        h = json.load(open(PRICE_FILE, encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        h = {"snaps": []}
    now = int(time.time())
    cur = {e["id"]: [e["now_cost"], e.get("transfers_in_event", 0), e.get("transfers_out_event", 0)] for e in boot["elements"]}
    snaps = h["snaps"]
    ev = next((e["id"] for e in boot["events"] if e["is_next"]), 0)
    last = snaps[-1] if snaps else None
    changed = last and any(last["p"].get(str(i), [0])[0] != c[0] for i, c in cur.items() if str(i) in last["p"])
    if not last or now - last["ts"] >= 20 * 60 or changed:
        snaps.append({"ts": now, "gw": ev, "p": {str(i): c for i, c in cur.items()}})
    h["snaps"] = [s for s in snaps if now - s["ts"] <= keep_days * 86400][-300:]
    json.dump(h, open(PRICE_FILE, "w"), separators=(",", ":"))
    changes = []
    for a, b in zip(h["snaps"], h["snaps"][1:]):
        for k, v in b["p"].items():
            o = a["p"].get(k)
            if o and o[0] != v[0]:
                changes.append({"id": int(k), "from": o[0] / 10.0, "to": v[0] / 10.0, "ts": b["ts"]})
    changes.sort(key=lambda x: -x["ts"])
    mom = {}
    if len(h["snaps"]) >= 2:
        base = next((s for s in reversed(h["snaps"][:-1]) if h["snaps"][-1]["ts"] - s["ts"] >= 3 * 3600 and s["gw"] == ev), None)
        if base:
            hrs = max(1.0, (h["snaps"][-1]["ts"] - base["ts"]) / 3600.0)
            for k, v in h["snaps"][-1]["p"].items():
                o = base["p"].get(k)
                if o:
                    d = (v[1] - v[2]) - (o[1] - o[2])
                    if abs(d) >= 2000:
                        mom[k] = round(d / hrs)         # net transfers per hour since that snapshot
    return {"changes": changes[:40], "momentum": mom, "snapshots": len(h["snaps"]),
            "since": h["snaps"][0]["ts"] if h["snaps"] else now}


# ------------------------------------------------------------------ saving the Score weights as the default
def save_weights(root, body):
    """Validate {wo, wp, wv} (integers 0-100, not all zero) and merge them into prefs.local.json. Returns the saved dict."""
    try:
        w = {k: int(body[k]) for k in ("wo", "wp", "wv")}
    except (KeyError, TypeError, ValueError):
        raise ValueError("weights must be three integers")
    if any(v < 0 or v > 100 for v in w.values()) or sum(w.values()) == 0:
        raise ValueError("weights must be between 0 and 100 and not all zero")
    path = os.path.join(root, "prefs.local.json")
    try:
        cur = json.load(open(path, encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        cur = {}
    cur["weights"] = {"outlook": w["wo"], "perf": w["wp"], "value": w["wv"]}
    cur["auto_pick"] = False                                                   # a manual choice stops auto-picking
    tmp = path + ".tmp"
    json.dump(cur, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)
    return cur["weights"]


# ------------------------------------------------------------------ saving a team-model weight set as the default
TM_LIMITS = {"xg_weight": (0.0, 1.0), "half_life_gw": (1.0, 200.0), "prior_strength": (0.0, 100.0), "prior_scale": (0.0, 0.5)}


def save_model_set(root, body):
    """Validate the four team-model knobs and merge them into prefs.local.json. Returns the saved dict."""
    out = {}
    for k, (lo, hi) in TM_LIMITS.items():
        try:
            v = float(body[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{k} must be a number")
        if not (lo <= v <= hi) or v != v:
            raise ValueError(f"{k} must be between {lo} and {hi}")
        out[k] = int(v) if k == "half_life_gw" and v == int(v) else v
    path = os.path.join(root, "prefs.local.json")
    try:
        cur = json.load(open(path, encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        cur = {}
    cur.update(out)
    if "wo" in body:                                    # a full set: team-model weights + Score weights
        w = {k: int(body[k]) for k in ("wo", "wp", "wv")}
        if any(v < 0 or v > 100 for v in w.values()) or sum(w.values()) != 100:
            raise ValueError("score weights must be between 0 and 100 and sum to 100")
        cur["weights"] = {"outlook": w["wo"], "perf": w["wp"], "value": w["wv"]}
        out.update(w)
    cur["auto_pick"] = False                            # a manual choice stops auto-picking
    tmp = path + ".tmp"
    json.dump(cur, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)
    return out


# ------------------------------------------------------------------ AI layer
def export_context(data, boot, news):
    """Compact, self-contained JSON for the skill. Headlines are numbered so evidence can be checked."""
    P = {p["id"]: p for p in data["players"]}
    T = {t["id"]: t["short"] for t in data["teams"]}
    gws = data["meta"]["gws"]
    nxt = data["meta"]["gw_next"]

    def card(p):
        fx = [[f[0], T.get(f[1]), "H" if f[2] else "A", f[3]] for f in p["fx"] if f[0] in gws[:5]]
        return {"id": p["id"], "name": p["n"], "team": T.get(p["t"]), "pos": ["", "GKP", "DEF", "MID", "FWD"][p["p"]], "price": p["pr"],
                "own_pct": p["own"], "form": p["form"], "status": p["st"], "chance_next": p["chance"], "fpl_news": p["news"][:200],
                "yellow_cards": p.get("yc"), "ban_risk": p.get("ban"), "minutes_last5": (p.get("histm") or [])[-5:],
                "points_last5": (p.get("hist") or [])[-5:], "xpts_by_gw": fx}

    mine = data.get("mine") or {}
    my = []
    for k in (mine.get("picks") or []):
        p = P.get(k["id"])
        if p:
            c = card(p); c.update({"squad_slot": k["pos"], "captain": k["cap"], "sell_price": k.get("sell")}); my.append(c)
    owned = {k["id"] for k in (mine.get("picks") or [])}
    cands = []
    for pt in (1, 2, 3, 4):
        top = sorted([p for p in data["players"] if p["p"] == pt and p["id"] not in owned and p["fx"]],
                     key=lambda p: -sum(f[3] for f in p["fx"][:5]))[:10]
        cands += [card(p) for p in top]
    items = []
    for i, it in enumerate((news or {}).get("items", [])[:60]):
        items.append({"ref": f"n{i + 1}", "source": it["src"], "title": it["title"], "kind": it["kind"], "player_ids": it["ids"],
                      "age_hours": round((time.time() - it["ts"]) / 3600) if it.get("ts") else None})
    ctx = {"generated": datetime.datetime.now().isoformat(timespec="minutes"), "next_gameweek": nxt, "gameweeks_projected": gws,
           "squad": my, "bank": mine.get("bank"), "free_transfers": mine.get("free_transfers"),
           "chips_available": mine.get("chips_left"), "candidate_players": cands, "headlines": items,
           "headline_refs": [i["ref"] for i in items]}
    os.makedirs(DATA, exist_ok=True)
    json.dump(ctx, open(CONTEXT_FILE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return {"refs": set(ctx["headline_refs"]), "n_items": len(items)}


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _txt(v, n=400):
    return _CTRL.sub("", str(v or ""))[:n].strip()


def load_insights(boot, ctx_refs):
    """Validated insights or None. Anything that does not check out is dropped and counted."""
    try:
        raw = json.load(open(INSIGHTS_FILE, encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return None
    ids = {e["id"] for e in boot["elements"]}
    names = {e["id"]: e["web_name"] for e in boot["elements"]}
    nxt = next((e["id"] for e in boot["events"] if e["is_next"]), 1)
    dropped = []

    def pid(x):
        try:
            x = int(x)
        except (TypeError, ValueError):
            return None
        return x if x in ids else None

    def ev(lst):
        out = []
        for r in (lst or [])[:6]:
            r = _txt(r, 40)
            if r in ctx_refs or r == "fpl_status":
                out.append(r)
        return out

    adj = []
    for a in (raw.get("adjustments") or [])[:25]:
        p = pid(a.get("player_id")); e = ev(a.get("evidence"))
        try:
            m = float(a.get("multiplier"))
        except (TypeError, ValueError):
            m = None
        if p is None or m is None or not e:
            dropped.append(f"adjustment {a.get('player_id')}: missing id, multiplier or valid evidence"); continue
        m = min(1.2, max(0.2, m))
        try:
            until = int(a.get("until_gw") or nxt)
        except (TypeError, ValueError):
            until = nxt
        adj.append({"id": p, "mult": round(m, 2), "until": min(38, max(nxt, until)), "reason": _txt(a.get("reason")), "evidence": e,
                    "conf": a.get("confidence") if a.get("confidence") in ("low", "medium", "high") else "low"})
    tr = []
    for t in (raw.get("transfers") or [])[:6]:
        o, i = pid(t.get("out")), pid(t.get("in"))
        if o and i and o != i:
            tr.append({"out": o, "in": i, "reason": _txt(t.get("reason")), "evidence": ev(t.get("evidence"))})
        else:
            dropped.append("transfer with unknown player")
    cap = raw.get("captain") or {}
    captain = None
    if pid(cap.get("player_id")):
        captain = {"id": pid(cap["player_id"]), "alts": [x for x in (pid(y) for y in (cap.get("alternatives") or [])[:3]) if x],
                   "reason": _txt(cap.get("reason"))}
    ch = raw.get("chips") or {}
    chips = {"rec": ch.get("recommendation") if ch.get("recommendation") in ("none", "freehit", "wildcard", "bboost", "3xc") else "none",
             "gw": int(ch["gw"]) if str(ch.get("gw") or "").isdigit() else None, "reason": _txt(ch.get("reason"))}
    watch = [{"id": pid(w.get("player_id")), "note": _txt(w.get("note"), 240)} for w in (raw.get("watch") or [])[:10] if pid(w.get("player_id"))]
    gen = datetime.datetime.fromtimestamp(os.path.getmtime(INSIGHTS_FILE)).strftime("%d %b %H:%M")   # never trust a model-written time
    return {"generated": gen, "for_gw": raw.get("for_gw"), "summary": _txt(raw.get("summary"), 700),
            "adjustments": adj, "transfers": tr, "captain": captain, "chips": chips, "watch": watch,
            "dropped": dropped[:10], "names": {str(k): names[k] for k in {a["id"] for a in adj}}}
