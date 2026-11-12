"""News layer: public RSS headlines matched to FPL players/teams. No keys, no scraping behind logins.

Sources are plain RSS/Atom feeds (BBC Sport, Guardian, Sky Sports, Fantasy Football Scout, r/FantasyPL). Only the
headline, link, source and time are kept (no article text). Items are matched to players by name and flagged when
the headline looks like injury / team-news / transfer talk. Cached in data/news.json for `ttl_min` minutes.

Honest limits: RSS carries headlines, not private leaks; matching is by name so very common surnames can be wrong
(the page shows the matched names so you can see what it thought).
"""
import hashlib, html, json, os, re, time, urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from paths import DATA

FEEDS = [
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml"),
    ("Guardian", "https://www.theguardian.com/football/premierleague/rss"),
    ("Sky Sports", "https://www.skysports.com/rss/12040"),
    ("FF Scout", "https://www.fantasyfootballscout.co.uk/feed/"),
    ("r/FantasyPL", "https://www.reddit.com/r/FantasyPL/.rss"),
]
UA = "fpl-ranker/1.0 (personal hobby tool)"
CACHE = os.path.join(DATA, "news.json")
ATOM = "{http://www.w3.org/2005/Atom}"
HEALTH = re.compile(r"injur|doubt|ruled out|fitness|scan|hamstring|knock|sidelined|suspend|ban\b|return|back in training|"
                    r"availab|setback|absence|miss(es|ing)?\b|illness|surgery|recover", re.I)
TEAMNEWS = re.compile(r"team news|line-?up|starting xi|press conference|press conf|expected|predicted|rotation|"
                      r"rested|captain|differential|wildcard|price (rise|fall|change)|deadline|chip", re.I)
TRANSFER = re.compile(r"transfer|signs?\b|signing|loan|bid\b|deal\b|join(s|ed)?\b|leave(s)?\b|exit", re.I)
TAG_RE = re.compile(r"<[^>]+>")
# not men's Premier League football: women's football and other sports are dropped (checked on title and link)
EXCLUDE = re.compile(
    r"\bwomen'?s?\b|\bwomens\b|/women|\bwsl\b|\bnwsl\b|lioness|\bfemale\b|\bgirls\b|\bladies\b|"
    r"\bdarts\b|\bcricket\b|\brugby\b|formula[ -]?1|\bf1\b|\btennis\b|\bgolf\b|\bboxing\b|\bnfl\b|"
    r"\bnba\b|\bufc\b|\bcycling\b|\bsnooker\b|\bnetball\b|\bathletics\b", re.I)


def _fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/atom+xml, */*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _ts(s):
    if not s:
        return 0
    try:
        return int(parsedate_to_datetime(s).timestamp())
    except Exception:                                   # noqa: BLE001
        try:                                            # ISO 8601 (Atom)
            import datetime
            return int(datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
        except Exception:                               # noqa: BLE001
            return 0


def _parse(name, raw):
    root = ET.fromstring(raw)
    out = []
    for it in root.findall(".//item"):
        out.append((it.findtext("title"), it.findtext("link"), _ts(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"))))
    for it in root.findall(f".//{ATOM}entry"):
        link = it.find(f"{ATOM}link")
        out.append((it.findtext(f"{ATOM}title"), link.get("href") if link is not None else "",
                    _ts(it.findtext(f"{ATOM}updated") or it.findtext(f"{ATOM}published"))))
    res = []
    for title, link, ts in out:
        title = html.unescape(TAG_RE.sub("", title or "")).strip()
        if title and link and link.startswith(("http://", "https://")) and not EXCLUDE.search(title + " " + link):
            res.append({"src": name, "title": title[:220], "link": link.strip(), "ts": ts})
    return res


def _index(boot):
    """lower-case name -> player id (highest ownership wins); teams by name/short."""
    names = {}
    for e in boot["elements"]:
        own = float(e["selected_by_percent"] or 0)
        if own < 0.3 and e["minutes"] < 200:
            continue
        for nm in {e["web_name"], e["second_name"]}:
            nm = (nm or "").strip()
            if len(nm) >= 4 and " " not in nm:
                k = nm.lower()
                if k not in names or own > names[k][1]:
                    names[k] = (e["id"], own)
    teams = {}
    for t in boot["teams"]:
        teams[t["name"].lower()] = t["id"]
    return {k: v[0] for k, v in names.items()}, teams


def _match(title, names, teams):
    low = title.lower()
    ids, tids = [], []
    for w in set(re.findall(r"[a-zà-ÿ'’\-]{4,}", low)):
        w = w.replace("’s", "").replace("'s", "")
        if w in names and names[w] not in ids:
            ids.append(names[w])
    for nm, tid in teams.items():
        if nm in low and tid not in tids:
            tids.append(tid)
    return ids[:6], tids[:3]


def fetch_news(boot, ttl_min=15, force=False, log=print):
    """Returns the news list (newest first). Uses the cache when it is fresh or the network fails."""
    try:
        c = json.load(open(CACHE, encoding="utf-8"))
        if not force and time.time() - c.get("fetched", 0) < ttl_min * 60:
            return c
    except Exception:                                   # noqa: BLE001
        c = {}
    names, teams = _index(boot)
    items, ok, bad = [], [], []
    for name, url in FEEDS:
        try:
            got = _parse(name, _fetch(url))
            items += got; ok.append(f"{name} {len(got)}")
        except Exception as e:                          # noqa: BLE001
            bad.append(f"{name}: {str(e)[:40]}")
    if not items:
        return c if c else {"items": [], "fetched": 0, "sources": [], "errors": bad}
    seen, out = set(), []
    for it in sorted(items, key=lambda x: -x["ts"]):
        k = re.sub(r"\W+", "", it["title"].lower())[:60]
        if k in seen:
            continue
        seen.add(k)
        ids, tids = _match(it["title"], names, teams)
        kind = "health" if HEALTH.search(it["title"]) else "teamnews" if TEAMNEWS.search(it["title"]) else \
               "transfer" if TRANSFER.search(it["title"]) else "other"
        if kind == "other" and it["src"] not in ("r/FantasyPL", "FF Scout"):
            ids = []                                    # a surname alone in a match report/feature is too weak a signal
            if not tids:
                continue                                # drop generic non-FPL-relevant headlines
        out.append({**it, "ids": ids, "tids": tids, "kind": kind})
    res = {"items": out[:150], "fetched": int(time.time()), "sources": ok, "errors": bad}
    json.dump(res, open(CACHE, "w", encoding="utf-8"))
    log(f"  news: {len(out)} items ({', '.join(ok)})" + (f"; failed: {'; '.join(bad)}" if bad else ""))
    return res


def news_sig(news):
    return hashlib.md5(json.dumps([n["link"] for n in news.get("items", [])[:60]]).encode()).hexdigest()[:10]
