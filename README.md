# FPL Rankings

Fantasy Premier League player rankings in a single offline HTML page, built from the free official FPL API.
Everything (fetching, team-strength model, expected points, ranking, HTML) is deterministic Python.

## Run

```
python fpl.py --open
```

First run sets itself up (creates `data/` and `output/`, installs `numpy` and `scipy`, writes a default `prefs.json`) and builds `output/fpl_rankings.html`. Later runs are update mode:

| Situation | What happens |
|---|---|
| Last check was less than `min_refresh_minutes` ago | no network at all |
| Nothing changed (results, prices, injury news) | fetches ~2 MB, prints "Up to date", rebuilds nothing |
| Something changed | re-fetches only the players whose minutes/points changed, rebuilds |

Flags: `--force` (rebuild anyway), `--offline` (cached data only), `--open` (open the page).

## What it does

- **Team strength model.** Poisson attack/defence ratings fitted on finished matches (target = 65% xG + 35% goals, recent matches weigh more). FPL's coarse 2-5 strength is only a prior that fades as matches are played, so a team expected to be bad that is performing well stops counting as an easy fixture. Home advantage is fitted too.
- **Expected points per fixture** for every player: appearance, goals, assists (xG/xA blended with actual output, shrunk toward the position average), clean sheets, goals conceded, saves, bonus, defensive contribution and cards, scaled by the fixture model and a minutes/availability model. Blank and double gameweeks are handled automatically.
- **Ranking.** Composite score = weighted percentiles (within each position) of projected points over a look-ahead window, season performance index, and points per £. Weights and horizon are adjustable live in the page.
- **Page tabs:** Overview (top 8 per position), Rankings (filters, sortable, expandable breakdown per player), Teams & fixtures (current vs preseason rating, difficulty grid), Best squad (integer-programming optimiser: budget, 3 per club, valid formation), My team (swap suggestions, needs `fpl_team_id`).

## Preferences (`prefs.json`)

| Key | Meaning |
|---|---|
| `default_horizon`, `horizon_max` | look-ahead gameweeks (default 5, max 8) |
| `weights` | `outlook` / `perf` / `value` weights of the composite score |
| `budget` | squad budget for the optimiser (default 100.0) |
| `fpl_team_id` | your FPL team id (the number in the URL on the Points page); enables "My team" |
| `min_refresh_minutes` | skip the network if checked more recently than this |
| `prior_strength`, `prior_scale`, `half_life_gw`, `perf_min_minutes` | model knobs |

## Layout

```
fpl.py          entry point (setup + update logic)
prefs.json      preferences
app/            fpl_data.py (API + cache), fpl_model.py, fpl_report.py, board_template.html
data/           cached API data + state.json (git-ignored)
output/         fpl_rankings.html (git-ignored, data embedded)
```

## Pages

- **Home**: deadline countdown, team rating (your squad vs the best your budget could buy), what to change this week (lineup, captain, best transfer with a noise label, chip), squad alerts, headlines on your players, AI summary.
- **My team**: *Lineup* (best XI, formation, bench order, captain from your 15), *Transfers* (click a player, press Plan; hit worked out from your free transfers; best 1/2/3-transfer plans), *Planner* (multi-gameweek plan: transfers per week, free transfers rolling forward, chips; "Suggest plan" in a conservative/optimised/aggressive style; never-sell / never-buy lists; bench weights), *Chips & Free Hit* (per-gameweek table and the Free Hit squad), *AI*, *History & sync*.
- **Rankings**, **Fixtures** (ticker sortable by overall/attack/defence, any gameweek range), **Compare** (2-3 players), **Watch** (yellow-card bans, DefCon hitters, observed price changes, transfer momentum), **News**, **Best squad**.
- Hover any parameter, column, tag or stat for an explanation (or tab to the "i" icons).

## Things worth knowing

- **Suspension risk**: players on 4 yellow cards (and 9) are discounted by the chance they are banned (5 yellows by GW19 = 1 match, 10 by GW32 = 2); shown as a yellow-card tag with the ban chance.
- **Noise**: the backtest shows about 2 points of error per player per gameweek, so gains under about 3 points are labelled "within noise". Planner defaults to the conservative style.
- **Sync FPL team**: exact squad, selling prices, chips left and free transfers need your login. Serve the page (`start.bat`), drag the *Sync FPL team* button (History & sync) to your bookmarks bar and click it while logged in at fantasy.premierleague.com. It posts only to 127.0.0.1 (origin-checked, your team id only) and rebuilds. Fallback: it downloads `my_team_raw.json`; run `python fpl.py --import-snapshot <file>`.
- **News**: public RSS headlines (BBC Sport, Guardian, Sky Sports, FF Scout, r/FantasyPL when it answers), men's Premier League only, matched to players by name. Headlines and links only; not private leaks, and not fed into the projections.
- **AI analysis** (My team > AI, needs `start.bat`): the button runs the Claude Code skill `fpl-ai-insights` (`.claude/skills/`) headlessly with Read of `data/ai_context.json` and Write of `data/ai_insights.json` only (no web, no shell). The page validates every field (ids exist, multipliers 0.2-1.2, each adjustment must cite a real headline) before showing it. Adjustments are opt-in (checkbox) and change xPts everywhere when on. About 25 seconds and roughly $0.35 of Claude usage per run.
- `python fpl.py --backtest` scores the model on finished gameweeks and tests the ranking weights (one season so far: a sanity check, not proof).
- `python -m unittest discover -s tests` runs the logic tests (bans, free transfers, sell prices, snapshot, AI validation, news filter).
- Private settings (your team id) live in `prefs.local.json` (git-ignored); `prefs.json` stays shareable.

## Notes

- Data source: the official FPL API (unofficial but free, no key). Selling prices need a login, so "My team" uses current prices.
- Projections, not guarantees. The weights are judgement calls and haven't been backtested yet (past-season data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)).
- Works as a Claude Code skill (`/fpl-rankings`), but needs no AI to run; a news layer could write adjustments to a file the model reads.

## One-click mode (Update button)

Double-click `start.bat` (or run `python fpl.py --serve --open`). It serves the page on localhost and shows an **Update data** button in the corner: one click runs the same update flow and reloads the page, so you never need an editor or terminal. The server binds to 127.0.0.1 only and the page works read-only if opened directly as a file (the button only appears when served).

## License

Copyright (c) 2026 André Santos. All rights reserved. The code is public for viewing only; see [LICENSE](LICENSE). Contact me for permission to use it.
