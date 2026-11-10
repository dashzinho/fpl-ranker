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

## Notes

- Data source: the official FPL API (unofficial but free, no key). Selling prices need a login, so "My team" uses current prices.
- Projections, not guarantees. The weights are judgement calls and haven't been backtested yet (past-season data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)).
- Works as a Claude Code skill (`/fpl-rankings`), but needs no AI to run; a news layer could write adjustments to a file the model reads.

## License

Copyright (c) 2026 André Santos. All rights reserved. The code is public for viewing only; see [LICENSE](LICENSE). Contact me for permission to use it.
