# Contributing

Thanks for taking a look. Issues and pull requests are welcome. This is a small personal project, so replies may take a few days.

## Reporting a bug

Open an issue with:

- what you did and what you expected,
- what happened instead (the full error text, if any),
- your OS and Python version, and the output of `python fpl.py --now` if the build failed.

**Please don't paste your FPL team id, login, cookies or `my_team_raw.json`** into an issue. They are yours alone.

## Suggesting a feature

Open an issue first so we can agree on the idea before you write code. Two things are deliberately out of scope: live (in-match) mode and mini-league features. It is a single-user, local tool.

## Setting up

```
git clone https://github.com/dashzinho/fpl-ranker.git
cd fpl-ranker
python fpl.py --open          # first run installs numpy + scipy and builds the page
python -m unittest discover -s tests
```

No API key is needed. The first build fetches about 2 MB plus one request per player; later runs only re-fetch what changed.

## Pull requests

- Keep each PR focused, and add or update a test in `tests/test_core.py` for logic you change.
- Run `python -m unittest discover -s tests` and make sure it passes.
- **Changes to the model need evidence.** If you change the team-strength model, expected points or the Score, run `python fpl.py --backtest` before and after and include both results in the PR. A change that is inside the noise should say so. See [docs/MODEL.md](docs/MODEL.md) for how to read the backtest.
- The page is a single self-contained HTML file (`app/board_template.html`). Please don't add a build step, a framework or external network requests to it.
- Don't commit anything from `data/`, `output/` or `prefs.local.json` (they are git-ignored for a reason).

## Code of conduct

Be kind and be specific. Critique the code, not the person.

## License

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).
