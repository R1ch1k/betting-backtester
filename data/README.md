# data/

This directory holds historical match CSVs used by the `FootballDataLoader`. The files themselves are not committed — they're redistributable public data, but there's no reason for this repo to re-host them. Download them directly from the source.

## Expected layout

```
data/
├── 2020-21/E0.csv
├── 2021-22/E0.csv
├── 2022-23/E0.csv
└── 2023-24/E0.csv
```

Each `E0.csv` is one season of English Premier League match results and closing odds. The folder name (e.g. `2020-21`) isn't read by the loader — only the file basename (`E0`) is parsed as the league code — but the nested layout keeps seasons distinguishable on disk without renaming files.

## Where to download

[football-data.co.uk / England](https://www.football-data.co.uk/englandm.php)

On that page, scroll to the "Season-by-season" section. Each season link downloads a single ZIP or CSV. For the Premier League you want the `E0.csv` from each of:

- Season 2020/2021
- Season 2021/2022
- Season 2022/2023
- Season 2023/2024

Save each file to its corresponding subfolder above. The repo's `.gitignore` excludes all of `data/`, so your local downloads won't be accidentally committed.

## Other leagues

The loader works on any league file with the same column layout as football-data's mainline series (the `PSH/PSD/PSA` Pinnacle closing odds, match result, kickoff time). League codes match football-data's filename prefixes: `E0` (Premier League), `E1` (Championship), `SP1` (La Liga), `D1` (Bundesliga), and so on. See `src/betting_backtester/football_data.py` for the exact column requirements.