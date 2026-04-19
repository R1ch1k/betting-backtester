"""Generate real README numbers for betting-backtester.

Runs XgPoissonStrategy through the walk-forward evaluator against
Premier League data from football-data.co.uk. Prints output formatted
for direct paste into README.md and saves an equity curve plot.

Usage:
    # 1. Download CSVs from https://www.football-data.co.uk/englandm.php
    #    into ./data/ -- one file per season like E0_2020-21.csv, E0_2021-22.csv etc.
    # 2. Run:
    uv run python scripts/generate_readme_example.py

What this does:
    Loads the CSVs, configures XgPoissonStrategy with realistic
    parameters, wraps in WalkForwardEvaluator with 1yr train / 3mo test,
    runs the full pipeline, reports yield point estimate and bootstrap CI,
    saves docs/equity_curve.png showing the cumulative equity path.

Tweakables at the top of main(). Defaults are what the README describes.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt

from betting_backtester.commission import NetWinningsCommission
from betting_backtester.football_data import FootballDataLoader
from betting_backtester.reporting import compute_yield_ci
from betting_backtester.strategies.xg_poisson import XgPoissonStrategy
from betting_backtester.walk_forward import WalkForwardEvaluator, WindowSpec


def main() -> int:
    # ---- config -----------------------------------------------------------
    data_dir = Path("data")
    csv_glob = "*/E0.csv"  # one folder per season, each containing E0.csv
    starting_bankroll = 1_000.0
    edge_threshold = 0.02
    kelly_fraction = 0.25
    max_exposure_fraction = 0.05
    train_duration = timedelta(days=365)
    test_duration = timedelta(days=90)
    commission_rate = 0.05
    n_resamples = 10_000
    seed = 0

    # ---- load -------------------------------------------------------------
    csv_paths = sorted(data_dir.glob(csv_glob))
    if not csv_paths:
        print(
            f"ERROR: no CSVs found matching {data_dir}/{csv_glob}",
            file=sys.stderr,
        )
        print(
            "Download season CSVs from https://www.football-data.co.uk/englandm.php",
            file=sys.stderr,
        )
        return 1
    print(f"Loading {len(csv_paths)} CSV file(s): {[p.name for p in csv_paths]}")

    loader = FootballDataLoader(csv_paths)
    summary = loader.load_summary
    print(f"Loaded {summary.matches_loaded} matches")
    print(
        f"  skipped: "
        f"{summary.skipped_missing_date} missing_date / "
        f"{summary.skipped_missing_pinnacle_odds} missing_pinnacle_odds / "
        f"{summary.skipped_missing_result} missing_result / "
        f"{summary.skipped_invalid_odds} invalid_odds"
    )

    match_directory = loader.matches

    # ---- run --------------------------------------------------------------
    def strategy_factory() -> XgPoissonStrategy:
        return XgPoissonStrategy(
            match_directory=match_directory,
            edge_threshold=edge_threshold,
            kelly_fraction=kelly_fraction,
            max_exposure_fraction=max_exposure_fraction,
        )

    evaluator = WalkForwardEvaluator(
        event_source=loader,
        strategy_factory=strategy_factory,
        commission_model=NetWinningsCommission(rate=commission_rate),
        window_spec=WindowSpec(
            train_duration=train_duration,
            test_duration=test_duration,
        ),
        starting_bankroll=starting_bankroll,
        seed=seed,
        n_resamples=n_resamples,
    )

    print("Running walk-forward evaluation...")
    result = evaluator.run()

    # ---- report -----------------------------------------------------------
    summary_metrics = result.aggregate_summary
    yield_ci = result.aggregate_yield_ci

    print()
    print("=" * 60)
    print("README output block (copy from here):")
    print("=" * 60)
    print(f"Windows:     {len(result.per_window)}")
    print(f"Bets:        {summary_metrics.n_bets}")
    print(f"Net P&L:     {summary_metrics.net_pnl:+.2f}")
    print(
        f"Yield:       {yield_ci.mean:.4%}  "
        f"(95% CI: [{yield_ci.lower:.4%}, {yield_ci.upper:.4%}])"
    )
    print("=" * 60)

    # ---- plot -------------------------------------------------------------
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    equity = result.aggregate_equity_curve

    fig, ax = plt.subplots(figsize=(9, 4.5))
    timestamps = [p.timestamp for p in equity]
    bankrolls = [p.bankroll for p in equity]
    ax.plot(timestamps, bankrolls, linewidth=1.2)
    ax.axhline(starting_bankroll, linestyle="--", alpha=0.4, color="gray")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll (£)")
    ax.set_title(
        f"XgPoissonStrategy — walk-forward, "
        f"{train_duration.days}d train / {test_duration.days}d test"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = docs_dir / "equity_curve.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nEquity curve saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())