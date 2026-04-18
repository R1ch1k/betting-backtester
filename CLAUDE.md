# Project context

This is a public-facing portfolio project. A betting exchange strategy backtesting framework.
It will be published on GitHub as a signal to recruiters, so code quality and design
clarity matter more than feature count.

## Non-negotiables

- Every piece of code you write must be something I can walk a recruiter through.
  If I would struggle to explain a design choice, propose it and wait for me to agree
  before implementing.
- Tests are not optional. Every non-trivial module needs pytest unit tests before
  we move on to the next module.
- Type hints on all public functions. Ruff-clean. Mypy-clean in strict mode.
- No silent failures. No broad except clauses that swallow errors.
- When you're unsure, ask. Do not guess at architectural decisions.

## What this project is

A small, clean backtesting framework for betting exchange strategies. Event-driven,
commission-aware, with walk-forward validation and honest reporting.

Data sources:
1. football-data.co.uk closing odds + results (primary real data)
2. A synthetic data generator used for backtester self-validation

Shipped example strategies:
1. Back-the-favourite (trivial)
2. xG/Poisson value betting (non-trivial, drawn from my earlier work)
3. Pre-match arbitrage detector (drawn from my earlier work)

## What this project is NOT

- Not a live trading system. No execution, no brokerage connections.
- Not a dashboard or product. Results are exposed via notebooks and CLI reports.
- Not a profitable strategy showcase. The shipped strategies are expected to fail,
  honestly, after transaction costs. The framework's value is being able to show that.
- Not a rewrite of my existing private code. I am building the framework fresh.
  My older code (exchange-arb-bot, AiBetting) is reference material only.

## How to work with me

- Propose the design of a module in plain English before writing code. I will approve or redirect.
- After implementing, briefly summarise what you changed and why.
- If a task takes more than about 30 minutes of your output, pause and check in.
- Do not add features I haven't asked for. Small surface area is a goal, not a constraint.

## Code style

- Python 3.12
- Package manager: uv
- Formatting: ruff (default config)
- Typing: mypy strict
- Tests: pytest
- Use dataclasses or Pydantic models for structured data. No raw dicts floating through the codebase.