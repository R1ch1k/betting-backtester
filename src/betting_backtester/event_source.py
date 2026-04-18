"""Runtime-checkable protocol for components that produce the event stream.

The backtester consumes exactly one :class:`EventSource` per run. Two v1
implementations are planned:

* :class:`~betting_backtester.synthetic.SyntheticGenerator` (module 2) —
  samples fixtures and outcomes from a configured true probability
  distribution. Used as the ground-truth rig for backtester correctness.
* ``FootballDataLoader`` (module 6) — reads football-data.co.uk CSVs and
  emits one ``OddsAvailable`` + one ``MatchSettled`` per fixture.

The protocol is intentionally tiny: a single ``events()`` method returning a
fresh iterator. Re-callable — calling it twice yields equivalent, independent
streams.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from betting_backtester.models import Event


@runtime_checkable
class EventSource(Protocol):
    """Produces the canonical, time-ordered stream of backtester events.

    Implementations must guarantee:

    * Events are yielded in strict non-decreasing ``timestamp`` order, with
      ties broken deterministically per the rule in ``docs/DESIGN.md``
      (settlement before new-odds at equal timestamp; then by ``match_id``).
    * ``events()`` may be called multiple times. Each call returns a fresh,
      independent iterator yielding the same sequence given the same
      configuration.
    """

    def events(self) -> Iterator[Event]:
        """Return a fresh iterator over the canonical event stream."""
        ...
