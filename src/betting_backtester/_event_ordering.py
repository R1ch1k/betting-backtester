"""Canonical sort key for the backtester's event stream.

Kept in a dedicated module so every :class:`EventSource` implementation
imports the same function instead of carrying its own copy. The
tie-breaking rule is set by ``docs/DESIGN.md``: at equal timestamps,
``MatchSettled`` sorts before ``OddsAvailable``, then by ``match_id``.

Package-internal (leading underscore). Loaders use it; any future
``EventSource`` must too.
"""

from __future__ import annotations

from datetime import datetime

from betting_backtester.models import Event, MatchSettled


def stream_sort_key(event: Event) -> tuple[datetime, int, str]:
    """Canonical sort key for events in a backtester stream.

    Tie-breaking (at equal timestamps):

    * ``MatchSettled`` sorts before ``OddsAvailable`` (0 < 1 in the
      second tuple slot).
    * Within the same type at the same timestamp, order is by
      ``match_id`` ascending.

    Mirrors the invariant in ``docs/DESIGN.md`` and is the single source
    of truth for stream ordering across every ``EventSource``.
    """
    if isinstance(event, MatchSettled):
        return (event.timestamp, 0, event.result.match_id)
    return (event.timestamp, 1, event.snapshot.match_id)
