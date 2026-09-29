# TerminalWeave

TerminalWeave is a dependency-light Python engine for modelling day-to-day container-terminal operations. It covers marine access, berth use, quay cranes, yard placement, truck-gate appointments, customs holds, reefer monitoring, hazardous-goods segregation, equipment maintenance, rail blocks, executable rail departure plans, demurrage billing and transshipment connections.

The package is intentionally usable without a database or web framework. Domain services expose ordinary Python APIs, while the persistence layer supplies an append-only event store and snapshots for embedding in larger systems.

## Quick start

```bash
python -m pip install -e ".[test]"
python -m terminalweave.cli demo
python -m pytest -q
```

Run the small HTTP adapter with `terminalweave serve`. It exposes `GET /health`,
`POST /tide/clearance`, and `POST /billing/demurrage`. The other packages are
library components intended to be composed by an embedding application.

## Design principles

- domain calculations are deterministic for identical inputs;
- time values are timezone-aware UTC datetimes;
- operational state transitions are explicit and validated;
- safety constraints fail closed when required information is absent;
- money uses `Decimal`;
- core domain modules depend only on the Python standard library.

## Main packages

`marine`, `cranes`, `yard`, `gate`, `customs`, `reefer`, `hazmat`, `equipment`, `rail`, `billing`, `transshipment`, `planning`, `persistence`, `reporting`, and `api`.

## Rail departure plans

`rail.departure.DeparturePlanner` turns the existing rail primitives — block
formation, car consist, documentation/physical cutoffs and yard-track
allocation — into one executable `DeparturePlan`. A `plan()` packs eligible
boxes into wagons against weight/length capacity and the destination calling
order, checks every box's readiness against the cutoffs, and best-fits a yard
track. Assembly runs entirely in scratch space: if any stage fails no wagon is
left partially loaded and no track is booked. `confirm()` pins every
container-to-wagon placement and atomically reserves the track; a later
`replan()` close to the cutoff never moves a pinned container, never reorders
the confirmed cars and never changes the track — boxes that cannot be added
without breaking those guarantees are reported with a machine-readable
exclusion reason instead of forcing a reshuffle.
