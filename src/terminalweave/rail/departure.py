"""Executable train departure planning.

The rail package already knows how to form blocks (:mod:`blocks`), split the
train into rail cars (:mod:`train`), check documentation/physical cutoffs
(:mod:`cutoff`) and allocate a yard track (:mod:`tracks`).  This module fuses
those four capabilities, together with box readiness and the destination
calling order, into a single *executable* departure plan.

Guarantees
----------
* A plan is assembled entirely in scratch space.  Nothing is written back to
  the wagons or to the shared :class:`TrackAllocator` until *every* stage has
  passed, so a failure never leaves a partially loaded wagon or a half-booked
  track behind.
* Plans can be confirmed.  A confirmed plan pins every container-to-wagon
  placement and the allocated track.  A re-plan triggered close to the cutoff
  never moves a pinned container, never reorders the pinned wagons and never
  reallocates the track; anything that cannot be added without breaking those
  guarantees is reported as excluded rather than forced in.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from terminalweave.core.clock import ensure_utc, utc_now
from terminalweave.core.errors import (
    CapacityError,
    ConflictError,
    NotFoundError,
    StateTransitionError,
    ValidationError,
)

from .blocks import RailContainer, Wagon, WagonLoad
from .cutoff import ContainerReadiness, CutoffChecker, TrainCutoff
from .tracks import TrackAllocator, TrainMovement, YardTrack
from .train import TrainConsist

# Exclusion reason codes (stable, machine readable).
REASON_NOT_READY = 'cutoff_not_met'
REASON_UNKNOWN_DESTINATION = 'destination_not_in_order'
REASON_CAPACITY = 'no_capacity'
REASON_SEQUENCE_LOCKED = 'sequence_locked_by_confirmed_cars'


@dataclass(frozen=True)
class PlannedWagon:
    """A wagon in its final consist position.

    ``locked`` reflects whether the wagon holds a confirmed placement and is
    derived from the owning plan's pinned set rather than stored, so it stays
    correct when a re-plan draft is confirmed and its pin set expands.
    """

    position: int
    wagon: Wagon
    destination: str
    container_ids: tuple[str, ...]
    used_weight_kg: float
    used_length_ft: int
    locked: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.container_ids


@dataclass(frozen=True)
class ExcludedContainer:
    container_id: str
    destination: str
    reason: str
    detail: str
    documentation_ok: bool
    physical_ok: bool


@dataclass(frozen=True)
class DeparturePlan:
    """An executable departure plan.

    ``loads`` maps ``wagon_id -> (container_id -> container)``.  Both the
    mapping and the tuples are immutable so a returned plan cannot mutate the
    planner's state.
    """

    train_id: str
    revision: int
    status: str  # 'draft' | 'confirmed'
    cutoff: TrainCutoff
    destination_order: tuple[str, ...]
    departure: datetime
    track_id: str
    movement: TrainMovement
    wagons: tuple[PlannedWagon, ...]
    loads: Mapping[str, Mapping[str, RailContainer]]
    excluded: tuple[ExcludedContainer, ...]
    eligibility: tuple[dict[str, object], ...]
    pinned: frozenset[str]
    consist: TrainConsist | None
    wagon_ids: tuple[str, ...]
    total_payload_kg: float
    total_length_ft: int
    generated_at: datetime
    planned_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'departure', ensure_utc(self.departure))
        object.__setattr__(self, 'generated_at', ensure_utc(self.generated_at))
        object.__setattr__(self, 'planned_at', ensure_utc(self.planned_at))
        # Re-derive each wagon's lock flag from this plan's own pinned set so
        # it is always consistent (a confirmed replan expands the pin set).
        pinned = self.pinned
        resolved = tuple(
            replace(wagon, locked=any(cid in pinned
                                      for cid in wagon.container_ids))
            for wagon in self.wagons
        )
        object.__setattr__(self, 'wagons', resolved)

    @property
    def is_confirmed(self) -> bool:
        return self.status == 'confirmed'

    def wagon_of(self, container_id: str) -> str | None:
        for wagon_id, containers in self.loads.items():
            if container_id in containers:
                return wagon_id
        return None

    def is_locked(self, container_id: str) -> bool:
        return container_id in self.pinned

    def container(self, container_id: str) -> RailContainer | None:
        for containers in self.loads.values():
            found = containers.get(container_id)
            if found is not None:
                return found
        return None


@dataclass
class DepartureRequest:
    """Inputs for a fresh departure plan."""

    train_id: str
    cutoff: TrainCutoff
    destination_order: list[str]
    wagons: list[Wagon]
    containers: list[RailContainer]
    readiness: list[ContainerReadiness]
    movement: TrainMovement
    tracks: list[YardTrack]
    consist: TrainConsist | None = None
    generated_at: datetime | None = None


def _freeze_loads(
    loads: list[WagonLoad],
) -> Mapping[str, Mapping[str, RailContainer]]:
    frozen: dict[str, Mapping[str, RailContainer]] = {}
    for load in loads:
        frozen[load.wagon.wagon_id] = MappingProxyType(
            {c.container_id: c for c in load.containers}
        )
    return MappingProxyType(frozen)


class DeparturePlanner:
    """Builds, confirms and re-plans executable departure plans.

    A single planner instance owns one shared :class:`TrackAllocator` across
    many trains so that track occupancy is respected fleet-wide.
    """

    def __init__(self, tracks: list[YardTrack] | None = None):
        self._allocator = TrackAllocator(tracks or [])
        self._drafts: dict[str, DeparturePlan] = {}
        self._confirmed: dict[str, DeparturePlan] = {}
        self._movements: dict[str, TrainMovement] = {}

    # ------------------------------------------------------------------ views

    @property
    def allocator(self) -> TrackAllocator:
        return self._allocator

    def occupied_tracks(self) -> dict[str, str]:
        return dict(self._allocator.occupied)

    def get_plan(self, train_id: str) -> DeparturePlan | None:
        confirmed = self._confirmed.get(train_id)
        return confirmed if confirmed is not None else self._drafts.get(train_id)

    def get_confirmed(self, train_id: str) -> DeparturePlan | None:
        return self._confirmed.get(train_id)

    # ------------------------------------------------------------------ plan

    def plan(self, request: DepartureRequest) -> DeparturePlan:
        """Produce a fresh draft plan atomically.

        Raises on any validation/capacity failure without mutating state.
        """
        if request.train_id in self._confirmed:
            raise ConflictError(f'train {request.train_id} already confirmed')

        prepared = self._prepare(request)
        # Build on a scratch planner; only the returned immutable plan is kept.
        plan = self._build_fresh(prepared, request.generated_at or utc_now())
        self._drafts[request.train_id] = plan
        self._movements[request.train_id] = request.movement
        return plan

    def confirm(self, train_id: str) -> DeparturePlan:
        """Confirm the current draft, atomically booking its track.

        If the track cannot be booked, no state changes and the draft remains
        in place.
        """
        draft = self._drafts.get(train_id)
        if draft is None:
            if train_id in self._confirmed:
                raise StateTransitionError(f'train {train_id} already confirmed')
            raise NotFoundError(f'no draft plan for train {train_id}')

        movement = self._movements[train_id]
        # Reserve the specific track chosen during planning.  Any failure
        # unwinds automatically because nothing has been mutated yet.
        self._allocator.reserve(draft.track_id, movement)
        try:
            confirmed = replace(draft, status='confirmed',
                                pinned=_all_container_ids(draft))
        except BaseException:
            self._allocator.release(draft.track_id)
            raise
        self._confirmed[train_id] = confirmed
        self._drafts.pop(train_id, None)
        return confirmed

    def replan(
        self,
        train_id: str,
        containers: list[RailContainer],
        readiness: list[ContainerReadiness],
        generated_at: datetime | None = None,
    ) -> DeparturePlan:
        """Re-plan a confirmed train close to its cutoff.

        Confirmed placements and the track are pinned and are never moved; new
        boxes that cannot be added without breaking the pinned consist are
        returned in ``excluded`` rather than forcing a reshuffle.
        """
        baseline = self._confirmed.get(train_id)
        if baseline is None:
            if train_id in self._drafts:
                raise StateTransitionError(
                    f'train {train_id} has a draft; call plan() again instead')
            raise NotFoundError(f'no confirmed plan for train {train_id}')

        containers = list(containers)
        readiness = list(readiness)
        self._validate_unique_ids(containers, readiness)
        by_id = {c.container_id: c for c in containers}
        ready_by_id = {r.container_id: r for r in readiness}

        # Pinned containers must still be present and unchanged.
        for cid in sorted(baseline.pinned):
            current = by_id.get(cid)
            if current is None:
                raise ConflictError(
                    f'confirmed container {cid} missing from replan input')
            original = baseline.container(cid)
            if (current.destination != original.destination
                    or current.weight_kg != original.weight_kg
                    or current.length_ft != original.length_ft):
                raise ConflictError(
                    f'confirmed container {cid} changed; cannot disturb a '
                    'confirmed placement')

        # Equipment for the train is fixed by the confirmed plan.
        wagons = self._baseline_wagons(baseline)
        movement = self._movements[train_id]
        self._validate_track_still_compatible(baseline.track_id, movement)

        plan = self._build_replan(
            baseline, wagons, containers, by_id, ready_by_id,
            generated_at or utc_now(),
        )
        self._drafts[train_id] = plan
        return plan

    def release(self, train_id: str) -> bool:
        """Release a confirmed train's track and forget the plan."""
        plan = self._confirmed.pop(train_id, None)
        self._drafts.pop(train_id, None)
        self._movements.pop(train_id, None)
        if plan is None:
            return False
        self._allocator.release(plan.track_id)
        return True

    # ------------------------------------------------------------- internals

    def _prepare(self, request: DepartureRequest) -> dict[str, object]:
        if not request.train_id.strip():
            raise ValidationError('train id required')
        if request.movement.train_id != request.train_id:
            raise ValidationError('movement train id does not match request')
        if request.cutoff.train_id != request.train_id:
            raise ValidationError('cutoff train id does not match request')
        if not request.destination_order:
            raise ValidationError('destination order required')
        if len(set(request.destination_order)) != len(request.destination_order):
            raise ValidationError('duplicate destination in calling order')
        if not request.wagons:
            raise ValidationError('at least one wagon required')
        wagon_ids = [w.wagon_id for w in request.wagons]
        if len(set(wagon_ids)) != len(wagon_ids):
            raise ValidationError('duplicate wagon id')
        self._validate_unique_ids(request.containers, request.readiness)
        if request.consist is not None:
            self._validate_consist(request.consist, request.train_id)
        rank = {d: i for i, d in enumerate(request.destination_order)}
        return {
            'cutoff': request.cutoff,
            'order': tuple(request.destination_order),
            'rank': rank,
            'wagons': list(request.wagons),
            'containers': list(request.containers),
            'readiness': list(request.readiness),
            'movement': request.movement,
            'tracks': list(request.tracks),
            'consist': request.consist,
        }

    @staticmethod
    def _validate_unique_ids(
        containers: list[RailContainer],
        readiness: list[ContainerReadiness],
    ) -> None:
        cids = [c.container_id for c in containers]
        if len(set(cids)) != len(cids):
            raise ValidationError('duplicate container id')
        rids = [r.container_id for r in readiness]
        if len(set(rids)) != len(rids):
            raise ValidationError('duplicate readiness record')
        missing = set(cids) - set(rids)
        if missing:
            # Fail closed: without readiness we cannot prove a box meets the
            # documentation and physical cutoffs.
            raise ValidationError(
                'missing readiness for: ' + ', '.join(sorted(missing)))

    @staticmethod
    def _validate_consist(consist: TrainConsist, train_id: str) -> None:
        if consist.train_id != train_id:
            raise ValidationError('consist train id does not match request')
        # Construction already validates length/gross; re-run defensively.
        consist.validate()

    # ----- fresh build -----------------------------------------------------

    def _build_fresh(
        self, prepared: dict[str, object], generated_at: datetime
    ) -> DeparturePlan:
        cutoff: TrainCutoff = prepared['cutoff']
        rank: dict[str, int] = prepared['rank']
        wagons: list[Wagon] = prepared['wagons']
        containers: list[RailContainer] = prepared['containers']
        readiness: list[ContainerReadiness] = prepared['readiness']
        movement: TrainMovement = prepared['movement']
        consist: TrainConsist | None = prepared['consist']

        loads = [WagonLoad(w) for w in wagons]
        checker = CutoffChecker()
        ready_by_id = {r.container_id: r for r in readiness}

        eligibility_rows: list[dict[str, object]] = []
        excluded: list[ExcludedContainer] = []
        loadable: list[RailContainer] = []

        for container in containers:
            record = ready_by_id[container.container_id]
            row = checker.evaluate(cutoff, record)
            eligibility_rows.append(row)
            if not row['eligible']:
                excluded.append(ExcludedContainer(
                    container.container_id, container.destination,
                    REASON_NOT_READY,
                    _slack_detail(row),
                    bool(row['documentation_ok']), bool(row['physical_ok'])))
                continue
            if container.destination not in rank:
                excluded.append(ExcludedContainer(
                    container.container_id, container.destination,
                    REASON_UNKNOWN_DESTINATION,
                    f"destination {container.destination} is not on the "
                    'calling order',
                    bool(row['documentation_ok']), bool(row['physical_ok'])))
                continue
            loadable.append(container)

        # Best-fit packing, deterministic.  Reuse the proven block planner
        # assignment rules via WagonLoad.can_add so wagon capacity and the
        # destination binding are enforced in exactly one place.
        ordered = sorted(
            loadable,
            key=lambda c: (rank[c.destination], -c.priority, -c.weight_kg,
                           -c.length_ft, c.container_id),
        )
        for container in ordered:
            candidates = [
                load for load in loads
                if load.can_add(container)
                and load.wagon.destination in (None, container.destination)
            ]
            if not candidates:
                # Eligible box that cannot fit: hard failure so the caller
                # never receives a silently short train.  Nothing has been
                # committed anywhere.
                raise CapacityError(
                    f'no wagon capacity for eligible container '
                    f'{container.container_id}')
            chosen = _best_fit(candidates, container)
            chosen.add(container)

        self._assert_single_destination_and_order(loads, rank)
        fresh_order = self._fresh_wagon_order(loads, rank)
        planned_wagons = self._planned_wagons_in_order(
            fresh_order, frozenset())

        if consist is not None:
            self._check_consist_payload(consist, fresh_order)

        track_id = self._choose_track(prepared['tracks'], movement)

        total_weight = sum(l.used_weight_kg for l in fresh_order)
        total_length = sum(l.used_length_ft for l in fresh_order)
        return DeparturePlan(
            train_id=cutoff.train_id,
            revision=1,
            status='draft',
            cutoff=cutoff,
            destination_order=prepared['order'],
            departure=cutoff.departure,
            track_id=track_id,
            movement=movement,
            wagons=planned_wagons,
            loads=_freeze_loads(fresh_order),
            excluded=tuple(excluded),
            eligibility=tuple(eligibility_rows),
            pinned=frozenset(),
            consist=consist,
            wagon_ids=tuple(w.wagon_id for w in wagons),
            total_payload_kg=total_weight,
            total_length_ft=total_length,
            generated_at=generated_at,
        )

    # ----- replan build ----------------------------------------------------

    def _build_replan(
        self,
        baseline: DeparturePlan,
        wagons: list[Wagon],
        containers: list[RailContainer],
        by_id: dict[str, RailContainer],
        ready_by_id: dict[str, ContainerReadiness],
        generated_at: datetime,
    ) -> DeparturePlan:
        cutoff = baseline.cutoff
        rank = {d: i for i, d in enumerate(baseline.destination_order)}
        checker = CutoffChecker()

        # Every baseline wagon keeps its exact consist position.  Rebuild the
        # scratch loads in that order and re-place the pinned containers first.
        baseline_loads = [WagonLoad(pw.wagon) for pw in baseline.wagons]
        loads_by_id = {load.wagon.wagon_id: load for load in baseline_loads}
        for pw, load in zip(baseline.wagons, baseline_loads):
            for cid in pw.container_ids:
                load.add(by_id[cid])

        # Rank that the next newly activated wagon may not precede.  Pinned
        # wagons are already non-decreasing, so the tail pins the boundary.
        tail_rank = -1
        for load in baseline_loads:
            if load.containers:
                tail_rank = max(tail_rank, rank[load.containers[0].destination])

        pinned = set(baseline.pinned)
        eligibility_rows: list[dict[str, object]] = []
        excluded: list[ExcludedContainer] = []

        new_containers = [
            c for c in containers if c.container_id not in pinned
        ]
        for container in sorted(
            new_containers,
            key=lambda c: (rank.get(c.destination, len(rank)), -c.priority,
                           -c.weight_kg, -c.length_ft, c.container_id),
        ):
            cid = container.container_id
            record = ready_by_id.get(cid)
            if record is None:
                # Fail closed for any new box without readiness.
                raise ValidationError(f'missing readiness for {cid}')
            row = checker.evaluate(cutoff, record)
            eligibility_rows.append(row)

            if not row['eligible']:
                excluded.append(ExcludedContainer(
                    cid, container.destination, REASON_NOT_READY,
                    _slack_detail(row),
                    bool(row['documentation_ok']), bool(row['physical_ok'])))
                continue
            if container.destination not in rank:
                excluded.append(ExcludedContainer(
                    cid, container.destination, REASON_UNKNOWN_DESTINATION,
                    f"destination {container.destination} is not on the "
                    'calling order',
                    bool(row['documentation_ok']), bool(row['physical_ok'])))
                continue

            dest_rank = rank[container.destination]

            # 1) Top up a same-destination wagon in its existing position.
            same_dest = [
                load for load in baseline_loads
                if _load_destination(load) == container.destination
            ]
            fit_in_place = [load for load in same_dest
                            if load.can_add(container)]

            # 2) Otherwise activate an empty wagon that is still at the tail.
            #    Empty wagons never shift a confirmed car: they sit after the
            #    last non-empty wagon (dedicated empties are part of the
            #    baseline ordering and keep their slot).
            fit_empty: list[WagonLoad] = []
            if dest_rank >= tail_rank:
                for load in baseline_loads:
                    if load.containers:
                        continue
                    allowed_dest = load.wagon.destination
                    if allowed_dest not in (None, container.destination):
                        continue
                    if load.can_add(container):
                        fit_empty.append(load)

            if fit_in_place:
                _best_fit(fit_in_place, container).add(container)
            elif fit_empty:
                chosen = _best_fit(fit_empty, container)
                chosen.add(container)
                if dest_rank > tail_rank:
                    tail_rank = dest_rank
            else:
                reason, detail = _replan_exclusion_reason(
                    container, dest_rank, tail_rank, same_dest,
                    baseline_loads)
                excluded.append(ExcludedContainer(
                    cid, container.destination, reason, detail,
                    bool(row['documentation_ok']), bool(row['physical_ok'])))

        # Keep the baseline positional order verbatim; never resort.
        self._assert_single_destination_and_order(baseline_loads, rank)
        planned_wagons = self._planned_wagons_in_order(
            baseline_loads, frozenset(pinned))

        if baseline.consist is not None:
            self._check_consist_payload(baseline.consist, baseline_loads)

        total_weight = sum(l.used_weight_kg for l in baseline_loads)
        total_length = sum(l.used_length_ft for l in baseline_loads)
        return DeparturePlan(
            train_id=baseline.train_id,
            revision=baseline.revision + 1,
            status='draft',
            cutoff=baseline.cutoff,
            destination_order=baseline.destination_order,
            departure=baseline.departure,
            track_id=baseline.track_id,
            movement=baseline.movement,
            wagons=planned_wagons,
            loads=_freeze_loads(baseline_loads),
            excluded=tuple(excluded),
            eligibility=tuple(eligibility_rows),
            pinned=frozenset(pinned),
            consist=baseline.consist,
            wagon_ids=tuple(w.wagon_id for w in wagons),
            total_payload_kg=total_weight,
            total_length_ft=total_length,
            generated_at=generated_at,
        )

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _baseline_wagons(plan: DeparturePlan) -> list[Wagon]:
        # PlannedWagons already preserve the confirmed consist position order.
        return [pw.wagon for pw in plan.wagons]

    @staticmethod
    def _fresh_wagon_order(
        loads: list[WagonLoad], rank: dict[str, int]
    ) -> list[WagonLoad]:
        """Order wagons for a fresh plan.

        Non-empty wagons come first in destination calling order, then empty
        dedicated wagons at their own destination rank, then empty generic
        wagons at the very tail.  Keeping every empty wagon at the tail means
        a later near-cutoff replan can activate one without crossing a
        confirmed car.
        """
        def sort_key(load: WagonLoad):
            if load.containers:
                return (0, rank[load.containers[0].destination],
                        load.wagon.wagon_id)
            if load.wagon.destination is not None:
                return (1, rank[load.wagon.destination], load.wagon.wagon_id)
            return (2, len(rank), load.wagon.wagon_id)

        return sorted(loads, key=sort_key)

    @staticmethod
    def _planned_wagons_in_order(
        loads: list[WagonLoad],
        locked_ids: frozenset[str],
    ) -> tuple[PlannedWagon, ...]:
        out: list[PlannedWagon] = []
        for position, load in enumerate(loads):
            dest = _load_destination(load)
            if dest is None:
                # An empty generic wagon has no destination yet.
                dest = ''
            ids = tuple(c.container_id for c in load.containers)
            out.append(PlannedWagon(
                position=position,
                wagon=load.wagon,
                destination=dest,
                container_ids=ids,
                used_weight_kg=load.used_weight_kg,
                used_length_ft=load.used_length_ft,
                locked=any(cid in locked_ids for cid in ids),
            ))
        return tuple(out)

    @staticmethod
    def _assert_single_destination_and_order(
        loads: list[WagonLoad], rank: dict[str, int]
    ) -> None:
        seen = -1
        for load in loads:
            dests = {c.destination for c in load.containers}
            if len(dests) > 1:
                raise ConflictError(
                    f'wagon {load.wagon.wagon_id} mixes destinations')
            if not dests:
                continue
            dest = next(iter(dests))
            if dest not in rank:
                raise ValidationError(
                    f'unknown destination in sequence: {dest}')
            current = rank[dest]
            if current < seen:
                raise ConflictError(
                    'wagon order would require intermediate shunting')
            seen = current

    @staticmethod
    def _check_consist_payload(
        consist: TrainConsist, loads: list[WagonLoad]
    ) -> None:
        payload = sum(l.used_weight_kg for l in loads)
        capacity = consist.payload_capacity_kg()
        if payload > capacity:
            raise CapacityError(
                f'payload {payload} kg exceeds consist capacity {capacity} kg')

    def _choose_track(
        self,
        tracks: list[YardTrack],
        movement: TrainMovement,
    ) -> str:
        """Best-fit a track from the planner's shared pool.

        Only tracks known to the owning :class:`TrackAllocator` are eligible,
        and any track already occupied by another train is excluded so that a
        plan never points at a slot that cannot actually be booked.
        """
        candidates: list[tuple[float, str]] = []
        seen: set[str] = set()
        for track in tracks:
            if track.track_id in seen:
                continue
            seen.add(track.track_id)
            configured = self._allocator.tracks.get(track.track_id)
            if configured is None:
                raise ValidationError(
                    f'track {track.track_id} is not configured for this planner')
            occupant = self._allocator.occupied.get(track.track_id)
            if occupant is not None and occupant != movement.train_id:
                continue
            if configured.usable_length_m < movement.length_m:
                continue
            if movement.needs_electric and not configured.electrified:
                continue
            if movement.needs_service and not configured.service_track:
                continue
            candidates.append(
                (configured.usable_length_m - movement.length_m,
                 track.track_id))
        if not candidates:
            raise CapacityError(
                f'no compatible yard track for train {movement.train_id}')
        return min(candidates)[1]

    def _validate_track_still_compatible(
        self, track_id: str, movement: TrainMovement
    ) -> None:
        track = self._allocator.tracks.get(track_id)
        if track is None:
            raise NotFoundError(f'track {track_id} no longer configured')
        if track.usable_length_m < movement.length_m:
            raise CapacityError(
                f'movement no longer fits on pinned track {track_id}')
        if movement.needs_electric and not track.electrified:
            raise CapacityError(
                f'pinned track {track_id} is not electrified')
        if movement.needs_service and not track.service_track:
            raise CapacityError(
                f'pinned track {track_id} is not a service track')


def _load_destination(load: WagonLoad) -> str | None:
    if load.containers:
        return load.containers[0].destination
    return load.wagon.destination


def _replan_exclusion_reason(
    container: RailContainer,
    dest_rank: int,
    tail_rank: int,
    same_dest: list[WagonLoad],
    baseline_loads: list[WagonLoad],
) -> tuple[str, str]:
    """Classify why a new box cannot be added during a near-cutoff replan.

    The distinguishing question is whether *any* structurally usable wagon
    exists.  If the only thing blocking placement is the destination calling
    order relative to confirmed cars, that is a sequence lock; otherwise it is
    plain capacity.
    """
    # Is there an empty wagon that the box would physically fit?
    structural_empty = [
        load for load in baseline_loads
        if not load.containers
        and load.wagon.destination in (None, container.destination)
        and load.can_add(container)
    ]
    # Capacity remaining in the existing same-destination consist block.
    same_dest_room = any(load.can_add(container) for load in same_dest)

    if structural_empty and dest_rank < tail_rank:
        return (REASON_SEQUENCE_LOCKED,
                f'destination rank {dest_rank} precedes the confirmed tail '
                f'rank {tail_rank}; a new wagon would reorder confirmed cars')
    if same_dest and not same_dest_room and structural_empty:
        # Existing block is full; an extra wagon exists but cannot be slotted
        # in before later confirmed destinations.
        if dest_rank < tail_rank:
            return (REASON_SEQUENCE_LOCKED,
                    f'destination {container.destination} block is full and '
                    'another wagon cannot precede confirmed cars')
    if same_dest and not same_dest_room:
        return (REASON_CAPACITY,
                f'destination {container.destination} wagon is full and '
                'opening another wagon would reorder confirmed cars')
    if not structural_empty and dest_rank < tail_rank:
        return (REASON_SEQUENCE_LOCKED,
                f'destination rank {dest_rank} precedes the confirmed tail '
                f'rank {tail_rank}')
    return (REASON_CAPACITY,
            'no free wagon capacity without disturbing a confirmed placement')


def _best_fit(
    candidates: list[WagonLoad], container: RailContainer
) -> WagonLoad:
    def waste(load: WagonLoad):
        dedicated_first = 0 if load.wagon.destination == container.destination \
            else 1
        weight_left = (load.wagon.max_weight_kg
                       - (load.used_weight_kg + container.weight_kg))
        length_left = (load.wagon.slot_equivalent_ft
                       - (load.used_length_ft + container.length_ft))
        return (dedicated_first, length_left, weight_left,
                load.wagon.wagon_id)

    return min(candidates, key=waste)


def _all_container_ids(plan: DeparturePlan) -> frozenset[str]:
    ids: set[str] = set()
    for containers in plan.loads.values():
        ids.update(containers)
    return frozenset(ids)


def _slack_detail(row: Mapping[str, object]) -> str:
    parts = []
    if not row['documentation_ok']:
        parts.append(
            f"documentation slack {row['document_slack_seconds']}s")
    if not row['physical_ok']:
        parts.append(f"physical slack {row['physical_slack_seconds']}s")
    return '; '.join(parts) or 'within cutoff'
