from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import (
    ValidationError,
    CapacityError,
    ConflictError,
    NotFoundError,
    SafetyConstraintError,
)
from terminalweave.rail.blocks import RailContainer, Wagon, WagonLoad
from terminalweave.rail.train import TrainConsist
from terminalweave.rail.cutoff import TrainCutoff, ContainerReadiness, CutoffChecker
from terminalweave.rail.tracks import TrackAllocator, TrainMovement, YardTrack


@dataclass(frozen=True)
class CarLoad:
    car_id: str
    wagon_id: str
    position: int
    containers: tuple[str, ...]
    destination: str | None
    used_weight_kg: float
    gross_kg: float
    used_length_ft: int


@dataclass(frozen=True)
class DeparturePlan:
    train_id: str
    cars: tuple[CarLoad, ...]
    assignments: MappingProxyType
    track_id: str
    cutoff: TrainCutoff
    eligibility: MappingProxyType
    confirmed: frozenset
    total_gross_kg: float
    length_m: float
    boxes: MappingProxyType

    def containers(self) -> tuple[str, ...]:
        return tuple(self.assignments.keys())

    def car_of(self, container_id: str) -> str:
        try:
            return self.assignments[container_id]
        except KeyError:
            raise NotFoundError(f'container not in plan: {container_id}')

    def box(self, container_id: str) -> RailContainer:
        try:
            return self.boxes[container_id]
        except KeyError:
            raise NotFoundError(f'container not in plan: {container_id}')

    def _with(
        self,
        cars: tuple[CarLoad, ...],
        assignments: MappingProxyType,
        eligibility: MappingProxyType,
        confirmed: frozenset,
        track_id: str,
        boxes: MappingProxyType,
    ) -> 'DeparturePlan':
        return DeparturePlan(
            self.train_id,
            cars,
            assignments,
            track_id,
            self.cutoff,
            eligibility,
            confirmed,
            sum(c.gross_kg for c in cars),
            self.length_m,
            boxes,
        )

    def confirm(self, container_id: str) -> 'DeparturePlan':
        """Lock a container's slot. Locked slots survive later rearrangements."""
        if container_id not in self.assignments:
            raise NotFoundError(f'container not in plan: {container_id}')
        return self._with(
            self.cars,
            self.assignments,
            self.eligibility,
            self.confirmed | {container_id},
            self.track_id,
            self.boxes,
        )


class DeparturePlanner:
    """Builds an executable departure plan by solving packing, car capacity,
    destination sequence, box readiness and track occupation together.

    Planning happens entirely on shadow state; yard-track occupation is the
    only shared side effect and it is committed once, after every other check
    has passed. A failed attempt therefore never leaves a partial load or a
    dangling track reservation.
    """

    def __init__(
        self,
        consist: TrainConsist,
        wagons: list[Wagon],
        tracks: TrackAllocator,
        checker: CutoffChecker | None = None,
    ):
        self.consist = consist
        self.tracks = tracks
        self.checker = checker or CutoffChecker()
        self._pair_wagons(wagons)

    def _pair_wagons(self, wagons: list[Wagon]) -> None:
        if len({w.wagon_id for w in wagons}) != len(wagons):
            raise ValidationError('duplicate wagon id')
        if len(wagons) != len(self.consist.cars):
            raise ValidationError('each rail car requires exactly one wagon')
        car_ids = {c.car_id for c in self.consist.cars}
        by_car: dict[str, Wagon] = {}
        for wagon in wagons:
            if wagon.car_id is None:
                raise ValidationError(f'wagon {wagon.wagon_id} is not paired to a rail car')
            if wagon.car_id not in car_ids:
                raise ValidationError(
                    f'wagon {wagon.wagon_id} references unknown car {wagon.car_id}'
                )
            if wagon.car_id in by_car:
                raise ValidationError(f'car {wagon.car_id} paired to multiple wagons')
            by_car[wagon.car_id] = wagon
        # consist order == car order in the departing train
        self.wagons: list[Wagon] = [by_car[c.car_id] for c in self.consist.cars]

    # ------------------------------------------------------------ operations

    def plan(
        self,
        cutoff: TrainCutoff,
        containers: list[RailContainer],
        readiness: dict[str, ContainerReadiness],
        movement: TrainMovement,
        destination_order: list[str],
        now: datetime,
    ) -> DeparturePlan:
        return self._build(
            cutoff=cutoff,
            containers=containers,
            readiness=readiness,
            movement=movement,
            destination_order=destination_order,
            now=ensure_utc(now),
            confirmed=frozenset(),
            preseed={},
            known_eligibility={},
        )

    def replan(
        self,
        previous: DeparturePlan,
        containers: list[RailContainer],
        readiness: dict[str, ContainerReadiness],
        now: datetime,
        movement: TrainMovement | None = None,
        destination_order: list[str] | None = None,
    ) -> DeparturePlan:
        """Re-plan close to cutoff. Confirmed slots are seeded into the same
        cars first and never move; only unconfirmed boxes are re-packed. A
        confirmed box may only disappear via an explicit, pre-deadline
        release, so a rearrangement can never silently break a locked slot.
        """
        requested = {c.container_id for c in containers}
        missing = previous.confirmed - requested
        if missing:
            raise ConflictError(
                'confirmed containers cannot be dropped by a rearrangement: '
                + ', '.join(sorted(missing))
            )
        preseed = {
            cid: previous.assignments[cid]
            for cid in previous.confirmed
            if cid in requested
        }
        return self._build(
            cutoff=previous.cutoff,
            containers=containers,
            readiness=readiness,
            movement=movement or TrainMovement(previous.train_id, previous.length_m),
            destination_order=destination_order
            if destination_order is not None
            else self._destination_order(previous),
            now=ensure_utc(now),
            confirmed=previous.confirmed,
            preseed=preseed,
            known_eligibility={},
        )

    def release(
        self,
        previous: DeparturePlan,
        container_id: str,
        now: datetime,
    ) -> DeparturePlan:
        """Explicitly surrender a confirmed slot. Only possible before the
        documentation cutoff; afterwards the slot is frozen."""
        if container_id not in previous.confirmed:
            raise NotFoundError(f'container is not confirmed: {container_id}')
        if ensure_utc(now) >= previous.cutoff.documentation_deadline:
            raise ConflictError(
                f'cannot release {container_id}: documentation cutoff has passed'
            )
        kept = [
            cid
            for cid in previous.assignments
            if cid != container_id
        ]
        confirmed = frozenset(c for c in previous.confirmed if c != container_id)
        preseed = {cid: previous.assignments[cid] for cid in confirmed}
        return self._build(
            cutoff=previous.cutoff,
            containers=[previous.box(cid) for cid in kept],
            readiness={},
            movement=TrainMovement(previous.train_id, previous.length_m),
            destination_order=self._destination_order(previous),
            now=ensure_utc(now),
            confirmed=confirmed,
            preseed=preseed,
            # same planning cycle, same train: eligibility rows captured in
            # the previous build remain the evidence for the surviving boxes
            known_eligibility=dict(previous.eligibility),
        )

    @staticmethod
    def _destination_order(previous: DeparturePlan) -> list[str]:
        order: list[str] = []
        for car in previous.cars:
            if car.destination is not None and car.destination not in order:
                order.append(car.destination)
        return order

    # ---------------------------------------------------------------- engine

    def _build(
        self,
        cutoff: TrainCutoff,
        containers: list,
        readiness: dict[str, ContainerReadiness],
        movement: TrainMovement,
        destination_order: list[str],
        now: datetime,
        confirmed: frozenset,
        preseed: dict[str, str],
        known_eligibility: dict[str, dict],
    ) -> DeparturePlan:
        now = ensure_utc(now)
        if cutoff.train_id != self.consist.train_id:
            raise ValidationError('cutoff does not belong to this train')
        if movement.train_id != self.consist.train_id:
            raise ValidationError('movement does not belong to this train')
        if len(set(destination_order)) != len(destination_order):
            raise ValidationError('duplicate destination in sequence')
        rank = {d: i for i, d in enumerate(destination_order)}

        for item in containers:
            if not isinstance(item, RailContainer):
                raise ValidationError(f'expected RailContainer, got {item!r}')
        ids = [c.container_id for c in containers]
        if len(set(ids)) != len(ids):
            raise ValidationError('duplicate container id')
        by_id = {c.container_id: c for c in containers}

        for cid, wagon_id in preseed.items():
            if cid not in by_id:
                raise ConflictError(f'confirmed container {cid} missing from rearrangement')
            if wagon_id not in {w.wagon_id for w in self.wagons}:
                raise ConflictError(f'confirmed slot {wagon_id} no longer exists')
            if cid not in confirmed:
                raise ConflictError(f'seeded slot for {cid} is not confirmed')

        # ---- shadow packing state: nothing observable is touched yet ----
        loads: list[WagonLoad] = []
        for car, wagon in zip(self.consist.cars, self.wagons):
            car_payload = car.gross_limit_kg - car.tare_kg
            if car_payload <= 0:
                raise CapacityError(f'car {car.car_id} has no payload capacity')
            shadow = Wagon(
                wagon.wagon_id,
                min(wagon.max_weight_kg, car_payload),
                wagon.slot_equivalent_ft,
                wagon.destination,
                wagon.car_id,
            )
            loads.append(WagonLoad(shadow))
        index_of = {w.wagon_id: i for i, w in enumerate(self.wagons)}

        # ---- cutoff eligibility; fail closed when readiness is absent ----
        eligibility: dict[str, dict] = {}
        for container in containers:
            cid = container.container_id
            record = readiness.get(cid)
            if record is None and cid in known_eligibility:
                eligibility[cid] = known_eligibility[cid]
                continue
            if record is None:
                raise SafetyConstraintError(f'no readiness information for {cid}')
            if record.container_id != cid:
                raise ValidationError(
                    f'readiness record {record.container_id} does not match container {cid}'
                )
            row = self.checker.evaluate(cutoff, record)
            eligibility[cid] = row
            if not row['eligible']:
                if row['documentation_ok'] != row['physical_ok']:
                    kind = (
                        'documentation cutoff'
                        if not row['documentation_ok']
                        else 'physical cutoff'
                    )
                    raise ConflictError(f'{cid} misses {kind}')
                raise ConflictError(f'{cid} misses documentation and physical cutoffs')

        assignments: dict[str, str] = {}

        def can_place(load: WagonLoad, container: RailContainer) -> bool:
            if not load.can_add(container):
                return False
            # WagonLoad only enforces a wagon's fixed destination; a wagon
            # opened during this build must stay single-destination too.
            return all(c.destination == container.destination for c in load.containers)

        def place(container: RailContainer, forced_wagon_id: str | None) -> None:
            if container.destination not in rank:
                raise ValidationError(
                    f'unknown destination in sequence: {container.destination}'
                )
            if forced_wagon_id is not None:
                load = loads[index_of[forced_wagon_id]]
                if not can_place(load, container):
                    raise ConflictError(
                        f'confirmed slot for {container.container_id} on {forced_wagon_id} '
                        'no longer holds (destination, weight or length)'
                    )
                load.add(container)
                assignments[container.container_id] = forced_wagon_id
                return
            candidates = [load for load in loads if can_place(load, container)]
            if not candidates:
                raise CapacityError(
                    f'no car capacity for {container.container_id} to {container.destination}'
                )
            # Prefer a car already opened for this destination (consolidate),
            # then the earliest free car so destinations stay in sequence.
            chosen = min(
                candidates,
                key=lambda load: (
                    0 if load.containers else 1,
                    -load.used_weight_kg,
                    index_of[load.wagon.wagon_id],
                    load.wagon.wagon_id,
                ),
            )
            chosen.add(container)
            assignments[container.container_id] = chosen.wagon.wagon_id

        # confirmed slots first, in train order; they are immovable seed load
        seeded = sorted(
            (cid for cid in preseed if cid in by_id),
            key=lambda cid: index_of[preseed[cid]],
        )
        for cid in seeded:
            place(by_id[cid], preseed[cid])

        # remainder: destination sequence dominates, then priority / weight
        remaining = [c for c in containers if c.container_id not in assignments]
        remaining.sort(
            key=lambda c: (
                rank[c.destination] if c.destination in rank else len(rank),
                -c.priority,
                -c.weight_kg,
                -c.length_ft,
                c.container_id,
            )
        )
        for container in remaining:
            place(container, None)

        self._validate_sequence(loads, rank)

        # ---- train-level gross / length envelope ----
        total_gross = self.consist.tare_kg() + sum(
            load.used_weight_kg for load in loads
        )
        if total_gross > self.consist.max_gross_kg:
            raise CapacityError('train gross weight exceeded')
        if movement.length_m < self.consist.length_m():
            raise ValidationError('declared movement is shorter than the consist')

        # ---- track: select against a shadow view, commit only at the end ----
        track_id, previous_track = self._select_track(movement)

        cars = tuple(
            self._car_load(position, car, load)
            for position, (car, load) in enumerate(zip(self.consist.cars, loads))
        )
        plan = DeparturePlan(
            self.consist.train_id,
            cars,
            MappingProxyType(dict(assignments)),
            track_id,
            cutoff,
            MappingProxyType(eligibility),
            frozenset(c for c in confirmed if c in assignments),
            total_gross,
            self.consist.length_m(),
            MappingProxyType({c.container_id: c for c in containers}),
        )

        # single observable mutation, after every check above has passed
        self._commit_track(movement, track_id, previous_track)
        return plan

    def _validate_sequence(self, loads: list[WagonLoad], rank: dict[str, int]) -> None:
        seen = -1
        for load in loads:
            destinations = {c.destination for c in load.containers}
            if not destinations:
                continue
            if len(destinations) > 1:
                raise ConflictError(f'wagon {load.wagon.wagon_id} mixes destinations')
            current = rank[next(iter(destinations))]
            if current < seen:
                raise ConflictError('car order would require intermediate shunting')
            seen = current

    @staticmethod
    def _car_load(position: int, car, load: WagonLoad) -> CarLoad:
        used = load.used_weight_kg
        if used > car.gross_limit_kg - car.tare_kg:
            raise CapacityError(f'car {car.car_id} gross limit exceeded')
        destination = load.containers[0].destination if load.containers else None
        return CarLoad(
            car.car_id,
            load.wagon.wagon_id,
            position,
            tuple(c.container_id for c in load.containers),
            destination,
            used,
            car.tare_kg + used,
            load.used_length_ft,
        )

    def _select_track(self, movement: TrainMovement) -> tuple[str, str | None]:
        def usable(track: YardTrack) -> bool:
            if track.usable_length_m < movement.length_m:
                return False
            if movement.needs_electric and not track.electrified:
                return False
            if movement.needs_service and not track.service_track:
                return False
            return True

        previous_track = None
        for track_id, train_id in self.tracks.occupied.items():
            if train_id == movement.train_id:
                previous_track = track_id
                break

        # staying put is preferred whenever the current track still fits
        if previous_track is not None and usable(self.tracks.tracks[previous_track]):
            return previous_track, previous_track

        candidates = [
            (track.usable_length_m - movement.length_m, track.track_id)
            for track in self.tracks.tracks.values()
            if track.track_id not in self.tracks.occupied and usable(track)
        ]
        if not candidates:
            raise CapacityError('no compatible yard track')
        _, track_id = min(candidates)
        return track_id, previous_track

    def _commit_track(
        self,
        movement: TrainMovement,
        track_id: str,
        previous_track: str | None,
    ) -> None:
        if previous_track is not None and previous_track != track_id:
            self.tracks.occupied.pop(previous_track, None)
        holder = self.tracks.occupied.get(track_id)
        if holder is not None and holder != movement.train_id:
            # restore nothing: another train winning a shadow-free track would
            # mean concurrent mutation, which is a conflict rather than a
            # partial load of ours
            raise ConflictError(f'track {track_id} occupied by {holder}')
        self.tracks.occupied[track_id] = movement.train_id
