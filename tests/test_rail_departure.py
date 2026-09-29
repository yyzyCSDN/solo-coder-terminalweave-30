"""Tests for the executable rail departure planner."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.errors import (
    CapacityError,
    ConflictError,
    NotFoundError,
    StateTransitionError,
    ValidationError,
)
from terminalweave.rail.blocks import RailContainer, Wagon
from terminalweave.rail.cutoff import ContainerReadiness, TrainCutoff
from terminalweave.rail.departure import (
    REASON_CAPACITY,
    REASON_NOT_READY,
    REASON_SEQUENCE_LOCKED,
    REASON_UNKNOWN_DESTINATION,
    DeparturePlanner,
    DepartureRequest,
)
from terminalweave.rail.tracks import TrainMovement, YardTrack
from terminalweave.rail.train import RailCar, TrainConsist

UTC = timezone.utc
DEP = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _cutoff(doc_hours=4, phys_hours=2, train_id='T1', departure=DEP):
    return TrainCutoff(train_id, departure,
                       timedelta(hours=doc_hours), timedelta(hours=phys_hours))


def _ready(cid, doc_before=6, yard_before=3, departure=DEP):
    return ContainerReadiness(
        cid,
        departure - timedelta(hours=doc_before),
        departure - timedelta(hours=yard_before),
    )


def _tracks():
    return [YardTrack('TR1', 600), YardTrack('TR2', 900, electrified=True)]


def _movement(length_m=500, train_id='T1', **kw):
    return TrainMovement(train_id, length_m, **kw)


def _wagons():
    # Two generic 40ft-slot wagons with realistic weight capacity.
    return [Wagon('W1', 50000, 40), Wagon('W2', 50000, 40, 'B')]


def _request(containers, readiness, *, order=('A', 'B'), wagons=None,
             tracks=None, movement=None, consist=None, train_id='T1',
             cutoff=None):
    return DepartureRequest(
        train_id=train_id,
        cutoff=cutoff or _cutoff(train_id=train_id),
        destination_order=list(order),
        wagons=wagons if wagons is not None else _wagons(),
        containers=containers,
        readiness=readiness,
        movement=movement or _movement(train_id=train_id),
        tracks=tracks if tracks is not None else _tracks(),
        consist=consist,
    )


def _abc_boxes():
    return [
        RailContainer('CA1', 'A', 20000, 20, priority=1),
        RailContainer('CA2', 'A', 18000, 20),
        RailContainer('CB1', 'B', 22000, 20, priority=1),
    ]


def test_plan_packs_in_calling_order_and_picks_best_fit_track():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    plan = planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))

    assert plan.status == 'draft'
    assert plan.revision == 1
    # Smallest fitting track chosen (600 fits the 500m movement).
    assert plan.track_id == 'TR1'
    # Calling order A then B; B is forced into the B-dedicated wagon.
    assert [w.destination for w in plan.wagons if w.container_ids] == ['A', 'B']
    assert plan.wagon_of('CB1') == 'W2'
    a_wagon = plan.wagons[0]
    assert a_wagon.destination == 'A'
    assert set(a_wagon.container_ids) == {'CA1', 'CA2'}
    assert not any(w.locked for w in plan.wagons)
    assert plan.total_payload_kg == 60000
    assert plan.total_length_ft == 60


def test_box_missing_cutoff_is_excluded_but_plan_still_executes():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    readiness = [_ready(c.container_id) for c in boxes]
    # CA2 misses the physical cutoff (yard only 1h before; deadline is 2h).
    readiness[1] = _ready('CA2', doc_before=6, yard_before=1)
    plan = planner.plan(_request(boxes, readiness))

    assert plan.wagon_of('CA2') is None
    excluded = {e.container_id: e for e in plan.excluded}
    assert 'CA2' in excluded
    assert excluded['CA2'].reason == REASON_NOT_READY
    assert excluded['CA2'].documentation_ok is True
    assert excluded['CA2'].physical_ok is False
    # The rest of the train is still loaded and ordered.
    assert plan.wagon_of('CA1') == 'W1'
    assert plan.wagon_of('CB1') == 'W2'


def test_unknown_destination_is_excluded_not_fatal():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes() + [RailContainer('CZ1', 'Z', 5000, 20)]
    readiness = [_ready(c.container_id) for c in boxes]
    plan = planner.plan(_request(boxes, readiness))
    excluded = {e.container_id: e.reason for e in plan.excluded}
    assert excluded == {'CZ1': REASON_UNKNOWN_DESTINATION}
    assert plan.wagon_of('CZ1') is None


def test_eligible_box_without_capacity_fails_without_partial_state():
    planner = DeparturePlanner(_tracks())
    # Two wagons totalling 80ft; three 40ft boxes cannot fit.
    boxes = [
        RailContainer('CA1', 'A', 20000, 40),
        RailContainer('CB1', 'B', 20000, 40),
        RailContainer('CB2', 'B', 20000, 40),
    ]
    readiness = [_ready(c.container_id) for c in boxes]
    with pytest.raises(CapacityError):
        planner.plan(_request(boxes, readiness))
    # No draft, no track occupancy, no partial load survives.
    assert planner.get_plan('T1') is None
    assert planner.occupied_tracks() == {}


def test_hard_capacity_failure_is_priority_driven_and_atomic():
    # A single 40ft slot: the high-priority 40ft box is placed first, leaving
    # no room for the still-eligible low-priority box.  That must raise rather
    # than silently produce a short train.
    planner = DeparturePlanner(_tracks())
    wagons = [Wagon('W1', 50000, 40)]
    boxes = [RailContainer('LOW', 'A', 1000, 20, priority=0),
             RailContainer('HIGH', 'A', 1000, 40, priority=5)]
    tracks = _tracks()
    with pytest.raises(CapacityError):
        planner.plan(_request(boxes, [_ready('LOW'), _ready('HIGH')],
                              order=('A',), wagons=wagons, tracks=tracks))
    assert planner.get_plan('T1') is None
    assert planner.occupied_tracks() == {}


def test_missing_readiness_fails_closed():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    readiness = [_ready('CA1'), _ready('CB1')]  # CA2 missing
    with pytest.raises(ValidationError):
        planner.plan(_request(boxes, readiness))
    assert planner.get_plan('T1') is None
    assert planner.occupied_tracks() == {}


def test_plan_is_immutable_to_callers():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    plan = planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    with pytest.raises(TypeError):
        plan.wagons[0].container_ids[0] = 'HACK'  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.loads['W1']['HACK'] = boxes[0]  # type: ignore[index]


def test_confirm_books_track_and_pins_every_placement():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    confirmed = planner.confirm('T1')

    assert confirmed.is_confirmed
    assert planner.occupied_tracks() == {'TR1': 'T1'}
    assert confirmed.is_locked('CA1')
    assert confirmed.is_locked('CB1')
    assert planner.get_confirmed('T1') is confirmed

    # Cannot start a fresh plan over a confirmed train.
    with pytest.raises(ConflictError):
        planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))


def test_confirm_without_draft_raises_not_found():
    planner = DeparturePlanner(_tracks())
    with pytest.raises(NotFoundError):
        planner.confirm('GHOST')


def test_replan_near_cutoff_pins_wagons_track_and_positions():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    planner.confirm('T1')

    # A later B box tops up the confirmed B wagon in place.
    extra = boxes + [RailContainer('CB2', 'B', 1500, 20)]
    readiness = [_ready(c.container_id) for c in extra]
    replanned = planner.replan('T1', extra, readiness)

    # Wagon order/positions are byte-for-byte preserved.
    assert [w.wagon.wagon_id for w in replanned.wagons] == ['W1', 'W2']
    # Pinned boxes never move.
    assert replanned.wagon_of('CA1') == 'W1'
    assert replanned.wagon_of('CA2') == 'W1'
    assert replanned.wagon_of('CB1') == 'W2'
    # The new box was added into the confirmed B wagon.
    assert replanned.wagon_of('CB2') == 'W2'
    # Track and pin set unchanged; revision bumped.
    assert replanned.track_id == 'TR1'
    assert replanned.pinned == {'CA1', 'CA2', 'CB1'}
    assert replanned.revision == 2
    assert planner.occupied_tracks() == {'TR1': 'T1'}


def test_replan_activates_only_tail_empty_wagon():
    wagons = _wagons() + [Wagon('W3', 50000, 40)]
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(
        boxes, [_ready(c.container_id) for c in boxes],
        order=('A', 'B', 'C'), wagons=wagons))
    planner.confirm('T1')

    extra = boxes + [RailContainer('CC1', 'C', 5000, 20)]
    readiness = [_ready(c.container_id) for c in extra]
    replanned = planner.replan('T1', extra, readiness)

    # W3 is activated at the tail without moving W1/W2.
    assert [w.wagon.wagon_id for w in replanned.wagons] == ['W1', 'W2', 'W3']
    assert replanned.wagon_of('CC1') == 'W3'
    new_wagon = replanned.wagons[2]
    assert new_wagon.destination == 'C'
    assert new_wagon.locked is False


def test_confirming_a_replan_extends_pins_without_rebooking_track():
    wagons = _wagons()
    planner = DeparturePlanner(_tracks())
    boxes = [RailContainer('CA1', 'A', 20000, 20)]
    planner.plan(_request(boxes, [_ready('CA1')], wagons=wagons))
    planner.confirm('T1')

    # Replan adds a B box into the empty W2 at the tail.
    extra = boxes + [RailContainer('CB1', 'B', 10000, 20)]
    readiness = [_ready(c.container_id) for c in extra]
    draft = planner.replan('T1', extra, readiness)
    assert draft.pinned == frozenset({'CA1'})
    assert draft.wagon_of('CB1') == 'W2'
    assert draft.wagons[1].locked is False  # not confirmed yet

    confirmed = planner.confirm('T1')
    # The newly added box is now pinned too, and its wagon is marked locked.
    assert confirmed.pinned == frozenset({'CA1', 'CB1'})
    assert confirmed.wagons[1].locked is True
    assert confirmed.wagons[0].locked is True
    # Track is still held exactly once (confirm did not re-allocate it).
    assert planner.occupied_tracks() == {'TR1': 'T1'}
    assert confirmed.revision == 2


def test_replan_cannot_insert_before_confirmed_tail():
    wagons = _wagons() + [Wagon('W3', 50000, 40)]
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(
        boxes, [_ready(c.container_id) for c in boxes],
        order=('A', 'B', 'C'), wagons=wagons))
    planner.confirm('T1')  # tail is B (rank 1); W3 is empty

    # A new A box would need to go before the confirmed B cars; it must not
    # reshuffle them, so it is reported as sequence-locked.
    extra = boxes + [RailContainer('CA9', 'A', 1000, 20)]
    readiness = [_ready(c.container_id) for c in extra]
    replanned = planner.replan('T1', extra, readiness)

    assert replanned.wagon_of('CA9') is None
    excluded = {e.container_id: e for e in replanned.excluded}
    assert excluded['CA9'].reason == REASON_SEQUENCE_LOCKED
    # Confirmed placements are untouched.
    assert [w.wagon.wagon_id for w in replanned.wagons] == ['W1', 'W2', 'W3']


def test_replan_does_not_move_box_into_a_better_fit_wagon():
    # W1 could hold both A boxes; after confirmation a dedicated A wagon
    # appearing in the input must NOT cause a reshuffle.  Equipment is fixed
    # by the confirmed plan, so we instead prove a confirmed box is immovable
    # even when its wagon has spare capacity and a same-destination empty
    # wagon sits at the tail: a *new* A box still cannot use it to bypass
    # ordering, and the original boxes stay put.
    wagons = [Wagon('W1', 50000, 40, 'A'), Wagon('W2', 50000, 40, 'B'),
              Wagon('W3', 50000, 40, 'A')]
    planner = DeparturePlanner(_tracks())
    boxes = [RailContainer('CA1', 'A', 20000, 20),
             RailContainer('CB1', 'B', 20000, 20)]
    planner.plan(_request(
        boxes, [_ready(c.container_id) for c in boxes],
        order=('A', 'B'), wagons=wagons))
    confirmed = planner.confirm('T1')
    assert confirmed.wagon_of('CA1') == 'W1'

    # The trailing dedicated-A wagon W3 sits AFTER confirmed B cars.  Adding
    # an A box can never be reordered ahead of B, and CA1 is never relocated
    # into W3.
    extra = boxes + [RailContainer('CA2', 'A', 5000, 20)]
    readiness = [_ready(c.container_id) for c in extra]
    replanned = planner.replan('T1', extra, readiness)
    assert replanned.wagon_of('CA1') == 'W1'
    # CA2 tops up W1 in place (same destination, capacity available).
    assert replanned.wagon_of('CA2') == 'W1'


def test_replan_rejects_removed_or_mutated_confirmed_container():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    planner.confirm('T1')

    with pytest.raises(ConflictError):
        planner.replan('T1', boxes[:-1],  # CB1 removed
                       [_ready(c.container_id) for c in boxes[:-1]])

    mutated = [RailContainer('CA1', 'A', 99999, 20),  # weight changed
               boxes[1], boxes[2]]
    with pytest.raises(ConflictError):
        planner.replan('T1', mutated, [_ready(c.container_id) for c in mutated])


def test_replan_without_confirmation_is_rejected():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    with pytest.raises(StateTransitionError):
        planner.replan('T1', boxes, [_ready(c.container_id) for c in boxes])
    with pytest.raises(NotFoundError):
        DeparturePlanner(_tracks()).replan(
            'T1', boxes, [_ready(c.container_id) for c in boxes])


def test_no_compatible_track_fails_atomically():
    # Service track required but neither track is a service track.
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    req = _request(boxes, [_ready(c.container_id) for c in boxes],
                   movement=_movement(needs_service=True))
    with pytest.raises(CapacityError):
        planner.plan(req)
    assert planner.get_plan('T1') is None
    assert planner.occupied_tracks() == {}


def test_consist_gross_overload_fails_atomically():
    # Rail cars with almost no payload headroom.
    cars = [RailCar('C1', tare_kg=30000, gross_limit_kg=31000, length_m=12)]
    consist = TrainConsist('T1', max_length_m=100, max_gross_kg=200000,
                           cars=cars)
    planner = DeparturePlanner(_tracks())
    boxes = [RailContainer('CA1', 'A', 5000, 20)]
    with pytest.raises(CapacityError):
        planner.plan(_request(boxes, [_ready('CA1')], consist=consist))
    assert planner.get_plan('T1') is None


def test_track_occupied_by_another_train_is_not_reselected():
    tracks = _tracks()
    planner = DeparturePlanner(tracks)
    # Occupy TR1 with another train directly through the allocator.
    planner.allocator.allocate(TrainMovement('OTHER', 600))
    boxes = _abc_boxes()
    plan = planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    assert plan.track_id == 'TR2'  # only remaining compatible track


def test_release_frees_track_and_allows_a_new_plan():
    planner = DeparturePlanner(_tracks())
    boxes = _abc_boxes()
    planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    planner.confirm('T1')
    assert planner.release('T1') is True
    assert planner.occupied_tracks() == {}
    assert planner.get_confirmed('T1') is None
    # A fresh plan is possible again.
    plan = planner.plan(_request(boxes, [_ready(c.container_id) for c in boxes]))
    assert plan.status == 'draft'


def test_reserve_rejects_occupied_and_incompatible_track():
    planner = DeparturePlanner(_tracks())
    allocator = planner.allocator
    allocator.allocate(TrainMovement('OTHER', 600))  # books TR1
    with pytest.raises(ConflictError):
        allocator.reserve('TR1', TrainMovement('T1', 10))
    # TR2 is not a service track; a service requirement must fail closed.
    with pytest.raises(CapacityError):
        allocator.reserve('TR2', TrainMovement('T1', 10, needs_service=True))
    # A train longer than the track cannot be forced onto it.
    with pytest.raises(CapacityError):
        allocator.reserve('TR2', TrainMovement('T1', 9999))
    with pytest.raises(ValidationError):
        allocator.reserve('NOPE', TrainMovement('T1', 10))
    # A compatible reservation on the free, electrified TR2 succeeds.
    assert allocator.reserve('TR2', TrainMovement('T2', 10,
                                                  needs_electric=True)) == 'TR2'
