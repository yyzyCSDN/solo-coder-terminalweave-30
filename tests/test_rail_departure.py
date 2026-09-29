from datetime import datetime, timedelta, timezone
import pytest

from terminalweave.core.errors import (
    ValidationError,
    CapacityError,
    ConflictError,
    SafetyConstraintError,
)
from terminalweave.rail.blocks import RailContainer, Wagon
from terminalweave.rail.train import RailCar, TrainConsist
from terminalweave.rail.cutoff import TrainCutoff, ContainerReadiness
from terminalweave.rail.tracks import TrackAllocator, YardTrack, TrainMovement
from terminalweave.rail.departure import DeparturePlanner

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)


def car(cid='R1', tare=20000.0, gross=90000.0, length=10.0):
    return RailCar(cid, tare, gross, length)


def consist(cars=None, max_gross=180000.0, max_length=40.0):
    return TrainConsist('T1', max_length, max_gross, cars or [car(), car('R2')])


def wagon(wid, cid, max_weight=70000.0, slots=80, destination=None):
    return Wagon(wid, max_weight, slots, destination, cid)


def planner(cars_consist=None, wagons=None, tracks=None):
    cars_consist = cars_consist or consist()
    wagons = wagons or [wagon('W1', 'R1'), wagon('W2', 'R2')]
    tracks = tracks or TrackAllocator([YardTrack('TRK1', 100.0), YardTrack('TRK2', 50.0)])
    return DeparturePlanner(cars_consist, wagons, tracks), tracks


def cutoff(departure=None, doc_hours=4, phys_hours=2):
    return TrainCutoff(
        'T1',
        departure or D + timedelta(hours=24),
        timedelta(hours=doc_hours),
        timedelta(hours=phys_hours),
    )


def ready(cid, doc_at=None, yard_at=None):
    return ContainerReadiness(
        cid, doc_at or D + timedelta(hours=1), yard_at or D + timedelta(hours=2)
    )


def boxes():
    return [
        RailContainer('C1', 'A', 20000.0, 40),
        RailContainer('C2', 'A', 20000.0, 40),
        RailContainer('C3', 'B', 15000.0, 40),
        RailContainer('C4', 'B', 15000.0, 20),
    ]


def readiness(ids):
    return {cid: ready(cid) for cid in ids}


def movement(length=30.0, **kw):
    return TrainMovement('T1', length, **kw)


def build(p, containers=None, order=('A', 'B'), now=D, mv=None, cut=None):
    containers = boxes() if containers is None else containers
    return p.plan(
        cut or cutoff(),
        containers,
        readiness([c.container_id for c in containers]),
        mv or movement(),
        list(order),
        now,
    )


def test_plan_packs_by_destination_sequence_and_reserves_track():
    p, tracks = planner()
    plan = build(p)
    assert plan.assignments['C1'] == plan.assignments['C2'] == 'W1'
    assert plan.assignments['C3'] == plan.assignments['C4'] == 'W2'
    assert [c.destination for c in plan.cars if c.destination] == ['A', 'B']
    assert plan.cars[0].position == 0 and plan.cars[1].position == 1
    assert tracks.occupied == {'TRK2': 'T1'}
    assert plan.track_id == 'TRK2'
    assert plan.total_gross_kg == 110000.0
    assert set(plan.containers()) == {'C1', 'C2', 'C3', 'C4'}
    assert all(row['eligible'] for row in plan.eligibility.values())


def test_failed_cutoff_leaves_no_partial_load_or_track():
    p, tracks = planner()
    late = boxes() + [RailContainer('C5', 'A', 20000.0, 40)]
    data = readiness([c.container_id for c in late])
    data['C5'] = ContainerReadiness('C5', D + timedelta(hours=1), D + timedelta(hours=23))
    with pytest.raises(ConflictError):
        p.plan(cutoff(), late, data, movement(), ['A'], D)
    assert tracks.occupied == {}
    # planner is still usable: shadow attempt mutated nothing
    ok = build(p)
    assert tracks.occupied == {'TRK2': 'T1'}
    assert set(ok.containers()) == {'C1', 'C2', 'C3', 'C4'}


def test_packing_failure_is_atomic():
    p, tracks = planner()
    too_many = boxes() + [
        RailContainer('C5', 'B', 20000.0, 40),
        RailContainer('C6', 'B', 20000.0, 40),
    ]
    with pytest.raises(CapacityError):
        build(p, too_many)
    assert tracks.occupied == {}


def test_missing_readiness_fails_closed():
    p, tracks = planner()
    with pytest.raises(SafetyConstraintError):
        p.plan(cutoff(), boxes(), {'C1': ready('C1')}, movement(), ['A', 'B'], D)
    assert tracks.occupied == {}


def test_car_payload_capacity_tightens_wagon_capacity():
    cs = consist([car('R1', gross=45000.0), car('R2', gross=45000.0)])
    p, tracks = planner(cs, [wagon('W1', 'R1'), wagon('W2', 'R2')])
    heavy = [RailContainer('H1', 'A', 28000.0, 40)]  # wagon allows 70t, cars only 25t
    with pytest.raises(CapacityError):
        build(p, heavy, order=('A',))
    assert tracks.occupied == {}


def test_train_gross_envelope_enforced():
    cs = consist(max_gross=100000.0)  # tare 40000 + load 70000
    p, tracks = planner(cs)
    with pytest.raises(CapacityError):
        build(p)
    assert tracks.occupied == {}


def test_unknown_destination_rejected():
    p, _ = planner()
    with pytest.raises(ValidationError):
        build(p, order=('A',))  # B boxes not in the sequence


def test_destination_mixing_into_one_car_rejected():
    cs = consist([car()], max_length=20.0)
    p, tracks = planner(cs, [wagon('W1', 'R1', slots=200)])
    with pytest.raises(CapacityError):
        build(p, [RailContainer('X1', 'A', 10000.0, 40),
                  RailContainer('X2', 'B', 10000.0, 40)])
    assert tracks.occupied == {}


def test_wagon_fixed_destination_respected():
    p, _ = planner(wagons=[wagon('W1', 'R1', destination='A'),
                           wagon('W2', 'R2', destination='B')])
    plan = build(p)
    assert plan.assignments['C1'] == 'W1' and plan.assignments['C3'] == 'W2'


def test_wagon_car_pairing_validated():
    cs = consist()
    tracks = TrackAllocator([YardTrack('TRK1', 100.0)])
    with pytest.raises(ValidationError):
        DeparturePlanner(cs, [wagon('W1', 'R1')], tracks)
    with pytest.raises(ValidationError):
        DeparturePlanner(cs, [Wagon('W1', 70000.0, 80, None, None),
                              wagon('W2', 'R2')], tracks)
    with pytest.raises(ValidationError):
        DeparturePlanner(cs, [wagon('W1', 'RX'), wagon('W2', 'R2')], tracks)
    with pytest.raises(ValidationError):
        DeparturePlanner(cs, [wagon('W1', 'R1'), wagon('W2', 'R1')], tracks)


def test_confirmed_slot_is_pinned_during_replan():
    p, tracks = planner()
    plan = build(p).confirm('C1')
    assert 'C1' in plan.confirmed
    added = [RailContainer('C5', 'B', 10000.0, 20)]
    plan2 = p.replan(plan, boxes() + added, readiness(['C1', 'C2', 'C3', 'C4', 'C5']), D + timedelta(hours=3))
    assert plan2.assignments['C1'] == 'W1'
    c1_car = next(c for c in plan2.cars if 'C1' in c.containers)
    assert c1_car.car_id == 'R1' and c1_car.position == 0
    assert 'C1' in plan2.confirmed
    assert 'C5' in plan2.assignments and 'C5' not in plan2.confirmed
    assert tracks.occupied == {'TRK2': 'T1'}  # stayed put


def test_replan_cannot_silently_drop_confirmed_box():
    p, tracks = planner()
    plan = build(p).confirm('C1')
    with pytest.raises(ConflictError):
        p.replan(plan, boxes()[1:], readiness(['C2', 'C3', 'C4']), D + timedelta(hours=3))
    # prior plan and reservation remain intact
    assert plan.assignments['C1'] == 'W1'
    assert tracks.occupied == {'TRK2': 'T1'}


def test_failed_replan_keeps_previous_track_and_plan():
    p, tracks = planner()
    plan = build(p).confirm('C1')
    too_many = boxes() + [
        RailContainer('C5', 'B', 20000.0, 40),
        RailContainer('C6', 'B', 20000.0, 40),
    ]
    with pytest.raises(CapacityError):
        p.replan(plan, too_many,
                 readiness(['C1', 'C2', 'C3', 'C4', 'C5', 'C6']),
                 D + timedelta(hours=3))
    assert tracks.occupied == {'TRK2': 'T1'}
    assert set(plan.containers()) == {'C1', 'C2', 'C3', 'C4'}


def test_release_before_doc_cutoff_rebuilds_after_cutoff_is_blocked():
    p, tracks = planner()
    plan = build(p).confirm('C1')
    rebuilt = p.release(plan, 'C1', D + timedelta(hours=2))
    assert 'C1' not in rebuilt.assignments
    assert set(rebuilt.containers()) == {'C2', 'C3', 'C4'}
    assert 'C1' not in rebuilt.confirmed

    with pytest.raises(ConflictError):
        p.release(plan, 'C1', D + timedelta(hours=21))  # doc deadline 20:00
    assert tracks.occupied == {'TRK2': 'T1'}


def test_replan_near_cutoff_rechecks_readiness_against_locked_box():
    p, tracks = planner()
    plan = build(p).confirm('C1')
    late = dict(readiness(['C1', 'C2', 'C3', 'C4']))
    late['C1'] = ContainerReadiness('C1', D + timedelta(hours=1), D + timedelta(hours=23))
    with pytest.raises(ConflictError):
        p.replan(plan, boxes(), late, D + timedelta(hours=5))
    assert tracks.occupied == {'TRK2': 'T1'}


def test_replan_moves_to_another_track_when_current_no_longer_fits():
    short = TrackAllocator([YardTrack('TRK1', 30.0), YardTrack('TRK2', 100.0)])
    cs = consist()
    p = DeparturePlanner(cs, [wagon('W1', 'R1'), wagon('W2', 'R2')], short)
    plan = p.plan(cutoff(), boxes(), readiness(['C1', 'C2', 'C3', 'C4']),
                  TrainMovement('T1', 30.0), ['A', 'B'], D)
    assert plan.track_id == 'TRK1'
    longer_cars = TrainConsist('T1', 60.0, 180000.0, [car(length=20.0), car('R2', length=20.0)])
    p2 = DeparturePlanner(longer_cars, [wagon('W1', 'R1'), wagon('W2', 'R2')], short)
    plan2 = p2.replan(
        plan, boxes(), readiness(['C1', 'C2', 'C3', 'C4']),
        D + timedelta(hours=2),
        movement=TrainMovement('T1', 40.0),
    )
    assert plan2.track_id == 'TRK2'
    assert short.occupied == {'TRK2': 'T1'}
    assert 'TRK1' not in short.occupied


def test_no_compatible_track_is_atomic():
    p, tracks = planner(
        tracks=TrackAllocator([YardTrack('TRK1', 20.0)]))
    with pytest.raises(CapacityError):
        build(p)
    assert tracks.occupied == {}


def test_electric_requirement_enforced():
    tracks = TrackAllocator([YardTrack('TRK1', 100.0, electrified=False)])
    p = DeparturePlanner(consist(), [wagon('W1', 'R1'), wagon('W2', 'R2')], tracks)
    with pytest.raises(CapacityError):
        p.plan(cutoff(), boxes(), readiness(['C1', 'C2', 'C3', 'C4']),
               movement(needs_electric=True), ['A', 'B'], D)
    assert tracks.occupied == {}
