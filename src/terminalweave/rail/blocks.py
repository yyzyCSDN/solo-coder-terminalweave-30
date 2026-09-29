from __future__ import annotations
from dataclasses import dataclass, field
from terminalweave.core.errors import ValidationError, CapacityError, ConflictError

@dataclass(frozen=True)
class RailContainer:
    container_id: str
    destination: str
    weight_kg: float
    length_ft: int
    priority: int = 0

    def __post_init__(self) -> None:
        if self.weight_kg <= 0 or self.length_ft not in {20, 40, 45, 48, 53}:
            raise ValidationError('invalid rail container')

@dataclass(frozen=True)
class Wagon:
    wagon_id: str
    max_weight_kg: float
    slot_equivalent_ft: int
    destination: str | None = None
    car_id: str | None = None

    def __post_init__(self) -> None:
        if self.max_weight_kg <= 0 or self.slot_equivalent_ft <= 0:
            raise ValidationError('invalid wagon')

@dataclass
class WagonLoad:
    wagon: Wagon
    containers: list[RailContainer] = field(default_factory=list)

    @property
    def used_weight_kg(self) -> float:
        return sum((c.weight_kg for c in self.containers))

    @property
    def used_length_ft(self) -> int:
        return sum((c.length_ft for c in self.containers))

    def can_add(self, container: RailContainer) -> bool:
        if self.wagon.destination and container.destination != self.wagon.destination:
            return False
        return self.used_weight_kg + container.weight_kg <= self.wagon.max_weight_kg and self.used_length_ft + container.length_ft <= self.wagon.slot_equivalent_ft

    def add(self, container: RailContainer) -> None:
        if not self.can_add(container):
            raise CapacityError('container exceeds wagon constraints')
        self.containers.append(container)

class RailBlockPlanner:

    def __init__(self, wagons: list[Wagon]):
        if len({w.wagon_id for w in wagons}) != len(wagons):
            raise ValidationError('duplicate wagon id')
        self.loads = [WagonLoad(w) for w in wagons]

    def assign(self, containers: list[RailContainer]) -> dict[str, str]:
        if len({c.container_id for c in containers}) != len(containers):
            raise ValidationError('duplicate container id')
        ordered = sorted(containers, key=lambda c: (-c.priority, -c.weight_kg, -c.length_ft, c.container_id))
        assignments = {}
        for container in ordered:
            candidates = [load for load in self.loads if load.can_add(container)]
            if not candidates:
                raise CapacityError(f'no wagon capacity for {container.container_id}')

            def waste(load: WagonLoad):
                weight_left = load.wagon.max_weight_kg - (load.used_weight_kg + container.weight_kg)
                length_left = load.wagon.slot_equivalent_ft - (load.used_length_ft + container.length_ft)
                return (length_left, weight_left, load.wagon.wagon_id)
            chosen = min(candidates, key=waste)
            chosen.add(container)
            assignments[container.container_id] = chosen.wagon.wagon_id
        return assignments

    def validate_sequence(self, destination_order: list[str]) -> None:
        rank = {d: i for i, d in enumerate(destination_order)}
        seen = -1
        for load in self.loads:
            destinations = {c.destination for c in load.containers}
            if not destinations:
                continue
            if len(destinations) > 1:
                raise ConflictError('wagon mixes destinations')
            destination = next(iter(destinations))
            if destination not in rank:
                raise ValidationError(f'unknown destination in sequence: {destination}')
            current = rank[destination]
            if current < seen:
                raise ConflictError('wagon order would require intermediate shunting')
            seen = current

    @property
    def utilization(self) -> float:
        total = sum((load.wagon.slot_equivalent_ft for load in self.loads))
        used = sum((load.used_length_ft for load in self.loads))
        return used / total if total else 0.0
