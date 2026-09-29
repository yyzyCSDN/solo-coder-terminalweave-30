from __future__ import annotations
from dataclasses import dataclass
from terminalweave.core.errors import ValidationError, CapacityError, ConflictError

@dataclass(frozen=True)
class YardTrack:
    track_id: str
    usable_length_m: float
    electrified: bool = False
    service_track: bool = False

    def __post_init__(self):
        if self.usable_length_m <= 0:
            raise ValidationError('track length must be positive')

@dataclass(frozen=True)
class TrainMovement:
    train_id: str
    length_m: float
    needs_electric: bool = False
    needs_service: bool = False

class TrackAllocator:

    def __init__(self, tracks: list[YardTrack]):
        self.tracks = {t.track_id: t for t in tracks}
        self.occupied: dict[str, str] = {}

    def _compatible(self, t: YardTrack, m: TrainMovement) -> bool:
        if t.usable_length_m < m.length_m:
            return False
        if m.needs_electric and not t.electrified:
            return False
        if m.needs_service and not t.service_track:
            return False
        return True

    def allocate(self, m: TrainMovement) -> str:
        candidates = []
        for t in self.tracks.values():
            if t.track_id in self.occupied:
                continue
            if not self._compatible(t, m):
                continue
            candidates.append((t.usable_length_m - m.length_m, t.track_id))
        if not candidates:
            raise CapacityError('no compatible yard track')
        _, track = min(candidates)
        self.occupied[track] = m.train_id
        return track

    def reserve(self, track_id: str, m: TrainMovement) -> str:
        """Reserve a specific, already-chosen compatible track.

        Used to atomically book a track that was selected during planning
        without re-running best-fit at confirmation time.
        """
        track = self.tracks.get(track_id)
        if track is None:
            raise ValidationError(f'unknown yard track {track_id}')
        current = self.occupied.get(track_id)
        if current is not None and current != m.train_id:
            raise ConflictError(f'track {track_id} occupied by {current}')
        if not self._compatible(track, m):
            raise CapacityError(f'track {track_id} cannot host train {m.train_id}')
        self.occupied[track_id] = m.train_id
        return track_id

    def release(self, track_id: str) -> str | None:
        return self.occupied.pop(track_id, None)
