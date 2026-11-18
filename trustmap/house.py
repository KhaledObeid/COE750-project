"""What a house is, as far as the episode and observation code need it.

This holds only what Points 3 and 4 use: the type of each of the 5 rooms and
the table of planned distances between room viewpoints. The occupancy grid,
the room geometry and the viewpoints themselves will be added together with
the house generator (Point 3a), whose parameters are still open (see
docs/OPEN_QUESTIONS.md).
"""

from dataclasses import dataclass

import numpy as np

from trustmap.cooccurrence import ROOM_TYPES

NUM_ROOMS = 5  # Point 3a: every house has exactly 5 searchable rooms.

# Point 3b: four core rooms in every house plus a variable fifth room.
CORE_ROOM_TYPES = ("kitchen", "living", "bathroom", "bedroom")
FIFTH_ROOM_PROBABILITIES = {"bedroom": 0.50, "office": 0.25, "dining": 0.25}

# Point 3a: house pools are fixed by seed range.
TRAIN_SEEDS = range(0, 5_000)
VALIDATION_SEEDS = range(100_000, 100_200)
TEST_SEEDS = range(200_000, 200_200)
POOL_SEEDS = {"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS, "test": TEST_SEEDS}


def draw_room_types(rng):
    """Draw the 5 room types of a new house (Point 3b, room specs).

    Returns the four core types followed by the fifth room. Which type goes
    to which part of the floor plan is up to the house generator.
    """
    names = list(FIFTH_ROOM_PROBABILITIES)
    probs = list(FIFTH_ROOM_PROBABILITIES.values())
    fifth = names[rng.choice(len(names), p=probs)]
    return CORE_ROOM_TYPES + (fifth,)


@dataclass(frozen=True)
class House:
    """One house: 5 room types and planned distances between room viewpoints.

    distances[i, j] is the A* path length in meters on the occupancy grid
    from room i's viewpoint to room j's viewpoint (Point 3a: precomputed once
    per house so training only does table lookups).
    """

    room_types: tuple
    distances: np.ndarray
    seed: int | None = None

    def __post_init__(self):
        if len(self.room_types) != NUM_ROOMS:
            raise ValueError(f"a house has exactly {NUM_ROOMS} rooms")
        for t in self.room_types:
            if t not in ROOM_TYPES:
                raise ValueError(f"unknown room type {t!r}")
        d = np.asarray(self.distances, dtype=float)
        if d.shape != (NUM_ROOMS, NUM_ROOMS):
            raise ValueError("distances must be a 5x5 matrix")
        if not np.allclose(d, d.T) or np.any(np.diag(d) != 0) or np.any(d + np.eye(NUM_ROOMS) <= 0):
            raise ValueError("distances must be symmetric, zero on the diagonal, positive elsewhere")
        object.__setattr__(self, "distances", d)

    @property
    def max_distance(self):
        """Largest room-to-room planned distance; used to scale features (Point 4a)."""
        return float(self.distances.max())
