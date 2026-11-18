"""Observation builder (Point 4a) and comparison-variant masks (Point 4c).

The observation is a 5 x 5 array: one row per room, in the episode's shuffled
row order, never containing room IDs or room-type names. Columns:

  0 distance    planned distance from the robot's current room viewpoint to this
                room's viewpoint, divided by the largest room-to-room distance in
                the house, so it lies in [0, 1]
  1 searched    1 if this room has been searched, else 0
  2 claimed     1 only on the room the map claims
  3 confidence  displayed confidence c on the claimed row, 0 on the others
  4 habit       co-occurrence weight of the target in this room, normalized over the house

Rooms the robot only drives through do not count as searched (Point 4b); that
rule lives in whatever code updates `searched`, which comes with Point 5.
"""

from enum import Enum

import numpy as np

from trustmap.house import NUM_ROOMS

FEATURES = ("distance", "searched", "claimed", "confidence", "habit")
NUM_FEATURES = len(FEATURES)


class Variant(Enum):
    """The three PPO agents of Point 4c.

    They share one network and differ only by a fixed mask: a 0/1 array the
    observation is multiplied by, so masked inputs are always 0 (zeroed, not
    deleted, so the input size stays the same for all three).
    """

    CONF = "PPO-Conf"        # sees claim and confidence
    BINARY = "PPO-Binary"    # sees the claim, confidence always 0
    NOPRIOR = "PPO-NoPrior"  # claim and confidence always 0 ("commonsense only")


# Short names used on the command line and in run folder names.
VARIANT_KEYS = {"conf": Variant.CONF, "binary": Variant.BINARY, "noprior": Variant.NOPRIOR}

_ZEROED_COLUMNS = {
    Variant.CONF: (),
    Variant.BINARY: ("confidence",),
    Variant.NOPRIOR: ("claimed", "confidence"),
}


def variant_mask(variant):
    """1 x 5 mask of the variant, broadcast over the 5 rows."""
    mask = np.ones(NUM_FEATURES, dtype=np.float32)
    for name in _ZEROED_COLUMNS[variant]:
        mask[FEATURES.index(name)] = 0.0
    return mask


def build_observation(house, episode, current_room, searched, variant=Variant.CONF):
    """Build the 5 x 5 observation.

    current_room: house room index (0-4) where the robot is now.
    searched:     length-5 sequence of booleans in house room order.
    Row i of the result describes house room episode.row_order[i].
    """
    searched = np.asarray(searched, dtype=bool)
    if searched.shape != (NUM_ROOMS,):
        raise ValueError("searched must have one entry per room")

    rows = np.zeros((NUM_ROOMS, NUM_FEATURES), dtype=np.float32)
    for i, room in enumerate(episode.row_order):
        is_claimed = room == episode.claimed_room
        rows[i] = (
            house.distances[current_room, room] / house.max_distance,
            float(searched[room]),
            float(is_claimed),
            episode.c if is_claimed else 0.0,
            episode.habit[room],
        )
    return rows * variant_mask(variant)
