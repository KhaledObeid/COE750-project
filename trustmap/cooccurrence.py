"""Room types, target categories and the co-occurrence table (Point 3b).

The table gives, for each target, a relative weight per room type: how
"usual" it is to find that object there. Weights are not probabilities; they
are turned into probabilities by dividing by their sum over the rooms that
actually exist in the current house ("renormalizing"). Every entry is at least
0.5, so no object is ever impossible in any room.
"""

import numpy as np

ROOM_TYPES = ("kitchen", "living", "bathroom", "bedroom", "office", "dining")

# Column order of COOCCURRENCE rows: same as ROOM_TYPES.
COOCCURRENCE = {
    #                 kitchen living bathroom bedroom office dining
    "mug":            (6.0,   2.0,   0.5,     1.0,    3.0,   3.0),
    "remote_control": (0.5,   8.0,   0.5,     2.0,    0.5,   0.5),
    "towel":          (1.0,   0.5,   8.0,     2.0,    0.5,   0.5),
    "alarm_clock":    (0.5,   1.0,   0.5,     8.0,    2.0,   0.5),
    "laptop":         (1.0,   3.0,   0.5,     3.0,    6.0,   2.0),
    "book":           (0.5,   4.0,   1.0,     4.0,    4.0,   1.0),
    "keys":           (3.0,   3.0,   0.5,     3.0,    2.0,   2.0),
    "cell_phone":     (2.0,   3.0,   1.0,     3.0,    3.0,   2.0),
}

TARGETS = tuple(COOCCURRENCE)

HABIT_STRENGTH = {
    "mug": "strong",
    "remote_control": "strong",
    "towel": "strong",
    "alarm_clock": "strong",
    "laptop": "weak",
    "book": "weak",
    "keys": "weak",
    "cell_phone": "weak",
}

PLACEMENTS = ("table", "uniform")


def room_weights(target, room_types, placement="table"):
    """Raw table weights of `target` for each room of one house.

    `room_types` lists the type of every room in the house, in house order;
    duplicate rooms (two bedrooms) each get the full weight of their type.
    With placement="uniform" the whole table is replaced by ones, as Khaled
    chose on 2026-11-17: this affects the wrong-claim object draw, the table
    branch of the claimed-room draw, and the habit feature alike.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    if placement == "uniform":
        return np.ones(len(room_types))
    row = COOCCURRENCE[target]
    return np.array([row[ROOM_TYPES.index(t)] for t in room_types])


def normalize(weights):
    """Rescale non-negative weights so they sum to 1."""
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()
