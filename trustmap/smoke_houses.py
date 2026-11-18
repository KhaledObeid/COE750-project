"""STAND-IN houses for smoke tests and timing only. NOT the Point 3a generator.

The real generator (ProcTHOR-style room specs, occupancy grid, A* distances)
waits on docs/OPEN_QUESTIONS.md Q2-Q5. Until then, these houses let the
environment, the training script and the evaluation run end to end: each has
the five room types of Point 3b and a distance table made from five random
room centers in a 12 m x 10 m box, using straight-line distances instead of
A* paths. Results obtained on them must never be reported.
"""

import numpy as np

from trustmap.house import POOL_SEEDS, House, draw_room_types


def random_house(rng, seed=None):
    pts = rng.uniform((0.0, 0.0), (12.0, 10.0), size=(5, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return House(room_types=draw_room_types(rng), distances=d, seed=seed)


def smoke_house(seed):
    return random_house(np.random.default_rng(seed), seed=seed)


def smoke_pool(name):
    """Stand-in pool with the same seed ranges as the real pools (3a)."""
    return [smoke_house(s) for s in POOL_SEEDS[name]]
