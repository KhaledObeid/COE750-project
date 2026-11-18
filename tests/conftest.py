import numpy as np
import pytest

from trustmap.house import House

KITCHEN, LIVING, BATHROOM, BEDROOM_A, BEDROOM_B = range(5)


@pytest.fixture
def two_bedroom_house():
    """Test-only house with made-up distances (not a generated house)."""
    d = np.array([
        [0.0, 4.0, 7.0, 8.0, 9.0],
        [4.0, 0.0, 5.0, 6.0, 7.0],
        [7.0, 5.0, 0.0, 3.0, 4.0],
        [8.0, 6.0, 3.0, 0.0, 2.0],
        [9.0, 7.0, 4.0, 2.0, 0.0],
    ])
    return House(room_types=("kitchen", "living", "bathroom", "bedroom", "bedroom"), distances=d)


def random_house(rng):
    """Test-only house: 5 random room centers in a 12 m x 10 m box, straight-line distances.

    Not a generated house (the Point 3a generator waits on OPEN_QUESTIONS Q2-Q5);
    it only gives the solver and baselines many different distance tables.
    """
    from trustmap.house import draw_room_types

    pts = rng.uniform((0, 0), (12, 10), size=(5, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return House(room_types=draw_room_types(rng), distances=d)
