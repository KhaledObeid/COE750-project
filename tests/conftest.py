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
    """Test-only house with random straight-line distances (see trustmap/smoke_houses.py)."""
    from trustmap.smoke_houses import random_house as _random_house

    return _random_house(rng)
