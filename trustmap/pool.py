"""Build, save and load the house pools (Point 3a).

Pools are fixed by seed range: training 0-4,999, validation 100,000-100,199,
test 200,000-200,199. Each house's room types and 5 x 5 planned-distance table
are computed once and saved to data/houses/<pool>.npz, so training only does
table lookups. The full layout (grid, doors, viewpoints) is not saved: it is
rebuilt from the seed with `generate_house(seed)` whenever it is needed
(plots, Gazebo export), and always comes out the same.

Build all pools once (uses all CPU cores; a few minutes):

    python -m trustmap.pool
"""

import argparse
import multiprocessing
import time
from pathlib import Path

import numpy as np

from trustmap.generator import GENERATOR_VERSION, generate_house
from trustmap.house import POOL_SEEDS, House

POOL_DIR = Path(__file__).resolve().parent.parent / "data" / "houses"


def _record(seed):
    house = generate_house(seed).house
    return seed, house.room_types, house.distances


def pool_path(name):
    return POOL_DIR / f"{name}.npz"


def build_pool(name, workers=None, verbose=True):
    seeds = list(POOL_SEEDS[name])
    start = time.time()
    # Several processes generate houses side by side. The __main__ guard in
    # main() is what makes this safe on Windows.
    with multiprocessing.Pool(workers) as pool:
        records = pool.map(_record, seeds, chunksize=16)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pool_path(name),
        version=GENERATOR_VERSION,
        seeds=np.array([r[0] for r in records]),
        room_types=np.array([r[1] for r in records]),
        distances=np.stack([r[2] for r in records]),
    )
    if verbose:
        print(f"{name}: {len(records)} houses in {time.time() - start:.0f} s -> {pool_path(name)}")


def load_pool(name):
    """The saved pool as a list of `House`. Raises if it is missing or out of date."""
    path = pool_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the pools first with: python -m trustmap.pool")
    data = np.load(path)
    if str(data["version"]) != GENERATOR_VERSION:
        raise RuntimeError(f"{path} was built by an older generator; rebuild with: python -m trustmap.pool")
    return [
        House(room_types=tuple(str(t) for t in types), distances=d, seed=int(s))
        for s, types, d in zip(data["seeds"], data["room_types"], data["distances"])
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", nargs="+", choices=list(POOL_SEEDS), default=list(POOL_SEEDS))
    p.add_argument("--workers", type=int, default=None, help="processes to use (default: all cores)")
    a = p.parse_args()
    for name in a.pools:
        build_pool(name, a.workers)


if __name__ == "__main__":
    main()
