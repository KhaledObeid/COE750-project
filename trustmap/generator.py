"""House generator (Point 3a, engineering parameters locked 2026-11-18).

Builds one house per seed, in the ProcTHOR room-spec style simplified by
Khaled's rule "the simplest option that works":

  1. Room types: the four core rooms plus the fifth room (3b), drawn once.
  2. Footprint: a rectangle, width and depth each uniform in [8, 12] m.
  3. Zones: the rectangle is cut across its longer side into the public zone
     (kitchen, living, dining if present) and the private zone (bedrooms,
     bathroom, office if present), with areas in proportion to the summed
     growth weights of their rooms.
  4. Rooms: each zone is cut across its own longer side into one strip per
     room, widths in proportion to the growth weights.
  5. Rejection: a room narrower than 2 m, or a house whose doors cannot be
     placed under the rules below, is thrown away and the geometry redrawn
     (rejection sampling).
  6. Doors: 1.0 m wide, centred on the shared wall, forming a tree (exactly
     one route between any two rooms). Public rooms are linked along their
     zone; each private room gets one door to a public neighbour if it
     touches one, otherwise to an already connected private neighbour that is
     not the bathroom. The bathroom therefore has exactly one door.
  7. Grid: 0.10 m cells, 0.20 m walls; walls inflated by the robot radius of
     0.20 m for planning (a door leaves a 0.6 m passage).
  8. Viewpoints: the free cell (after inflation) nearest each room centre.
  9. Distances: the 5 x 5 table of 8-connected A*/Dijkstra lengths between
     viewpoints.

All geometry is kept in whole grid cells (integers), so walls are exactly two
cells thick and doors exactly ten cells wide, with no rounding surprises.

Small choices not spelled out in DECISIONS.md (logged in OPEN_QUESTIONS Q9):
the public zone always takes the low-coordinate side (a mirrored house has
the same distances, so this changes nothing); the room order inside a zone is
shuffled; a private room touching several public rooms gets its door to a
random one of them; two rooms count as neighbours only if their shared wall
is at least 1.2 m long, so a centred 1.0 m door stays clear of the corner
walls; on rejection only the geometry is redrawn, not the room types, so the
3b fifth-room shares stay exactly 50/25/25.
"""

import math
from dataclasses import dataclass

import numpy as np

from trustmap.camera import DETECTION_RANGE_M, target_visible
from trustmap.grid import OccupancyGrid, inflate
from trustmap.house import House, draw_room_types
from trustmap.planning import distance_table

# Locked parameters (DECISIONS.md, engineering parameters, 2026-11-18).
FOOTPRINT_RANGE_M = (8.0, 12.0)
GROWTH_WEIGHTS = {"living": 3.0, "kitchen": 2.0, "dining": 2.0, "bedroom": 2.0, "office": 1.5, "bathroom": 1.0}
PUBLIC_TYPES = ("kitchen", "living", "dining")
MIN_ROOM_WIDTH_M = 2.0
DOOR_WIDTH_M = 1.0
CELL_SIZE_M = 0.10
WALL_THICKNESS_M = 0.20
ROBOT_RADIUS_M = 0.20
CONNECTIVITY = 8

# Derived, in whole cells.
_CELLS_PER_M = round(1.0 / CELL_SIZE_M)
_DOOR_CELLS = round(DOOR_WIDTH_M / CELL_SIZE_M)                   # 10
_HALF_WALL_CELLS = round(WALL_THICKNESS_M / 2 / CELL_SIZE_M)      # 1
_MIN_SHARED_WALL_CELLS = _DOOR_CELLS + 2 * _HALF_WALL_CELLS       # 12 = 1.2 m
_MARGIN_CELLS = 2 * _HALF_WALL_CELLS                              # grid border around the outer walls
MAX_ATTEMPTS = 10_000

# Bump when anything above changes, so saved pools are rebuilt.
GENERATOR_VERSION = "1"


class HouseRejected(Exception):
    """This draw broke a rule; draw the geometry again."""


@dataclass(frozen=True)
class Room:
    type: str
    rect: tuple  # (x0, y0, x1, y1) in whole cells: wall centre lines of the room


@dataclass(frozen=True)
class Door:
    rooms: tuple     # (room index a, room index b)
    vertical: bool   # True if the wall is vertical (constant x)
    line: int        # the wall's x (vertical) or y (horizontal) in cells
    center: int      # door centre along the wall, in cells


@dataclass(frozen=True)
class HouseLayout:
    """Everything the generator produces for one seed."""

    seed: int
    width_m: float
    depth_m: float
    rooms: tuple
    doors: tuple
    grid: OccupancyGrid           # walls only, used by the camera
    planning_grid: OccupancyGrid  # walls inflated by the robot radius, used by A*
    viewpoints: tuple             # (row, col) per room, on the planning grid
    house: House                  # room types and the planned-distance table

    def viewpoint_xy(self, room):
        return self.grid.cell_to_world(self.viewpoints[room])

    def room_cells(self, room):
        """(row, col) of every floor cell inside the room's walls."""
        x0, y0, x1, y1 = self.rooms[room].rect
        m, h = _MARGIN_CELLS, _HALF_WALL_CELLS
        return [(r, c) for r in range(y0 + h + m, y1 - h + m) for c in range(x0 + h + m, x1 - h + m)]


# ---------------------------------------------------------------- layout ----

def _split(start, end, weights):
    """Cut [start, end) into len(weights) whole-cell pieces in proportion to the weights."""
    total = sum(weights)
    cuts = [start]
    acc = 0.0
    for w in weights[:-1]:
        acc += w
        cuts.append(start + round((end - start) * acc / total))
    cuts.append(end)
    return list(zip(cuts[:-1], cuts[1:]))


def _cut_across_longer_side(rect, weights):
    """Cut a rectangle into strips across its longer side, widths by weight."""
    x0, y0, x1, y1 = rect
    if x1 - x0 >= y1 - y0:  # longer side along x: cut with vertical lines
        return [(a, y0, b, y1) for a, b in _split(x0, x1, weights)]
    return [(x0, a, x1, b) for a, b in _split(y0, y1, weights)]


def _shared_wall(a, b):
    """Shared wall of two room rectangles as (vertical, line, lo, hi), or None."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 == bx0 or bx1 == ax0:
        line = ax1 if ax1 == bx0 else ax0
        lo, hi = max(ay0, by0), min(ay1, by1)
        return (True, line, lo, hi) if hi > lo else None
    if ay1 == by0 or by1 == ay0:
        line = ay1 if ay1 == by0 else ay0
        lo, hi = max(ax0, bx0), min(ax1, bx1)
        return (False, line, lo, hi) if hi > lo else None
    return None


def _neighbours(rooms):
    """Pairs of rooms whose shared wall is long enough for a door: {(i, j): wall}."""
    out = {}
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            wall = _shared_wall(rooms[i].rect, rooms[j].rect)
            if wall and wall[3] - wall[2] >= _MIN_SHARED_WALL_CELLS:
                out[(i, j)] = out[(j, i)] = wall
    return out


def _door(i, j, wall):
    vertical, line, lo, hi = wall
    return Door(rooms=(i, j), vertical=vertical, line=line, center=(lo + hi) // 2)


def _place_doors(rooms, public_order, private_rooms, rng):
    """Door tree following the locked rules; raises HouseRejected if impossible."""
    nb = _neighbours(rooms)
    doors = []
    # Public rooms: link each strip to the next one along the zone.
    for a, b in zip(public_order[:-1], public_order[1:]):
        if (a, b) not in nb:
            raise HouseRejected("public strips do not share a door-wide wall")
        doors.append(_door(a, b, nb[(a, b)]))
    connected = set(public_order)
    # Private rooms touching a public room: one door to a public neighbour.
    pending = []
    for k in private_rooms:
        public_nb = [p for p in public_order if (k, p) in nb]
        if public_nb:
            p = public_nb[rng.integers(len(public_nb))]
            doors.append(_door(p, k, nb[(p, k)]))
            connected.add(k)
        else:
            pending.append(k)
    # The others: one door to an already connected private room that is not the bathroom.
    while pending:
        progress = False
        for k in list(pending):
            options = [q for q in private_rooms
                       if q in connected and rooms[q].type != "bathroom" and (k, q) in nb]
            if options:
                q = options[rng.integers(len(options))]
                doors.append(_door(q, k, nb[(q, k)]))
                connected.add(k)
                pending.remove(k)
                progress = True
        if not progress:
            raise HouseRejected("a private room cannot be connected")
    return doors


def _draw_geometry(room_types, rng):
    """One attempt at the room rectangles and doors for the given room types."""
    lo, hi = FOOTPRINT_RANGE_M
    w = round(rng.uniform(lo, hi) * _CELLS_PER_M)
    d = round(rng.uniform(lo, hi) * _CELLS_PER_M)

    public = [i for i, t in enumerate(room_types) if t in PUBLIC_TYPES]
    private = [i for i, t in enumerate(room_types) if t not in PUBLIC_TYPES]
    public = [public[k] for k in rng.permutation(len(public))]    # random order inside the zone
    private = [private[k] for k in rng.permutation(len(private))]

    zone_weights = [sum(GROWTH_WEIGHTS[room_types[i]] for i in zone) for zone in (public, private)]
    public_zone, private_zone = _cut_across_longer_side((0, 0, w, d), zone_weights)

    rects = {}
    for zone, zone_rect in ((public, public_zone), (private, private_zone)):
        strips = _cut_across_longer_side(zone_rect, [GROWTH_WEIGHTS[room_types[i]] for i in zone])
        rects.update(zip(zone, strips))
    rooms = [Room(room_types[i], rects[i]) for i in range(len(room_types))]

    min_cells = MIN_ROOM_WIDTH_M * _CELLS_PER_M
    for r in rooms:
        x0, y0, x1, y1 = r.rect
        if min(x1 - x0, y1 - y0) < min_cells - 1e-9:
            raise HouseRejected("room narrower than 2 m")
    doors = _place_doors(rooms, public, private, rng)
    return w, d, rooms, doors


# ------------------------------------------------------------------ grid ----

def _rasterize(w, d, rooms, doors):
    """Occupancy grid: floor inside every room, walls on the room borders, door openings."""
    m, h = _MARGIN_CELLS, _HALF_WALL_CELLS
    occupied = np.ones((d + 2 * m, w + 2 * m), dtype=bool)  # [row = y, col = x]; outside is blocked
    for r in rooms:
        x0, y0, x1, y1 = r.rect
        occupied[y0 + h + m:y1 - h + m, x0 + h + m:x1 - h + m] = False
    half_door = _DOOR_CELLS // 2
    for door in doors:
        a, b = door.center - half_door, door.center + half_door
        if door.vertical:   # opening through the two wall cells either side of x = line
            occupied[a + m:b + m, door.line - h + m:door.line + h + m] = False
        else:
            occupied[door.line - h + m:door.line + h + m, a + m:b + m] = False
    origin = (-m * CELL_SIZE_M, -m * CELL_SIZE_M)
    return OccupancyGrid(occupied, CELL_SIZE_M, origin)


def _viewpoint(planning_grid, room_rect, room_cells):
    """Free cell (after inflation) nearest the room centre."""
    x0, y0, x1, y1 = room_rect
    cx = (x0 + x1) / 2 + _MARGIN_CELLS  # room centre in grid units
    cy = (y0 + y1) / 2 + _MARGIN_CELLS
    free = [c for c in room_cells if planning_grid.is_free(c)]
    if not free:
        raise HouseRejected("room has no free cell after inflation")
    return min(free, key=lambda rc: ((rc[1] + 0.5 - cx) ** 2 + (rc[0] + 0.5 - cy) ** 2, rc))


# --------------------------------------------------------------- public -----

def generate_house(seed):
    """The house of one seed. The same seed always gives the same house."""
    rng = np.random.default_rng(seed)
    room_types = draw_room_types(rng)
    for _ in range(MAX_ATTEMPTS):
        try:
            w, d, rooms, doors = _draw_geometry(room_types, rng)
        except HouseRejected:
            continue
        grid = _rasterize(w, d, rooms, doors)
        planning_grid = inflate(grid, ROBOT_RADIUS_M)
        layout = HouseLayout(seed, w / _CELLS_PER_M, d / _CELLS_PER_M, tuple(rooms), tuple(doors),
                             grid, planning_grid, (), None)
        viewpoints = tuple(_viewpoint(planning_grid, r.rect, layout.room_cells(i)) for i, r in enumerate(rooms))
        distances = distance_table(planning_grid, viewpoints, CONNECTIVITY)
        house = House(room_types=room_types, distances=distances, seed=seed)
        return HouseLayout(seed, layout.width_m, layout.depth_m, layout.rooms, layout.doors,
                           grid, planning_grid, viewpoints, house)
    raise RuntimeError(f"seed {seed}: no valid house after {MAX_ATTEMPTS} attempts")


def object_candidates(layout, room):
    """Cells where the target may be placed in `room` (Q4, locked 2026-11-18).

    A floor cell of the room, visible from the room's viewpoint (within 1.5 m,
    clear line of sight on the wall grid), at least the robot radius away from
    the viewpoint so the object is not under the robot, and not visible from
    any other room's viewpoint, so detection is certain in the object's room
    and impossible elsewhere, exactly as the optimum assumes.
    """
    full_circle = 2.0 * math.pi  # the robot turns on the spot: a 360-degree look
    own = layout.viewpoint_xy(room)
    others = [layout.viewpoint_xy(k) for k in range(len(layout.rooms)) if k != room]
    out = []
    for cell in layout.room_cells(room):
        xy = layout.grid.cell_to_world(cell)
        if math.dist(xy, own) < ROBOT_RADIUS_M:
            continue
        if not target_visible(layout.grid, (*own, 0.0), xy, full_circle, DETECTION_RANGE_M):
            continue
        if any(target_visible(layout.grid, (*o, 0.0), xy, full_circle, DETECTION_RANGE_M) for o in others):
            continue
        out.append(cell)
    return out


def place_object(layout, room, rng):
    """World (x, y) of the target in `room`: a random cell from object_candidates."""
    cells = object_candidates(layout, room)
    if not cells:
        raise RuntimeError(f"seed {layout.seed}: no valid object cell in room {room}")
    return layout.grid.cell_to_world(cells[rng.integers(len(cells))])
