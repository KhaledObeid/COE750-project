"""ProcTHOR converter on a hand-made house in ProcTHOR's JSON format (no download needed)."""

import pytest

from trustmap.procthor import Skip, convert

ROOM_TYPES = ("Kitchen", "LivingRoom", "Bedroom", "Bathroom", "Bedroom")
W = 3.0  # each room is a 3 m x 3 m square; the rooms form a row along x


def _wall(room, p0, p1, empty=False):
    (x0, z0), (x1, z1) = p0, p1
    w = {"id": f"wall|{room}|{min(x0, x1)}|{min(z0, z1)}|{max(x0, x1)}|{max(z0, z1)}", "roomId": f"room|{room}",
         "polygon": [{"x": x0, "y": 0, "z": z0}, {"x": x1, "y": 0, "z": z1},
                     {"x": x0, "y": 2.5, "z": z0}, {"x": x1, "y": 2.5, "z": z1}]}
    if empty:
        w["empty"] = True
    return w


def toy_house(doors=True, open_first=False):
    rooms, walls, door_list = [], [], []
    for k, t in enumerate(ROOM_TYPES):
        x0, x1 = k * W, (k + 1) * W
        rooms.append({"id": f"room|{k}", "roomType": t,
                      "floorPolygon": [{"x": x0, "y": 0, "z": 0}, {"x": x0, "y": 0, "z": W},
                                       {"x": x1, "y": 0, "z": W}, {"x": x1, "y": 0, "z": 0}]})
        walls += [_wall(k, (x0, 0), (x0, W), empty=open_first and k == 1),
                  _wall(k, (x0, W), (x1, W)),
                  _wall(k, (x1, W), (x1, 0), empty=open_first and k == 0),
                  _wall(k, (x1, 0), (x0, 0))]
    if doors:
        for k in range(4):
            a = _wall(k, (k * W + W, W), (k * W + W, 0))["id"]
            b = _wall(k + 1, (k * W + W, 0), (k * W + W, W))["id"]
            door_list.append({"id": f"door|{k}|{k + 1}", "room0": f"room|{k}", "room1": f"room|{k + 1}",
                              "wall0": a, "wall1": b, "assetId": "Doorway_1",
                              "holePolygon": [{"x": 1.0, "y": 0, "z": 0}, {"x": 2.0, "y": 2.0, "z": 0}]})
    return {"rooms": rooms, "walls": walls, "doors": door_list}


def test_row_of_rooms_converts_with_our_rules():
    lay = convert(toy_house(), seed=1_000_000)
    assert lay.room_types == ("kitchen", "living", "bedroom", "bathroom", "bedroom")
    d = lay.house.distances
    # Viewpoints sit at the room centres, 3 m apart along a straight corridor of doors.
    for k in range(4):
        assert d[k, k + 1] == pytest.approx(W, abs=0.15)
    assert d[0, 4] == pytest.approx(4 * W, abs=0.3)
    # Walls are two cells (0.20 m) thick: the column of the wall between rooms 0 and 1 is blocked at z = 0.5 m.
    r, c = lay.grid.world_to_cell((W, 0.5))
    assert lay.grid.occupied[r, c - 1] and lay.grid.occupied[r, c]
    assert not lay.grid.occupied[r, c - 2] and not lay.grid.occupied[r, c + 1]
    # The door opening (1.0-2.0 m along the wall) is free.
    r, c = lay.grid.world_to_cell((W, 1.5))
    assert not lay.grid.occupied[r, c - 1] and not lay.grid.occupied[r, c]


def test_empty_wall_is_open_and_no_doors_means_unreachable():
    lay = convert(toy_house(open_first=True), seed=1)
    r, c = lay.grid.world_to_cell((W, 0.5))
    assert not lay.grid.occupied[r, c]          # "empty" wall: no wall drawn
    with pytest.raises(Skip, match="unreachable"):
        convert(toy_house(doors=False), seed=2)


def test_door_at_a_corner_keeps_the_crossing_wall_closed():
    h = toy_house()
    # Move the first door to the end of its wall: 2-3 m from the wall's first point (3, 3) is z = 1..0,
    # so it runs up to the outer wall at z = 0.
    h["doors"][0]["holePolygon"] = [{"x": 2.0, "y": 0, "z": 0}, {"x": 3.0, "y": 2.0, "z": 0}]
    lay = convert(h, seed=3)
    r, c = lay.grid.world_to_cell((W - 0.05, 0.05))   # inside both the door band and the outer wall band
    assert lay.grid.occupied[r, c]
    r, c = lay.grid.world_to_cell((W - 0.05, 0.5))    # the doorway itself is open
    assert not lay.grid.occupied[r, c]
