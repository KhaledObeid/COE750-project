"""Export generated houses to Gazebo Fortress and Nav2 (Points 2, 9e, Q9.7).

    python -m trustmap.gazebo_export 200000 200001

writes, for every house seed, gazebo/worlds/house_<seed>/:
  world.sdf   the house: walls built from exactly the blocked cells of the shared
              occupancy grid (so A*, the virtual camera and Nav2 see the same
              walls as the 2D simulator), floors coloured by room type (visual
              only), no furniture (Q9.7), lights, the robot, and a fixed
              overhead camera for the demo videos (visual only, no effect on
              the robot; DECISIONS "Demo material")
  map.pgm / map.yaml   the same grid as a Nav2 map (Nav2 inflates it itself by the
              robot radius set in gazebo/config/nav2_params.yaml)
  house.json  room types, room rectangles and viewpoints in world meters, and the
              overhead camera's position and field of view (to place labels)

The robot (Point 2 and the robot paragraph of the engineering parameters):
two driven wheels plus two casters, footprint radius 0.20 m, a 360-degree
lidar for Nav2, and an RGB camera for the demo only (feeds no decision).
Its pose comes from Gazebo's OdometryPublisher system, which reports the
model's true pose, so the virtual camera uses ground-truth pose (Agreements)
and the robot can be moved to each episode's start room between episodes.

The world coordinates are the grid's world coordinates (meters), with the
map frame equal to the Gazebo world frame.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from trustmap.generator import ROBOT_RADIUS_M, generate_house

REPO = Path(__file__).resolve().parent.parent
WORLDS_DIR = REPO / "gazebo" / "worlds"
WORLD_NAME = "house"
ROBOT_NAME = "robot"
WALL_HEIGHT_M = 1.0
FLOOR_COLORS = {  # RGB, rooms told apart by floor colour (Q9.7)
    "kitchen": (0.95, 0.85, 0.55), "living": (0.70, 0.85, 0.65), "bathroom": (0.60, 0.80, 0.95),
    "bedroom": (0.90, 0.70, 0.75), "office": (0.80, 0.75, 0.95), "dining": (0.95, 0.70, 0.50),
}

# Robot dimensions (wheel base, wheels and sensors only matter in Gazebo, not in training).
WHEEL_RADIUS = 0.05
WHEEL_SEPARATION = 0.30
BASE_HEIGHT = 0.12

# Overhead demo camera (demo only): a static camera straight above the house centre,
# looking down, with image up = world +y and image right = world +x.
OVERHEAD_HEIGHT_M = 25.0
OVERHEAD_PIXELS = 720
OVERHEAD_RATE_HZ = 4
OVERHEAD_MARGIN_M = 0.6


def overhead_camera(layout):
    """Position (x, y, z) and horizontal field of view (radians) of the overhead camera."""
    grid = layout.grid
    rows, cols = grid.occupied.shape
    (ox, oy), cs = grid.origin, grid.cell_size
    cx, cy = ox + cols * cs / 2, oy + rows * cs / 2
    half = max(cols, rows) * cs / 2 + OVERHEAD_MARGIN_M
    fov = 2 * float(np.arctan(half / (OVERHEAD_HEIGHT_M - WALL_HEIGHT_M)))  # wall tops stay inside the view
    return (cx, cy, OVERHEAD_HEIGHT_M), fov


def overhead_sdf(layout):
    """Static overhead camera model. Pitch +90 degrees points the camera's x axis down; yaw
    +90 degrees turns the image so that up is world +y."""
    (x, y, z), fov = overhead_camera(layout)
    n = OVERHEAD_PIXELS
    return f"""
  <model name='overhead_camera'><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 1.5708 1.5708</pose>
    <link name='l'>
      <sensor name='overhead' type='camera'>
        <topic>overhead/image</topic><update_rate>{OVERHEAD_RATE_HZ}</update_rate><always_on>1</always_on>
        <camera><horizontal_fov>{fov:.5f}</horizontal_fov><image><width>{n}</width><height>{n}</height></image>
          <clip><near>1.0</near><far>{z + 5:.1f}</far></clip></camera>
      </sensor>
    </link>
  </model>"""


def wall_boxes(grid):
    """Blocked cells merged into rectangles: list of (row0, col0, rows, cols).

    Each run of blocked cells in a row is grown downwards while the next rows
    have exactly the same run, which turns the grid into a few dozen boxes.
    """
    occ = grid.occupied.copy()
    boxes = []
    rows, cols = occ.shape
    for r in range(rows):
        c = 0
        while c < cols:
            if not occ[r, c]:
                c += 1
                continue
            c_end = c
            while c_end < cols and occ[r, c_end]:
                c_end += 1
            r_end = r + 1
            while r_end < rows and occ[r_end, c:c_end].all() and \
                    (c == 0 or not occ[r_end, c - 1]) and (c_end == cols or not occ[r_end, c_end]):
                r_end += 1
            occ[r:r_end, c:c_end] = False
            boxes.append((r, c, r_end - r, c_end - c))
            c = c_end
    return boxes


def _box_link(name, x, y, z, sx, sy, sz, rgb, collision=True):
    color = " ".join(f"{v:.3f}" for v in rgb)
    geom = f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"
    col = f"<collision name='c'>{geom}</collision>" if collision else ""
    return (f"<link name='{name}'><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>{col}"
            f"<visual name='v'>{geom}<material><ambient>{color} 1</ambient><diffuse>{color} 1</diffuse>"
            f"</material></visual></link>")


def robot_sdf(x=0.0, y=0.0, yaw=0.0):
    """The robot model as an SDF <model> element."""
    r, h = ROBOT_RADIUS_M, BASE_HEIGHT
    z_base = WHEEL_RADIUS + 0.02
    wheels = ""
    for side, sign in (("left", 1), ("right", -1)):
        wheels += f"""
    <link name='{side}_wheel'>
      <pose>0 {sign * WHEEL_SEPARATION / 2:.3f} {WHEEL_RADIUS} -1.5708 0 0</pose>
      <inertial><mass>0.2</mass><inertia><ixx>1.3e-4</ixx><iyy>1.3e-4</iyy><izz>2.5e-4</izz></inertia></inertial>
      <collision name='c'><geometry><cylinder><radius>{WHEEL_RADIUS}</radius><length>0.03</length></cylinder></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
      <visual name='v'><geometry><cylinder><radius>{WHEEL_RADIUS}</radius><length>0.03</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
    </link>
    <joint name='{side}_wheel_joint' type='revolute'><parent>base_link</parent><child>{side}_wheel</child>
      <axis><xyz expressed_in='__model__'>0 1 0</xyz></axis></joint>"""
    casters = ""
    for name, cx in (("front_caster", 0.14), ("rear_caster", -0.14)):
        casters += f"""
      <collision name='{name}'><pose>{cx} 0 {-z_base + 0.02:.3f} 0 0 0</pose><geometry><sphere><radius>0.02</radius></sphere></geometry>
        <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>"""
    return f"""
  <model name='{ROBOT_NAME}'>
    <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose>
    <link name='base_link'>
      <pose>0 0 {z_base:.3f} 0 0 0</pose>
      <inertial><mass>2.0</mass><inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.04</izz></inertia></inertial>
      <collision name='body'><pose>0 0 {h / 2:.3f} 0 0 0</pose><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
      <visual name='body_visual'><pose>0 0 {h / 2:.3f} 0 0 0</pose><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
        <material><ambient>0.2 0.3 0.8 1</ambient><diffuse>0.2 0.3 0.8 1</diffuse></material></visual>{casters}
      <sensor name='lidar' type='gpu_lidar'>
        <pose>0 0 {h + 0.05:.3f} 0 0 0</pose>
        <topic>scan</topic><ignition_frame_id>lidar_link</ignition_frame_id>
        <update_rate>10</update_rate><always_on>1</always_on>
        <lidar><scan><horizontal><samples>360</samples><resolution>1</resolution>
          <min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
          <range><min>0.25</min><max>8.0</max><resolution>0.01</resolution></range></lidar>
      </sensor>
      <sensor name='demo_camera' type='camera'>
        <pose>{r - 0.02:.3f} 0 {h + 0.1:.3f} 0 0 0</pose>
        <topic>camera/image</topic><ignition_frame_id>camera_link</ignition_frame_id>
        <update_rate>15</update_rate><always_on>1</always_on>
        <camera><horizontal_fov>1.2</horizontal_fov><image><width>640</width><height>480</height></image>
          <clip><near>0.05</near><far>20</far></clip></camera>
      </sensor>
    </link>{wheels}
    <plugin filename='ignition-gazebo-diff-drive-system' name='ignition::gazebo::systems::DiffDrive'>
      <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
      <wheel_separation>{WHEEL_SEPARATION}</wheel_separation><wheel_radius>{WHEEL_RADIUS}</wheel_radius>
      <topic>cmd_vel</topic><odom_topic>wheel_odom</odom_topic><tf_topic>wheel_tf</tf_topic>
    </plugin>
    <plugin filename='ignition-gazebo-odometry-publisher-system' name='ignition::gazebo::systems::OdometryPublisher'>
      <odom_frame>odom</odom_frame><robot_base_frame>base_link</robot_base_frame>
      <odom_topic>odom</odom_topic><tf_topic>tf</tf_topic><odom_publish_frequency>30</odom_publish_frequency>
      <dimensions>2</dimensions>
    </plugin>
  </model>"""


def target_sdf(name, x, y):
    """The target object: a small red cylinder, visual only (no collision, Q9.7)."""
    return (f"<sdf version='1.8'><model name='{name}'><static>true</static><pose>{x:.3f} {y:.3f} 0 0 0 0</pose>"
            f"<link name='l'><visual name='v'><pose>0 0 0.15 0 0 0</pose><geometry><cylinder><radius>0.06</radius>"
            f"<length>0.3</length></cylinder></geometry><material><ambient>0.9 0.1 0.1 1</ambient>"
            f"<diffuse>0.9 0.1 0.1 1</diffuse></material></visual></link></model></sdf>")


def world_sdf(layout):
    grid = layout.grid
    cs, (ox, oy) = grid.cell_size, grid.origin
    walls = []
    for i, (r, c, nr, nc) in enumerate(wall_boxes(grid)):
        x = ox + (c + nc / 2) * cs
        y = oy + (r + nr / 2) * cs
        walls.append(_box_link(f"wall_{i}", x, y, WALL_HEIGHT_M / 2, nc * cs, nr * cs, WALL_HEIGHT_M,
                               (0.85, 0.85, 0.85)))
    floors = []
    for k, room in enumerate(layout.rooms):
        x0, y0, x1, y1 = (v * cs for v in room.rect)
        floors.append(_box_link(f"floor_{k}_{room.type}", (x0 + x1) / 2, (y0 + y1) / 2, 0.001,
                                x1 - x0, y1 - y0, 0.002, FLOOR_COLORS[room.type], collision=False))
    sx, sy = layout.viewpoint_xy(0)
    return f"""<?xml version='1.0'?>
<!-- Generated by trustmap.gazebo_export from house seed {layout.seed}. Do not edit by hand. -->
<sdf version='1.8'>
<world name='{WORLD_NAME}'>
  <physics name='1ms' type='ignored'><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
  <plugin filename='ignition-gazebo-physics-system' name='ignition::gazebo::systems::Physics'/>
  <plugin filename='ignition-gazebo-user-commands-system' name='ignition::gazebo::systems::UserCommands'/>
  <plugin filename='ignition-gazebo-scene-broadcaster-system' name='ignition::gazebo::systems::SceneBroadcaster'/>
  <plugin filename='ignition-gazebo-sensors-system' name='ignition::gazebo::systems::Sensors'><render_engine>ogre2</render_engine></plugin>
  <light type='directional' name='sun'><cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -1</direction></light>
  <model name='ground'><static>true</static><link name='l'>
    <collision name='c'><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
    <visual name='v'><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
      <material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material></visual></link></model>
  <model name='walls'><static>true</static>{''.join(walls)}</model>
  <model name='floors'><static>true</static>{''.join(floors)}</model>
{robot_sdf(sx, sy)}
{overhead_sdf(layout)}
</world>
</sdf>
"""


def write_map(layout, out_dir):
    """Nav2 map: the raw grid as PGM (0 = wall, 254 = free) plus its YAML description."""
    grid = layout.grid
    img = np.where(grid.occupied, 0, 254).astype(np.uint8)[::-1]  # PGM rows go top to bottom
    rows, cols = img.shape
    with open(out_dir / "map.pgm", "wb") as f:
        f.write(f"P5\n{cols} {rows}\n255\n".encode())
        f.write(img.tobytes())
    (out_dir / "map.yaml").write_text(
        f"image: map.pgm\nmode: trinary\nresolution: {grid.cell_size}\n"
        f"origin: [{grid.origin[0]}, {grid.origin[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")


def export_house(seed, out_root=WORLDS_DIR):
    layout = generate_house(seed)
    out_dir = Path(out_root) / f"house_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "world.sdf").write_text(world_sdf(layout))
    write_map(layout, out_dir)
    cs = layout.grid.cell_size
    (out_dir / "house.json").write_text(json.dumps({
        "seed": seed, "world_name": WORLD_NAME, "robot_name": ROBOT_NAME,
        "room_types": list(layout.house.room_types),
        "rooms_m": [[v * cs for v in r.rect] for r in layout.rooms],
        "viewpoints_m": [list(layout.viewpoint_xy(k)) for k in range(len(layout.rooms))],
        "max_distance_m": layout.house.max_distance,
        "overhead_camera": {"xyz_m": list(overhead_camera(layout)[0]), "horizontal_fov": overhead_camera(layout)[1],
                            "pixels": OVERHEAD_PIXELS, "rate_hz": OVERHEAD_RATE_HZ},
    }, indent=2))
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seeds", type=int, nargs="+")
    p.add_argument("--out", default=str(WORLDS_DIR))
    a = p.parse_args()
    for s in a.seeds:
        print(f"wrote {export_house(s, a.out)}")


if __name__ == "__main__":
    main()
