"""Virtual camera shared by the 2D simulator and Gazebo (Agreements, Point 5b).

The target counts as detected when it is (1) inside the camera's
field-of-view cone, (2) within the detection range, and (3) in line of
sight: the straight segment from the robot to the target crosses no blocked
cell of the occupancy grid. In Gazebo the robot pose comes from ground truth,
so both simulators call exactly these functions.

The range of about 1.5 m is stated in DECISIONS.md (Agreements). The
field-of-view angle and the robot's heading on arrival are still open
(OPEN_QUESTIONS Q4), so the angle is a required argument; a 360-degree look
is fov_rad = 2*pi.
"""

import math

DETECTION_RANGE_M = 1.5  # DECISIONS.md Agreements: visible within about 1.5 m


def cells_on_segment(grid, p0, p1):
    """Every grid cell the straight segment from p0 to p1 (world meters) passes through.

    Walks the segment cell by cell, always crossing into whichever cell
    border comes next along the line (the Amanatides-Woo grid traversal).
    If the line passes exactly through a cell corner, both cells beside the
    corner are included, so a ray cannot slip between two diagonal wall
    cells (the same rule as "no corner cutting" in planning.py).
    """
    x0, y0 = grid.to_grid_coords(p0)
    x1, y1 = grid.to_grid_coords(p1)
    col, row = math.floor(x0), math.floor(y0)
    end_col, end_row = math.floor(x1), math.floor(y1)
    dx, dy = x1 - x0, y1 - y0
    step_c = 1 if dx > 0 else -1
    step_r = 1 if dy > 0 else -1
    # Fraction of the segment at which it reaches the next vertical / horizontal
    # cell border, and how much that fraction grows per cell crossed.
    if dx != 0:
        t_next_c = ((col + (1 if dx > 0 else 0)) - x0) / dx
        t_step_c = abs(1.0 / dx)
    else:
        t_next_c = t_step_c = math.inf
    if dy != 0:
        t_next_r = ((row + (1 if dy > 0 else 0)) - y0) / dy
        t_step_r = abs(1.0 / dy)
    else:
        t_next_r = t_step_r = math.inf

    cells = [(row, col)]
    cols_left = abs(end_col - col)
    rows_left = abs(end_row - row)
    while cols_left or rows_left:
        through_corner = cols_left and rows_left and math.isclose(t_next_c, t_next_r, abs_tol=1e-12)
        if through_corner:
            cells.append((row, col + step_c))
            cells.append((row + step_r, col))
            col += step_c
            row += step_r
            t_next_c += t_step_c
            t_next_r += t_step_r
            cols_left -= 1
            rows_left -= 1
        elif cols_left and (not rows_left or t_next_c < t_next_r):
            col += step_c
            t_next_c += t_step_c
            cols_left -= 1
        else:
            row += step_r
            t_next_r += t_step_r
            rows_left -= 1
        cells.append((row, col))
    return cells


def line_of_sight(grid, robot_xy, target_xy):
    """True if no blocked cell lies between the robot and the target.

    The target's own cell is not checked, so an object standing on furniture
    (a blocked cell) can still be seen; the robot's own cell is not checked
    either. Every cell in between must be free and inside the grid.
    """
    cells = cells_on_segment(grid, robot_xy, target_xy)
    return all(grid.is_free(cell) for cell in cells[1:-1])


def angle_difference(a, b):
    """Smallest signed difference a - b between two angles, in (-pi, pi]."""
    d = (a - b) % (2.0 * math.pi)
    return d - 2.0 * math.pi if d > math.pi else d


def in_field_of_view(robot_pose, target_xy, fov_rad, max_range_m=DETECTION_RANGE_M):
    """True if the target is inside the view cone: close enough and within half the angle of the heading.

    robot_pose = (x, y, heading in radians, 0 = facing +x).
    """
    x, y, heading = robot_pose
    tx, ty = target_xy
    if math.hypot(tx - x, ty - y) > max_range_m:
        return False
    if fov_rad >= 2.0 * math.pi:
        return True
    bearing = math.atan2(ty - y, tx - x)
    return abs(angle_difference(bearing, heading)) <= fov_rad / 2.0


def target_visible(grid, robot_pose, target_xy, fov_rad, max_range_m=DETECTION_RANGE_M):
    """The virtual camera: field-of-view cone plus line-of-sight ray cast."""
    return in_field_of_view(robot_pose, target_xy, fov_rad, max_range_m) and line_of_sight(
        grid, robot_pose[:2], target_xy
    )
