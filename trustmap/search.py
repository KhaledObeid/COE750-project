"""Room-level search rules: actions (Point 5) and reward (Point 6).

One decision = one step: the policy picks one of the 5 observation rows, the
robot goes to that room's viewpoint along the planned path, and the room is
checked for the target. This is the core that the Gymnasium environment
(after Point 7) and the Gazebo runner will both call.

Action masking (Point 5a): the policy may only pick rooms not searched yet.
`action_mask()` returns five yes/no values, one per observation row; the
policy later sets the probability of every "no" row to exactly zero, so it
can never waste a decision on a searched room. Rewards (Point 6):
  - each decision costs minus its planned distance divided by the house's
    largest room-to-room distance (6a),
  - if the last allowed decision misses, the episode fails and -1 is added (6b),
  - no discounting, gamma = 1 (6c), so the return is just the sum.
"""

import numpy as np

from trustmap.house import NUM_ROOMS

FAILURE_PENALTY = 1.0  # Point 6b: P = 1, the longest possible single trip


class RoomSearch:
    """The state of one episode while it is being played.

    The true state is the current room plus the set of searched rooms
    (Points 4a and 5b); everything else is fixed at reset in `episode`.
    """

    def __init__(self, house, episode, max_decisions):
        self.house = house
        self.episode = episode
        self.max_decisions = max_decisions
        self.current_room = episode.start_room
        self.searched = np.zeros(NUM_ROOMS, dtype=bool)
        self.searched[episode.start_room] = True  # Point 3a: start room counts as searched
        self.decisions = 0
        self.distance_m = 0.0  # planned meters so far, for reporting in meters
        self.done = False
        self.success = False

    def action_mask(self):
        """True for rows whose room is still unsearched, in shuffled row order (5a)."""
        return ~self.searched[self.episode.row_order]

    def step(self, row):
        """Search the room shown in observation row `row`; return the reward."""
        if self.done:
            raise RuntimeError("episode is over")
        room = int(self.episode.row_order[row])
        if self.searched[room]:
            raise ValueError("masked action: room already searched")

        meters = float(self.house.distances[self.current_room, room])
        reward = -meters / self.house.max_distance  # Point 6a
        self.distance_m += meters
        self.current_room = room
        self.searched[room] = True  # Point 4b: only the chosen room is searched
        self.decisions += 1

        # Point 5b: success is automatic when the target is in the chosen room.
        # Detection is deterministic, so this is "is the object in this room?".
        # The virtual camera (OPEN_QUESTIONS Q4) must give the same answer.
        if room == self.episode.object_room:
            self.done = True
            self.success = True
        elif self.decisions == self.max_decisions:
            self.done = True  # Point 6b: cap reached without finding the target
            reward -= FAILURE_PENALTY
        return reward
