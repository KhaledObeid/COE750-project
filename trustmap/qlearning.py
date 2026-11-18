"""E0 sanity check: tabular Q-learning reaches the exact optimum (Point 8c).

For one fixed episode configuration (house, target, claim, r, start room),
the room-level problem is a small table: state = (current room, searched set),
action = room to search. Each training episode draws only where the object
is, exactly as the 3c recipe does once the claim and start are fixed (with
probability r in the claimed room, otherwise among the other three rooms by
the table). Q-learning (M4-2) should then converge to the value-iteration
Q-values of optimum.py and pick the same rooms.

Update, with gamma = 1 (6c):
    Q(s, a) <- Q(s, a) + alpha * (reward + max over allowed a' of Q(s', a') - Q(s, a))
where the max is 0 when the episode has ended. Exploration is epsilon-greedy
over unsearched rooms only (the 5a mask). The step size is alpha = 1 / (number
of times (s, a) was updated), the incremental-mean step size from the course;
with it the estimates settle on exact averages instead of jittering.

These learning settings only affect how fast E0 converges, not what it
converges to. Rooms the greedy policy rarely picks are tried only through
epsilon (about 5% of decisions each), so their Q-values carry more sampling
noise than the Q-value of the best room.
"""

import dataclasses

import numpy as np

from trustmap.house import NUM_ROOMS
from trustmap.optimum import ALL_SEARCHED, object_belief
from trustmap.search import RoomSearch


def q_learning(house, episode, max_decisions, num_episodes=20_000, epsilon=0.2, seed=0, on_episode=None):
    """Learn Q for one episode configuration. Returns Q[current room, searched bits, room].

    on_episode(n, Q), if given, is called after every training episode n = 1, 2, ...
    (E0 uses it to record the learning curve).
    """
    rng = np.random.default_rng(seed)
    belief = object_belief(episode, episode.r)
    Q = np.zeros((NUM_ROOMS, ALL_SEARCHED + 1, NUM_ROOMS))
    visits = np.zeros_like(Q)

    for n in range(1, num_episodes + 1):
        # The only randomness per training episode: where the object is.
        ep = dataclasses.replace(episode, object_room=int(rng.choice(NUM_ROOMS, p=belief)))
        search = RoomSearch(house, ep, max_decisions)
        while not search.done:
            s_room, s_bits = search.current_room, search.searched_bits
            allowed = [k for k in range(NUM_ROOMS) if not search.searched[k]]
            if rng.random() < epsilon:
                room = int(rng.choice(allowed))
            else:
                room = max(allowed, key=lambda k: Q[s_room, s_bits, k])
            reward = search.step(search.row_of_room(room))

            if search.done:
                target = reward
            else:
                next_allowed = [k for k in range(NUM_ROOMS) if not search.searched[k]]
                target = reward + max(Q[search.current_room, search.searched_bits, k] for k in next_allowed)
            visits[s_room, s_bits, room] += 1
            alpha = 1.0 / visits[s_room, s_bits, room]
            Q[s_room, s_bits, room] += alpha * (target - Q[s_room, s_bits, room])
        if on_episode is not None:
            on_episode(n, Q)
    return Q
