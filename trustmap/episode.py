"""Episode generation recipe (Point 3c, final recipe; uses the Point 3b table).

All randomness of an episode happens here, at reset. After this everything is
deterministic. The steps, in the locked order:

  1) draw a house from the pool          (done by the caller, the environment)
  2) draw a target uniformly
  3) draw the map's true reliability r ~ Beta(10(1 - rho), 10 rho), r = 1 if rho = 0,
     and the displayed confidence c = sigmoid(kappa * logit(r))
  4) flip the lambda coin and pick the claimed room (table weights or uniform)
  5) draw the start room uniformly among the four non-claimed rooms
  6) flip the r coin: if the claim is correct the object is in the claimed room,
     otherwise it goes to one of the remaining three rooms by the table weights

Plus the per-episode shuffle of the observation row order (Point 4a).
"""

from dataclasses import dataclass, field

import numpy as np

from trustmap.cooccurrence import PLACEMENTS, TARGETS, normalize, room_weights
from trustmap.house import NUM_ROOMS

BETA_CONCENTRATION = 10.0  # Point 3c: n = alpha + beta = 10
DEFAULT_LAMBDA = 0.5       # Point 3c: claimed-room mixture


@dataclass(frozen=True)
class EpisodeConfig:
    """The dials of one training run or test condition.

    rho:       corruption level, the map's average error rate (Point 1).
    kappa:     calibration factor; 1 calibrated, > 1 overconfident, < 1 underconfident.
    lam:       probability that the claimed room is drawn uniformly instead of from the table.
    placement: "table" or "uniform" (uniform replaces the whole table by ones).
    targets:   target categories drawn from (all eight by default; one for debugging).
    """

    rho: float
    kappa: float
    lam: float = DEFAULT_LAMBDA
    placement: str = "table"
    targets: tuple = field(default=TARGETS)

    def __post_init__(self):
        # rho = 1 would give alpha = 0, which is not a valid Beta distribution.
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        if not self.targets or any(t not in TARGETS for t in self.targets):
            raise ValueError(f"targets must be a non-empty subset of {TARGETS}")


@dataclass(frozen=True)
class Episode:
    """Everything drawn at reset. Room numbers are house room indices (0-4)."""

    target: str
    r: float                  # true reliability of the map in this episode (hidden from the policy)
    c: float                  # displayed confidence (seen by the policy)
    claim_drawn_uniform: bool # result of the lambda coin
    claimed_room: int
    start_room: int
    claim_correct: bool       # result of the r coin (hidden from the policy)
    object_room: int          # where the target really is (hidden from the policy)
    habit: np.ndarray         # table weights of the target normalized over this house (Point 4a feature 5)
    row_order: np.ndarray     # row_order[i] = house room shown in observation row i (Point 4a shuffle)


def sample_reliability(rho, rng):
    """Step 3a: r ~ Beta(10(1 - rho), 10 rho), whose mean is 1 - rho.

    A Beta distribution gives numbers between 0 and 1; alpha and beta act like
    counts of past successes and failures. rho = 0 means a perfect map, r = 1.
    """
    if rho == 0.0:
        return 1.0
    alpha = BETA_CONCENTRATION * (1.0 - rho)
    beta = BETA_CONCENTRATION * rho
    return float(rng.beta(alpha, beta))


def displayed_confidence(r, kappa):
    """Step 3b: c = sigmoid(kappa * logit(r)).

    logit(r) = log(r / (1 - r)) and sigmoid(x) = 1 / (1 + exp(-x)). Combining
    them gives the same value as r**kappa / (r**kappa + (1 - r)**kappa), which
    is used here because it also works at r = 1 (gives c = 1) without dividing
    by zero.
    """
    a = r ** kappa
    b = (1.0 - r) ** kappa
    return float(a / (a + b))


def generate_episode(house, config, rng):
    """Draw one episode in `house` following the locked recipe (steps 2-6)."""
    rooms = np.arange(NUM_ROOMS)

    # Step 2: target, uniformly among the configured targets.
    target = config.targets[rng.integers(len(config.targets))]
    weights = room_weights(target, house.room_types, config.placement)

    # Step 3: true reliability and displayed confidence.
    r = sample_reliability(config.rho, rng)
    c = 1.0 if r == 1.0 else displayed_confidence(r, config.kappa)

    # Step 4: lambda coin. With probability lam the claim is uniform over the
    # 5 rooms, otherwise it follows the target's table weights.
    claim_drawn_uniform = bool(rng.random() < config.lam)
    claim_probs = np.full(NUM_ROOMS, 1.0 / NUM_ROOMS) if claim_drawn_uniform else normalize(weights)
    claimed_room = int(rng.choice(rooms, p=claim_probs))

    # Step 5: start room, uniformly among the four rooms the map does not claim.
    # It is drawn before the object, so it carries no hint about the object.
    start_candidates = rooms[rooms != claimed_room]
    start_room = int(rng.choice(start_candidates))

    # Step 6: r coin. P(claim correct) = r exactly.
    claim_correct = bool(rng.random() < r)
    if claim_correct:
        object_room = claimed_room
    else:
        remaining = rooms[(rooms != claimed_room) & (rooms != start_room)]
        object_room = int(rng.choice(remaining, p=normalize(weights[remaining])))

    # Point 4a: row order shuffled every episode so a row position never
    # stands for a fixed room.
    row_order = rng.permutation(NUM_ROOMS)

    return Episode(
        target=target,
        r=r,
        c=c,
        claim_drawn_uniform=claim_drawn_uniform,
        claimed_room=claimed_room,
        start_room=start_room,
        claim_correct=claim_correct,
        object_room=object_room,
        habit=normalize(weights),
        row_order=row_order,
    )
