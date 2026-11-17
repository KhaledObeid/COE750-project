"""Per-episode metrics locked in Point 6b (decision regret is still open, Q6).

SPL is Success weighted by Path Length (Anderson et al., 2018):
    SPL = success * shortest / max(taken, shortest)
Here "shortest" is the planned distance from the start room straight to the
object's room, and "taken" is the planned meters the policy spent. A failed
episode scores 0. Everything is reported in meters (6a).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    spl: float
    success_first_decision: bool  # found on the very first search (6b extra metric)
    decisions: int
    distance_m: float


def episode_metrics(search):
    """Metrics of a finished `RoomSearch`."""
    if not search.done:
        raise ValueError("episode not finished")
    ep = search.episode
    shortest = float(search.house.distances[ep.start_room, ep.object_room])
    spl = shortest / max(search.distance_m, shortest) if search.success else 0.0
    return EpisodeMetrics(
        success=search.success,
        spl=spl,
        success_first_decision=search.success and search.decisions == 1,
        decisions=search.decisions,
        distance_m=search.distance_m,
    )
