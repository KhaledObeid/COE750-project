"""Per-episode metrics: Point 6b (success, SPL, first-decision success) and 8b (regret).

SPL is Success weighted by Path Length (Anderson et al., 2018):
    SPL = success * shortest / max(taken, shortest)
Here "shortest" is the planned distance from the start room straight to the
object's room, and "taken" is the planned meters the policy spent. A failed
episode scores 0. Everything is reported in meters (6a).

Decision regret (Point 8b) judges each decision by what was knowable when it
was made, not by where the object turned out to be:
    regret of one decision = Q*(s, best room) - Q*(s, room the agent chose)
using the exact optimum with the true r (8a). An episode's regret is the sum
over the decisions actually made, multiplied by the house's largest distance
to turn house-widths into meters. It is never negative, and it is 0 exactly
when every decision was an optimal one.
"""

from dataclasses import dataclass

from trustmap.optimum import solve_episode


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    spl: float
    success_first_decision: bool  # found on the very first search (6b extra metric)
    decisions: int
    distance_m: float
    regret_m: float | None = None  # decision regret in meters (8b); None if not computed


def decision_regret(search, solution):
    """Expected decision regret of a finished or running episode, in meters (8b).

    `solution` must be the optimum of this episode (`solve_episode` with the
    true r). Uses the decisions stored in `search.history`.
    """
    total = 0.0
    for current, searched_bits, chosen in search.history:
        q = solution.q_values(current, searched_bits)
        total += q.max() - q[chosen]
    return total * search.house.max_distance


def episode_metrics(search, solution=None):
    """Metrics of a finished `RoomSearch`.

    Pass the episode's optimum as `solution` to also get decision regret; it
    is optional because solving costs about 5 ms per episode.
    """
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
        regret_m=None if solution is None else decision_regret(search, solution),
    )


def optimum_for(search):
    """Convenience: the 8a optimum (true r) for the episode being played."""
    return solve_episode(search.house, search.episode, search.max_decisions)
