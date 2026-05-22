from typing import Dict, List, Optional
from egttools.numerical.structure import (
    NetworkMCEstimatorPC, NetworkMCEstimatorBD, NetworkMCEstimatorDB,
    NetworkMCEstimatorTDPC, NetworkMCEstimatorLP,
    NetworkCoEvolutionaryPC, NetworkCoEvolutionaryPCHomophilic,
)
from egttools.games import AbstractSpatialGame


_UPDATE_RULE_ESTIMATORS = {
    "PC": NetworkMCEstimatorPC,
    "PairwiseComparison": NetworkMCEstimatorPC,
    "BD": NetworkMCEstimatorBD,
    "BirthDeath": NetworkMCEstimatorBD,
    "DB": NetworkMCEstimatorDB,
    "DeathBirth": NetworkMCEstimatorDB,
    "TDPC": NetworkMCEstimatorTDPC,
    "TimeDependentPC": NetworkMCEstimatorTDPC,
    "LP": NetworkMCEstimatorLP,
    "LinearProportional": NetworkMCEstimatorLP,
}


def network_mc_estimator_factory(
    game: AbstractSpatialGame,
    topology: Dict[int, List[int]],
    nb_strategies: int,
    beta: float,
    mu: float,
    update_rule: str = "PC",
    cache_size: int = 100000,
):
    """
    Create a NetworkMCEstimator for the requested update rule.

    Parameters
    ----------
    game : egttools.games.AbstractSpatialGame
    topology : dict[int, list[int]]
        Network adjacency dictionary (e.g. from ``{n: list(nbrs) for n, nbrs in G.adjacency()}``).
    nb_strategies : int
    beta : float
        Selection intensity.  For ``"LP"`` / ``"LinearProportional"`` pass
        ``beta = max(T, 1.0) - min(S, 0.0)`` (payoff-normalisation constant D_>).
    mu : float
        Mutation probability.
    update_rule : str
        One of ``"PC"`` / ``"PairwiseComparison"``,
        ``"BD"`` / ``"BirthDeath"``,
        ``"DB"`` / ``"DeathBirth"``,
        ``"TDPC"`` / ``"TimeDependentPC"``,
        ``"LP"`` / ``"LinearProportional"``.
    cache_size : int, optional

    Returns
    -------
    NetworkMCEstimatorPC | NetworkMCEstimatorBD | NetworkMCEstimatorDB |
    NetworkMCEstimatorTDPC | NetworkMCEstimatorLP
    """
    cls = _UPDATE_RULE_ESTIMATORS.get(update_rule)
    if cls is None:
        raise ValueError(
            f"Unknown update_rule '{update_rule}'. "
            f"Choose from: {list(_UPDATE_RULE_ESTIMATORS.keys())}"
        )
    return cls(game, topology, nb_strategies, beta, mu, cache_size)


def network_coevo_factory(
    game: AbstractSpatialGame,
    topology: Dict[int, List[int]],
    nb_strategies: int,
    beta: float,
    mu: float,
    rewiring_probability: float,
    rewiring_rule: str = "random",
    cache_size: int = 100000,
):
    """
    Create a NetworkCoEvolutionary estimator for the requested rewiring rule.

    Parameters
    ----------
    game : egttools.games.AbstractSpatialGame
    topology : dict[int, list[int]]
    nb_strategies : int
    beta : float
    mu : float
    rewiring_probability : float
        Probability per time step that a rewiring event occurs.
    rewiring_rule : str
        ``"random"`` (Santos 2006) or ``"homophilic"`` (Borges 2023).
    cache_size : int, optional

    Returns
    -------
    NetworkCoEvolutionaryPC | NetworkCoEvolutionaryPCHomophilic
    """
    rewiring_map = {
        "random": NetworkCoEvolutionaryPC,
        "homophilic": NetworkCoEvolutionaryPCHomophilic,
    }
    cls = rewiring_map.get(rewiring_rule.lower())
    if cls is None:
        raise ValueError(
            f"Unknown rewiring_rule '{rewiring_rule}'. "
            f"Choose from: {list(rewiring_map.keys())}"
        )
    return cls(game, topology, nb_strategies, beta, mu, rewiring_probability, cache_size)
