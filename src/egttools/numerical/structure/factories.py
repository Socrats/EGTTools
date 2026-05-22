from typing import Dict, List, Optional
from egttools.numerical.structure import (
    Network, NetworkGroup, NetworkSync, NetworkGroupSync,
    NetworkMCEstimatorPC, NetworkMCEstimatorBD, NetworkMCEstimatorDB,
    NetworkMCEstimatorTDPC,
    NetworkCoEvolutionaryPC, NetworkCoEvolutionaryPCHomophilic,
)
from egttools.games import AbstractSpatialGame


def network_factory(nb_strategies: int, beta: float, mu: float, game: AbstractSpatialGame, cache_size: int,
                    node_list: List[Dict[int, List[int]]]) -> List[Network]:
    """
    Generates a list of Network objects from the list of node, neighbours dictionaries.

    Parameters
    ----------
    nb_strategies : int
        Number of strategies in the population
    beta : float
        Intensity of selection
    mu : float
        Mutation rate
    game : egttools.games.AbstractSpatialGame
        A game to associate with each network
    cache_size : int
        The size of the cache memory to use
    node_list : List[Dict[int, List[int]]
        A list of dictionaries containing the nodes and their neighbours

    Returns
    -------
    List[Network]
        A list of Network objects

    """
    network_list = []
    for i, node_dictionary in enumerate(node_list):
        network_list.append(Network(nb_strategies, beta, mu, node_dictionary, game, cache_size))

    return network_list


def network_group_factory(nb_strategies: int, beta: float, mu: float, game: AbstractSpatialGame, cache_size: int,
                          node_list: List[Dict[int, List[int]]]) -> List[Network]:
    """
    Generates a list of NetworkGroup objects from the list of node, neighbours dictionaries.

    Parameters
    ----------
    nb_strategies : int
        Number of strategies in the population
    beta : float
        Intensity of selection
    mu : float
        Mutation rate
    game : egttools.games.AbstractSpatialGame
        A game to associate with each network
    cache_size : int
        The size of the cache memory to use
    node_list : List[Dict[int, List[int]]
        A list of dictionaries containing the nodes and their neighbours

    Returns
    -------
    List[Network]
        A list of Network objects

    """
    network_list = []
    for i, node_dictionary in enumerate(node_list):
        network_list.append(NetworkGroup(nb_strategies, beta, mu, node_dictionary, game, cache_size))

    return network_list


def network_sync_factory(nb_strategies: int, beta: float, mu: float, game: AbstractSpatialGame, cache_size: int,
                         node_list: List[Dict[int, List[int]]]) -> List[Network]:
    """
    Generates a list of Network objects from the list of node, neighbours dictionaries.

    Parameters
    ----------
    nb_strategies : int
        Number of strategies in the population
    beta : float
        Intensity of selection
    mu : float
        Mutation rate
    game : egttools.games.AbstractSpatialGame
        A game to associate with each network
    cache_size : int
        The size of the cache memory to use
    node_list : List[Dict[int, List[int]]
        A list of dictionaries containing the nodes and their neighbours

    Returns
    -------
    List[Network]
        A list of Network objects

    """
    network_list = []
    for i, node_dictionary in enumerate(node_list):
        network_list.append(NetworkSync(nb_strategies, beta, mu, node_dictionary, game, cache_size))

    return network_list


def network_group_sync_factory(nb_strategies: int, beta: float, mu: float, game: AbstractSpatialGame,
                               cache_size: int,
                               node_list: List[Dict[int, List[int]]]) -> List[Network]:
    """
    Generates a list of NetworkGroup objects from the list of node, neighbours dictionaries.

    Parameters
    ----------
    nb_strategies : int
        Number of strategies in the population
    beta : float
        Intensity of selection
    mu : float
        Mutation rate
    game : egttools.games.AbstractSpatialGame
        A game to associate with each network
    cache_size : int
        The size of the cache memory to use
    node_list : List[Dict[int, List[int]]
        A list of dictionaries containing the nodes and their neighbours

    Returns
    -------
    List[Network]
        A list of Network objects

    """
    network_list = []
    for i, node_dictionary in enumerate(node_list):
        network_list.append(NetworkGroupSync(nb_strategies, beta, mu, node_dictionary, game, cache_size))

    return network_list


_UPDATE_RULE_ESTIMATORS = {
    "PC": NetworkMCEstimatorPC,
    "PairwiseComparison": NetworkMCEstimatorPC,
    "BD": NetworkMCEstimatorBD,
    "BirthDeath": NetworkMCEstimatorBD,
    "DB": NetworkMCEstimatorDB,
    "DeathBirth": NetworkMCEstimatorDB,
    "TDPC": NetworkMCEstimatorTDPC,
    "TimeDependentPC": NetworkMCEstimatorTDPC,
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
    Create a :class:`NetworkMCEstimator` for the requested update rule.

    Parameters
    ----------
    game : egttools.games.AbstractSpatialGame
    topology : dict[int, list[int]]
        Network adjacency dictionary (e.g. from ``dict(G.adjacency())``).
    nb_strategies : int
    beta : float
        Selection intensity.
    mu : float
        Mutation probability.
    update_rule : str
        One of ``"PC"`` / ``"PairwiseComparison"``,
        ``"BD"`` / ``"BirthDeath"``,
        ``"DB"`` / ``"DeathBirth"``,
        ``"TDPC"`` / ``"TimeDependentPC"``.
    cache_size : int, optional

    Returns
    -------
    NetworkMCEstimatorPC | NetworkMCEstimatorBD | NetworkMCEstimatorDB | NetworkMCEstimatorTDPC
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
    Create a :class:`NetworkCoEvolutionary` estimator for the requested rewiring rule.

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
