from typing import Dict, List
from egttools.numerical.structure import Network, NetworkGroup
from egttools.games import AbstractSpatialGame


def network_factory(nb_strategies: int, beta: float, mu: float, games: List[AbstractSpatialGame], cache_size: int,
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
    games : List[egttools.games.AbstractSpatialGame]
        A list of games to associate with each network
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
        network_list.append(Network(nb_strategies, beta, mu, node_dictionary, games[i], cache_size))

    return network_list


def network_group_factory(nb_strategies: int, beta: float, mu: float, games: List[AbstractSpatialGame], cache_size: int,
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
    games : List[egttools.games.AbstractSpatialGame]
        A list of games to associate with each network
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
        network_list.append(NetworkGroup(nb_strategies, beta, mu, node_dictionary, games[i], cache_size))

    return network_list
