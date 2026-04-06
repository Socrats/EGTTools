"""
Utilities for random seed generation.
"""
from __future__ import annotations
__all__: list[str] = ['current_seed', 'generate', 'init', 'init_with_seed', 'seed']
def current_seed() -> int:
    """
    Return the current main seed.
    
    Returns
    -------
    int
        Seed currently used by the seed generator.
    
    See Also
    --------
    egttools.Random.init
    egttools.Random.init_with_seed
    egttools.Random.seed
    """
def generate() -> int:
    """
    Generate a new pseudo-random seed.
    
    Returns
    -------
    int
        Pseudo-random integer that can be used as a seed for other generators.
    
    See Also
    --------
    egttools.Random.current_seed
    """
def init() -> None:
    """
    Initialize the random seed generator using system entropy.
    
    This initializes the singleton seed generator with a seed obtained from the
    system random device.
    
    See Also
    --------
    egttools.Random.init_with_seed
    egttools.Random.seed
    egttools.Random.current_seed
    """
def init_with_seed(seed: int) -> None:
    """
    Initialize the random seed generator with a fixed seed.
    
    This allows fully reproducible stochastic simulations.
    
    Parameters
    ----------
    seed : int
        Seed used to initialize the internal random number generator.
    
    See Also
    --------
    egttools.Random.init
    egttools.Random.seed
    egttools.Random.current_seed
    """
def seed(seed: int) -> None:
    """
    Reset the random seed generator with a new seed.
    
    Parameters
    ----------
    seed : int
        New seed value.
    
    See Also
    --------
    egttools.Random.init
    egttools.Random.init_with_seed
    egttools.Random.current_seed
    """
