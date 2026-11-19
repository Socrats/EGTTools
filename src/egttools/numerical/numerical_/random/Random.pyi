"""
Utilities for random seed generation.
"""
from __future__ import annotations
import typing
__all__ = ['current_seed', 'generate', 'get_state', 'init', 'init_with_seed', 'seed', 'set_state']
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
def get_state() -> str:
    """
    Capture the full internal engine state, not just the originating seed.
    
    Unlike ``current_seed()``, which only reports the last explicitly-set seed,
    this captures the engine's current position in its draw sequence — enough
    to resume exactly where it left off via ``set_state()``. Intended for
    checkpointing long-running Monte Carlo estimations so a resumed run does
    not repeat already-consumed random draws.
    
    Returns
    -------
    str
        Opaque string encoding of the current engine state. Treat as a black
        box; only pass it back to ``set_state()``.
    
    See Also
    --------
    egttools.Random.set_state
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
def init_with_seed(seed: typing.SupportsInt | typing.SupportsIndex) -> None:
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
def seed(seed: typing.SupportsInt | typing.SupportsIndex) -> None:
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
def set_state(state: str) -> None:
    """
    Restore an engine state previously captured by ``get_state()``.
    
    Does not change what ``current_seed()`` reports — that still reflects the
    last seed set via ``init_with_seed()``/``seed()``, independent of the live
    engine position restored here.
    
    Parameters
    ----------
    state : str
        A string previously returned by ``get_state()``.
    
    See Also
    --------
    egttools.Random.get_state
    """
