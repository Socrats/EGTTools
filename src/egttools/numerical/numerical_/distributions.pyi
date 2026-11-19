"""
The `egttools.numerical.distributions` submodule contains functions and classes for probability distributions and timing uncertainty.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__ = ['TimingUncertainty', 'binom', 'comb', 'multinomial_pmf', 'multivariate_hypergeometric_pdf']
class TimingUncertainty:
    """
    
                    Timing uncertainty distribution container.
    
                    This class provides methods to sample the final round of a game.
                    By default, the timing uncertainty follows a geometric distribution.
    
                    Parameters
                    ----------
                    p : float
                        Probability that the game ends after the minimum number of rounds.
                    max_rounds : int, optional
                        Maximum number of rounds allowed. If 0, no maximum is enforced.
    
                    Examples
                    --------
                    >>> from egttools.numerical.distributions import TimingUncertainty
                    >>> tu = TimingUncertainty(0.2, 20)
                
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, p: typing.SupportsFloat | typing.SupportsIndex, max_rounds: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def calculate_end(self, min_rounds: typing.SupportsInt | typing.SupportsIndex, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> int:
        """
                            Sample the final round, truncated by `max_rounds`.
        
                            The returned round lies in the interval
                            `[min_rounds, max_rounds]` when `max_rounds > 0`.
        
                            Parameters
                            ----------
                            min_rounds : int
                                Minimum number of rounds.
                            seed : int, optional
                                Seed for this draw. If ``None`` (default), a seed is taken from the library's
                                global seed generator, so the results are reproducible after
                                :func:`egttools.Random.init_with_seed`.
        
                            Returns
                            -------
                            int
                                Sampled final round.
        """
    def calculate_full_end(self, min_rounds: typing.SupportsInt | typing.SupportsIndex, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> int:
        """
                            Sample the final round without truncation.
        
                            The returned round is at least `min_rounds`.
        
                            Parameters
                            ----------
                            min_rounds : int
                                Minimum number of rounds.
                            seed : int, optional
                                Seed for this draw. If ``None`` (default), a seed is taken from the library's
                                global seed generator, so the results are reproducible after
                                :func:`egttools.Random.init_with_seed`.
        
                            Returns
                            -------
                            int
                                Sampled final round.
        """
    @property
    def max_rounds(self) -> int:
        """
        Maximum allowed number of rounds. A value of 0 means no maximum.
        """
    @max_rounds.setter
    def max_rounds(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def p(self) -> float:
        """
        Probability that the game ends after the minimum number of rounds.
        """
def binom(n: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
                Calculate the binomial coefficient C(n, k).
    
                This implementation returns a floating-point approximation and should
                be equivalent to `scipy.special.binom`.
    
                Parameters
                ----------
                n : int
                    Size of the full set.
                k : int
                    Size of the subset.
    
                Returns
                -------
                float
                    Binomial coefficient C(n, k).
    
                See Also
                --------
                egttools.distributions.multivariate_hypergeometric_pdf
                egttools.distributions.comb
    """
def comb(n: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> typing.Any:
    """
                Calculate the binomial coefficient C(n, k).
    
                This implementation returns the exact result using multiprecision integers.
    
                Parameters
                ----------
                n : int
                    Size of the full set.
                k : int
                    Size of the subset.
    
                Returns
                -------
                int
                    Binomial coefficient C(n, k).
    
                See Also
                --------
                egttools.distributions.multivariate_hypergeometric_pdf
                egttools.distributions.binom
    """
def multinomial_pmf(x: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"], n: typing.SupportsInt | typing.SupportsIndex, p: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> float:
    """
                Calculate the probability mass function of a multinomial distribution.
    
                This function returns the probability of drawing counts `x` in a sample
                of size `n`, given category probabilities `p`.
    
                Parameters
                ----------
                x : numpy.ndarray
                    Counts for each category in the sample. Must sum to `n`.
                n : int
                    Total number of draws.
                p : numpy.ndarray
                    Category probabilities. Must sum to 1.
    
                Returns
                -------
                float
                    Probability of observing the counts `x`.
    
                See Also
                --------
                egttools.distributions.multivariate_hypergeometric_pdf
                egttools.distributions.binom
                egttools.distributions.comb
    
                Examples
                --------
                >>> import numpy as np
                >>> from egttools.numerical.distributions import multinomial_pmf
                >>> multinomial_pmf(np.array([2, 1]), 3, np.array([0.5, 0.5]))
    """
@typing.overload
def multivariate_hypergeometric_pdf(m: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, sample_counts: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], population_counts: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> float:
    """
                Calculate the probability mass function of a multivariate hypergeometric distribution.
    
                This function returns the probability of observing `sample_counts` when drawing
                a sample of size `n` from a population of size `m`.
    
                Parameters
                ----------
                m : int
                    Population size.
                k : int
                    Number of categories in the population.
                n : int
                    Sample size.
                sample_counts : list[int]
                    Counts for each category in the sample. Must sum to `n`.
                population_counts : numpy.ndarray
                    Counts for each category in the population. Must sum to `m`.
    
                Returns
                -------
                float
                    Probability of observing `sample_counts`.
    
                See Also
                --------
                egttools.distributions.binom
                egttools.distributions.comb
    """
@typing.overload
def multivariate_hypergeometric_pdf(m: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, sample_counts: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"], population_counts: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> float:
    """
                Calculate the probability mass function of a multivariate hypergeometric distribution.
    
                This function returns the probability of observing `sample_counts` when drawing
                a sample of size `n` from a population of size `m`.
    
                Parameters
                ----------
                m : int
                    Population size.
                k : int
                    Number of categories in the population.
                n : int
                    Sample size.
                sample_counts : numpy.ndarray
                    Counts for each category in the sample. Must sum to `n`.
                population_counts : numpy.ndarray
                    Counts for each category in the population. Must sum to `m`.
    
                Returns
                -------
                float
                    Probability of observing `sample_counts`.
    
                See Also
                --------
                egttools.distributions.binom
                egttools.distributions.comb
    """
__init__: str = 'The `egttools.numerical.distributions` submodule contains functions and classes that produce stochastic distributions.'
