"""
The `egttools.numerical.distributions` submodule contains functions and classes for probability distributions and timing uncertainty.
"""
from __future__ import annotations
import numpy
import typing
__all__: list[str] = ['TimingUncertainty', 'binom', 'comb', 'multinomial_pmf', 'multivariate_hypergeometric_pdf']
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
    def __init__(*args, **kwargs):
        ...
    @staticmethod
    def calculate_end(*args, **kwargs):
        """
        
                            Sample the final round, truncated by `max_rounds`.
        
                            The returned round lies in the interval
                            `[min_rounds, max_rounds]` when `max_rounds > 0`.
        
                            Parameters
                            ----------
                            min_rounds : int
                                Minimum number of rounds.
                            random_generator : object
                                Random number generator used internally.
        
                            Returns
                            -------
                            int
                                Sampled final round.
                        
        """
    @staticmethod
    def calculate_full_end(*args, **kwargs):
        """
        
                            Sample the final round without truncation.
        
                            The returned round is at least `min_rounds`.
        
                            Parameters
                            ----------
                            min_rounds : int
                                Minimum number of rounds.
                            random_generator : object
                                Random number generator used internally.
        
                            Returns
                            -------
                            int
                                Sampled final round.
                        
        """
    @property
    def max_rounds(*args, **kwargs):
        """
        Maximum allowed number of rounds. A value of 0 means no maximum.
        """
    @max_rounds.setter
    def max_rounds(*args, **kwargs):
        ...
    @property
    def p(*args, **kwargs):
        """
        Probability that the game ends after the minimum number of rounds.
        """
def binom(n: int, k: int) -> float:
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
def comb(n: int, k: int) -> typing.Any:
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
def multinomial_pmf(x: numpy.ndarray[numpy.uint64[m, 1]], n: int, p: numpy.ndarray[numpy.float64[m, 1]]) -> float:
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
def multivariate_hypergeometric_pdf(m: int, k: int, n: int, sample_counts: list[int], population_counts: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
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
def multivariate_hypergeometric_pdf(m: int, k: int, n: int, sample_counts: numpy.ndarray[numpy.uint64[m, 1]], population_counts: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
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
