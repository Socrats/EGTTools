"""Helpful implementations of stochastic distributions."""

try:
    from ..numerical.numerical_.distributions import (TimingUncertainty,
                                                      multinomial_pmf,
                                                      multivariate_hypergeometric_pdf,
                                                      binom,
                                                      comb)
except Exception:
    raise Exception("numerical package not initialized")

__all__ = ['TimingUncertainty', 'multinomial_pmf', 'multivariate_hypergeometric_pdf', 'binom', 'comb']
