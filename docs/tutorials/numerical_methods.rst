Apply numerical methods
=======================


Estimate fixation probabilities
-------------------------------


Estimate stationary distributions
---------------------------------

.. warning::
    This method should not use for states spaces larger than the number which can be stored in
    a 64 bit - `int64_t` - integer!

Estimate strategy distributions
-------------------------------


Run a single simulation
-----------------------


Evolve a population for a given number of rounds
------------------------------------------------

.. note::
    Although at the moment `egttools.numerical` only contain methods to
    study evolutionary dynamics in well-mixed populations, we have planned
    to add support for simulations in complex networks in version 0.14.0,
    and for multi-level selection in version 0.15.0.