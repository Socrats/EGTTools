Tutorials
=========

The `egttools` library is structured in a way that allows the
user to apply all `analytical` and `numerical` methods
implemented in the library to any `Game`, as long as it follows a common interface.
It also provides several plotting functions and classes to help the user visualize the result of the
models. The following figure shows, at a high level, what is the intended structure of `egttools`.

.. image:: images/schema_egttools.pdf
   :alt: Structure of egttools
   :align: center

In the following links you may find more information and examples about how to
use the analytical and numerical methods implemented in `egttools`; how to implement new `Games`;
and how to implement new strategies or `behaviors` for the existing games:

.. toctree::

    Create a new game in Python <tutorials/create_new_python_game>
    Create a new game in C++ and bind it to Python <tutorials/create_cpp_game>
    Create new strategies/behaviors for existing games <tutorials/create_new_behaviors>
    Use analytical methods <tutorials/analytical_methods>
    Use numerical methods <tutorials/numerical_methods>
    Visualize results <tutorials/plotting>
    References <tutorials/references>
