.. image:: images/logo-full.png

EGTtools -- Toolbox for Evolutionary Game Theory
================================================

**EGTtools** provides a centralized repository with analytical and numerical methods to study/model game theoretical
problems under the Evolutionary Game Theory (EGT) framework.

This library is composed of two parts:

- a set of analytical methods implemented in Python 3
- a set of computational methods implemented in C++ with (Python 3 bindings)

The second typed is used in cases where the state space is too big to solve analytically, and thus require estimating
the model parameters through monte-carlo simulations. The C++ implementation provides optimized computational methods
that can run in parallel in a reasonable time, while Python bindings make the methods easily accecible to a larger range
of researchers.

.. toctree::
   :hidden:

   Home page <self>
   Installation <installation>
   Tutorials <tutorials>
   Examples <examples>
   API reference <_autosummary/egttools>

Citing EGTtools
------------------


You may cite this repository in the following way:

.. code-block:: bibtex

   @misc{Fernandez2020,
     author = {Fernández Domingos, Elias},
     title = {EGTTools: Toolbox for Evolutionary Game Theory},
     year = {2020},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/Socrats/EGTTools}},
     doi = {10.5281/zenodo.3687125}
   }

Acknowledgements
----------------
I would like to thank Prof. Tom Lenaerts and Prof. Francisco C. Santos for the great help
in building this library.

We would also like to thank Yannick Jadoul author of
Parselmouth and Eugenio Bargiacchi author of AIToolBox
for their great advice on the implementation of EGTtools.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
