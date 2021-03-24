![EGTtools](docs/images/logo-full.png)

# Toolbox for Evolutionary Game Theory

[![DOI](https://zenodo.org/badge/242180332.svg)](https://zenodo.org/badge/latestdoi/242180332)
[![Documentation Status](https://readthedocs.org/projects/egttools/badge/?version=latest)](https://egttools.readthedocs.io/en/latest/?badge=latest)
![build](https://github.com/Socrats/EGTTools/workflows/build/badge.svg)

**EGTtools** provides a centralized repository with analytical and numerical methods to study/model game theoretical
problems under the Evolutionary Game Theory (EGT) framework.

This library is composed of two parts:

- a set of analytical methods implemented in Python 3
- a set of computational methods implemented in C++ with (Python 3 bindings)

The second typed is used in cases where the state space is too big to solve analytically, and thus require estimating
the model parameters through monte-carlo simulations. The C++ implementation provides optimized computational methods
that can run in parallel in a reasonable time, while Python bindings make the methods easily accecible to a larger range
of researchers.

---

## Installation from source

To **install all required packages** run:

```bash
python -m venv egttools-env
source egtenv/bin/activate
pip install -r requirements.txt
```

Or with anaconda:

```bash
conda env create -f environment.yml
conda activate egttools-env
```

Also, to make your virtual environment visible to jupyter:

```bash
conda install ipykernel # or pip install ipykernel
python -m ipykernel install --user --name=egttools-env
```

Finally, you can **install EGTtools** in your virtual environment by running:

```bash
python -m pip install <path>
```

Where ```<path>``` represents the path to the EGTtools folder. If you are running this while inside the EGTtools folder,
then ```<path>``` is simply ```\.```.

If you wish, you may also install EGTtools in **development** mode, this will allow the installation to update with new
modifications to the package:

```bash
python -m pip install -e <path>
```

---

The [EGTtools](egttools/analytical/sed_analytical.py) module contains classes and functions that you may use to
investigate the evolutionary dynamics in 2-player games.

The [Example](docs/examples/hawk_dove_dynamics.ipynb) is a jupyter notebook the analysis of the evolutionary dynamics in
a Hawk-Dove game.

You can find more information in the [documentation](https://egttools.readthedocs.io/en/latest/).

---

### Caveats

1. On Apple M1 (arm64) you should install (for the moment) [miniforge](https://github.com/conda-forge/miniforge), create
   a conda environment using it, and install EGTtools from the conda environment.

2. In MacOSX it is assumed that you have Homebrew installed.
3. You should install libomp with homebrew ``brew install libomp`` if you want to have support for parallel operations (
   there is a big difference in computation time).

4. You **must** have Eigen 3.3.* installed.

5. You **do not** need any of the above if you install EGTtools through ```pip install egttools```. This will soon be an
   option.

---

## Citing

You may cite this repository in the following way:

```latex
@misc{Fernandez2020,
  author = {Fernández Domingos, Elias},
  title = {EGTTools: Toolbox for Evolutionary Game Theory},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Socrats/EGTTools}},
  doi = {10.5281/zenodo.3687125}
}
```

## Licence

* EGTtools is released under the [GNU General Public Licence](LICENSE), version 3 or later.
* [pybind11](https://github.com/pybind/pybind11) is released under [a BSD-style license](pybind11/LICENSE).