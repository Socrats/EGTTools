![EGTtools](docs/images/logo-full.png)

# Toolbox for Evolutionary Game Theory

[![DOI](https://zenodo.org/badge/242180332.svg)](https://zenodo.org/badge/latestdoi/242180332)
[![Documentation Status](https://readthedocs.org/projects/egttools/badge/?version=latest)](https://egttools.readthedocs.io/en/latest/?badge=latest)

**EGTtools** provides a centralized repository with analytical 
and numerical methods to study/model game theoretical problems under the Evolutionary
Game Theory (EGT) framework.

This library is implemented both in Python and C++ (with Python bindings) in order to
provide optimized computational methods that can run in parallel in a reasonable time.

To install all required packages run:

```bash
python -m venv egtenv
source egtenv/bin/activate
pip install -r requirements.txt
```

Or with anaconda:

```bash
conda create --name egtenv
conda activate egtenv
pip install -r requirements.txt
```

Finally, to make your virtual environment visible to jupyter:

```bash
python -m ipykernel install --user --name=egtenv
```

The [EGTtools](egttools/analytical/sed_analytical.py) module contains classes and functions that you may use to investigate the evolutionary dynamics in 2-player games.

The [Example](docs/examples/hawk_dove_dynamics.ipynb) is a jupyter notebook the analysis of the evolutionary dynamics in a Hawk-Dove game.

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