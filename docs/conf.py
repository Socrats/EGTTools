# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#

import os
import sys
import subprocess
import tempfile
import zipfile
import io

# -- Path setup --------------------------------------------------------------

sys.path.insert(1, os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------

project = 'EGTtools'
copyright = '2019-2025, Elias Fernández'
author = 'Elias Fernández'

# -- Version info from package ------------------------------------------------

import egttools

version = '.'.join(egttools.__version__.split('.')[:2])
release = egttools.__version__

# -- Environment flags --------------------------------------------------------

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- General configuration ----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.duration',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'myst_parser',  # Markdown support
    'nbsphinx',
    'pybind11_docstrings',
    'sphinxcontrib.bibtex',
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Enable automatic section labels
autosectionlabel_prefix_document = True

# Source files: reStructuredText and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

language = 'en'

# Syntax highlighting style
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output
todo_include_todos = True

# Autosummary and Autodoc
autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = 'groupwise'
autodoc_inherit_docstrings = True
autoclass_content = 'both'
set_type_checking_flag = True
add_module_names = False
nbsphinx_allow_errors = True

# BibTeX bibliography
bibtex_bibfiles = ['references.bib']
bibtex_encoding = 'latin'
bibtex_default_style = 'unsrt'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}

# HTML options
html_theme = 'default' if on_rtd else 'furo'
html_logo = "images/logo.png"
html_css_files = ["readthedocs-custom.css"]
html_static_path = ['_static']
html_show_sourcelink = False

# Link checking
linkcheck_ignore = [
    'https://doi.org/10.1098/rspb.2008.1126',
    'https://royalsocietypublishing.org/doi/10.1098/rspb.2008.1126',
    'https://doi.org/10.1073/pnas.1015648108',
    'https://pnas.org/doi/full/10.1073/pnas.1015648108',
    'https://doi.org/10.1073/pnas.0709546105',
    'https://pnas.org/doi/full/10.1073/pnas.0709546105',
]

# Nitro picky warnings
nitpicky = True
nitpick_ignore = [
    ('py:class', 'pybind11_builtins.pybind11_object'),
    ('py:class', 'List'),
    ('py:class', 'Positive'),
    ('py:class', 'NonNegative'),
    ('py:class', 'numpy.uint64'),
    ('py:class', 'numpy.int64'),
    ('py:class', 'numpy.float64'),
    ('py:class', 'numpy.complex128'),
    ('py:obj', 'List'),
    ('py:class', 'm'),
    ('py:class', 'n'),
    ('py:class', '1'),
    ('py:exc', 'NetworkXError'),
    ('py:class', 'container'),
    ('py:class', 'node'),
]

# -- ReadTheDocs special wheel installation -----------------------------------

if on_rtd:
    import git
    import github
    import requests

    rtd_version = os.environ.get('READTHEDOCS_VERSION')
    branch = 'docs' if rtd_version == 'latest' else rtd_version

    github_token = os.environ['GITHUB_TOKEN']
    head_sha = git.Repo(search_parent_directories=True).head.commit.hexsha
    g = github.Github()
    runs = g.get_repo('Socrats/EGTTools').get_workflow("wheels.yml").get_runs(branch=branch)
    artifacts_url = next(r for r in runs if r.head_sha == head_sha).artifacts_url

    archive_download_url = next(
        artifact for artifact in requests.get(artifacts_url).json()['artifacts']
        if artifact['name'] == 'rtd-wheel'
    )['archive_download_url']

    artifact_bin = io.BytesIO(
        requests.get(archive_download_url, headers={'Authorization': f'token {github_token}'}, stream=True).content
    )

    with zipfile.ZipFile(artifact_bin) as zf, tempfile.TemporaryDirectory() as tmpdir:
        assert len(zf.namelist()) == 1
        zf.extractall(tmpdir)
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--force-reinstall', tmpdir + '/' + zf.namelist()[0]]
        )
