# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Mock imports for RTD ---------------------------------------------------
# Mock complex dependencies that aren't easily available on PyPI
from unittest.mock import MagicMock
import sys

MOCK_MODULES = [
    'pyart',
    'pyart.io',
    'pyart.graph',
    'pyart.graph.radarmap',
    'nexradaws',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ARM Radar Adaptive Scanning'
#copyright = '2025, Project Contributors'
#author = 'Project Contributors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': False,
    'show-source': False,
    'imported-members': False,  # Don't document imported classes from other modules
}
autodoc_typehints = 'description'
autodoc_preserve_defaults = True

# Don't include special members from imported modules
autodoc_mock_imports = []  # List any packages that can't be imported

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'logo': 'logo.png',
    'github_user': 'RBhupi',
    'github_repo': 'adapt',
    'github_button': True,
}
