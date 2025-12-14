# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
    'recommonmark',
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
}
autodoc_typehints = 'description'
autodoc_preserve_defaults = True

# -- Markdown support -------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}
