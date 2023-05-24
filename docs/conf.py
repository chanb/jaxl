# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "JAX Learning"
copyright = "2023, Bryan Chan"
author = "Bryan Chan"

version = "0.0.1"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # Supports Google / Numpy docstring
    "sphinx.ext.autodoc",  # Documentation from docstrings
    "sphinx.ext.doctest",  # Test snippets in documentation
    "sphinx.ext.todo",  # to-do syntax highlighting
    "sphinx.ext.ifconfig",  # Content based configuration
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
source_suffix = [".rst"]
