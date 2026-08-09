# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'comnumpy'
copyright = '2025, V. Choqueuse'
author = 'V. Choqueuse'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # Without napoleon the numpydoc sections of every docstring -- the
    # whole section-4.10 course-material template -- render as raw
    # reStructuredText: no parameter tables, and every D23 attribute
    # name ("sigma2_") read as a broken hyperlink reference.
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid']

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# No intersphinx, deliberately. It would make `np.ndarray` clickable, and
# it would also put the network inside the build -- which runs with -W in
# CI, so an unreachable inventory turns every pull request red for a
# reason that has nothing to do with the change. The gate is worth more
# than the links; add it back with `sphinx.ext.intersphinx` if the
# trade-off ever flips.


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
add_module_names = False
autodoc_member_order = 'bysource'
autosummary_generate = True
autodoc_inherit_docstrings = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "_static/logo.jpg"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.jpg",
        "text": "Comnumpy",
    },
    "github_url": "https://github.com/vincentchoqueuse/comnumpy.git",
}
