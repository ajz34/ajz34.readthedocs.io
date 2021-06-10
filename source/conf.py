# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'ajz34'
copyright = '2019-2020, ajz34'
author = 'ajz34'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        "nbsphinx",
        "sphinx.ext.mathjax",
        "sphinxcontrib.bibtex"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
        "tmp",
        "_build",
        "**.ipynb_checkpoints"
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

nbsphinx_allow_errors = True
nbsphinx_timeout = 720
# https://stackoverflow.com/questions/56336234/build-fail-sphinx-error-contents-rst-not-found
master_doc = "index"

def setup(app):
    # https://stackoverflow.com/questions/23211695/modifying-content-width-of-the-sphinx-theme-read-the-docs
    app.add_css_file('style.css')

mathjax_path = "MathJax/es5/tex-chtml-full.js"

bibtex_bibfiles = [
    "QC_Notes/assets/PUHF_and_PMP2.bib",
    "QC_Notes/DF_Series/assets/DF_SCF.bib",
    "QC_Notes/DF_Series/assets/LT_MP2.bib",
    "QC_Notes/Post_Series/mp3_mp4_energy.bib",
    "QC_Notes/Post_Series/dRPA_Comprehense.bib",
    "QC_Notes/Post_Series/scsRPA_Comprehense.bib",
    "QC_Notes/Prop_Series/Mag_NoGIAO_NumDeriv.bib",
    "QC_Notes/Prop_Series/Mag_GIAO_NumDeriv.bib",
    "ML_Notes/Autograd_Series/assets/Autograd_RHF.bib",
    "ML_Notes/SISSO/SISSO_SimpleNote.bib",
]
