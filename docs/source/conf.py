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
from datetime import datetime
import mock
from sphinx.builders.html import StandaloneHTMLBuilder
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'DeepCell Spots'
copyright = ('2019-{currentyear}, Van Valen Lab at the '
             'California Institute of Technology (Caltech)').format(
                 currentyear=datetime.now().year)
author = 'Van Valen Lab at Caltech'

# The full version, including alpha/beta/rc tags
release = '0.3.1'

# -- RTD configuration ------------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# This is used for linking and such so we link to the thing we're building
rtd_version = os.environ.get("READTHEDOCS_VERSION", "master")
if rtd_version not in ["stable", "latest", "master"]:
    rtd_version = "stable"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '2.3.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'm2r',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel'
]

napoleon_google_docstring = True

default_role = 'py:obj'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# Ignore warning:
suppress_warnings = ['autosectionlabel.*']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


# -- Extension configuration -------------------------------------------------
autodoc_mock_imports = [
    'pandas',
    'numpy',
    'scipy',
    'scikit-image',
    'scikit-learn',
    'tensorflow',
    'jupyter',
    'networkx',
    'opencv-python-headless',
    'deepcell',
    'tqdm',
    'trackpy',
    'torch',
    'torchvision',
    'pyro-ppl'
]

# Disable nbsphinx extension from running notebooks
nbsphinx_execute = 'never'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# TODO: fix relative URL for notebooks, using replace() is not perfect.
nbsphinx_prolog = (
r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'].replace('../', '') %}
{% else %}
{% set docpath = env.doc2path(env.docname, base='docs/source') %}
{% endif %}
.. raw:: html
    <div class="admonition note">
        <p>This page was generated from <a href="https://github.com/vanvalenlab/deepcell-spots/blob/""" + git_rev + r"""{{ docpath }}">{{ docpath }}</a>
        </p>
    </div>
"""
)

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'kiosk': ('https://deepcell-kiosk.readthedocs.io/en/{}/'.format(rtd_version), None),
    'kiosk-redis-consumer': (('https://deepcell-kiosk.readthedocs.io/'
                              'projects/kiosk-redis-consumer/en/{}/').format(rtd_version), None),
}

intersphinx_cache_limit = 0

# -- Custom Additions --------------------------------------------------------
nitpick_ignore = []
# See the following page for more information and syntax:
#  www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore

for line in open('.nitpick-ignore'):
    line = line.strip()
    if not line or line.startswith('#'):
        continue

    reftype, target = line.split(' ', 1)
    nitpick_ignore.append((reftype, target.strip()))

StandaloneHTMLBuilder.supported_image_types = [
    'image/svg+xml',
    'image/gif',
    'image/png',
    'image/jpeg'
]
