# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
from pathlib import Path

# general configuration
project = "BeamMe"
copyright = "2025, BeamMe Authors"
author = "BeamMe Authors"

# html theme configuration
html_theme = "pydata_sphinx_theme"
html_title = "BeamMe"
html_theme_options = {
    "github_url": "https://github.com/beamme-py/beamme",
}
html_show_sourcelink = False  # Hide "View Source" link on the right side of the page

# extensions
extensions = [
    "myst_parser",  # to enable Markdown support
    "nbsphinx",  # to enable Jupyter Notebook support
    "sphinxcontrib.jquery",  # to enable custom JavaScript (open links in new empty tab)
]

# markdown configuration
myst_enable_extensions = [
    "colon_fence",  # For ::: fenced code blocks
    "linkify",  # Auto-detects URLs and makes them hyperlinks
]
myst_heading_anchors = 3  # automatic heading anchors for Markdown files

# JavaScript configuration (to open links in new tabs)
html_js_files = ["js/custom.js"]
html_static_path = ["static"]

# Always execute notebooks to ensure outputs are up-to-date
nbsphinx_execute = "always"

# Prolog for nbsphinx to add a note with a link to the GitHub source and Binder
# at the top of each rendered html page
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=False)|string %}
{% set binder_path = "/doc/tree/" ~ docname %}
{% set binder_urlpath = binder_path | urlencode %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a href="https://github.com/beamme-py/beamme/blob/main/{{ docname|e }}">
        {{ docname|e }}
      </a>.
      <br>
      Interactive online version:
      <a href="https://mybinder.org/v2/gh/beamme-py/beamme/main?urlpath={{ binder_urlpath }}">
        <img alt="Binder badge"
             src="https://mybinder.org/badge_logo.svg"
             style="vertical-align:text-bottom">
      </a>
    </div>
"""

# Create the directory for static files created by pyvista plots
PYVISTA_DOCS_STATIC = Path(__file__).parent.parent / "build" / "_static" / "pyvista"
PYVISTA_DOCS_STATIC.mkdir(parents=True, exist_ok=True)
os.environ["PYVISTA_DOCS_STATIC"] = str(PYVISTA_DOCS_STATIC)

# Set a flag that we are building the docs with nbsphinx
os.environ["IS_NBSPHINX"] = "1"
