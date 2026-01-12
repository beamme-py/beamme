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
"""Helper functions for visualizations."""

import os as _os
import uuid as _uuid
from pathlib import Path as _Path

import ipywidgets as _widgets
import pyvista as _pv
from IPython.display import HTML as _HTML
from IPython.display import IFrame as _IFrame
from IPython.display import display as _display

from beamme.utils.environment import is_mybinder as _is_mybinder
from beamme.utils.environment import is_nbsphinx as _is_nbsphinx

# Start virtual framebuffer for MyBinder environments
if _is_mybinder():
    _pv.start_xvfb()


def show_plotter(plotter: _pv.Plotter, *, nbsphinx_export_3d_view: bool = True) -> None:
    """Show a PyVista plotter.

    This wrapper either directly displays the plotter, default
    development use for BeamMe. For the website representation of the
    examples, we export static and interactive versions of the plotter
    that will be embedded in the documentation.

    Args:
        plotter: The PyVista plotter to show.
        nbsphinx_export_3d_view: For nbsphinx documentation, whether to
            export an interactive 3D view alongside the static image.
    """

    if not _is_nbsphinx():
        plotter.show()
    else:
        # Path where static documents are stored for the website.
        static_doc_path = _Path(_os.environ["PYVISTA_DOCS_STATIC"])

        # Get a unique identifier for the current plotter
        plotter_uid = str(_uuid.uuid4().hex)

        # Export a screenshot of the plotter
        plotter.screenshot(static_doc_path / f"{plotter_uid}.png")
        static_frame_html = f"""
            <img src="../_static/pyvista/{plotter_uid}.png"
                style="max-width:100%; border-radius:8px;">
            """

        if nbsphinx_export_3d_view:
            # Export a html representation of the plotter
            plotter.export_html(static_doc_path / f"{plotter_uid}.html")

            interactive_frame = _IFrame(
                src=f"../_static/pyvista/{plotter_uid}.html",
                width="100%",
                height=600,
            )
            tab = _widgets.Tab(
                children=[
                    _widgets.HTML(static_frame_html),
                    _widgets.HTML(interactive_frame._repr_html_()),
                ]
            )
            tab.set_title(0, "Static Scene")
            tab.set_title(1, "Interactive Scene")

            _display(tab)
        else:
            _display(_HTML(static_frame_html))
