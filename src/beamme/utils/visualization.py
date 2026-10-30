# The MIT License (MIT)
#
# Copyright (c) 2018-2026 BeamMe Authors
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
import numpy as _np
import pyvista as _pv
import quaternion as _quaternion
from IPython.display import HTML as _HTML
from IPython.display import IFrame as _IFrame
from IPython.display import display as _display

from beamme.core.conf import bme as _bme
from beamme.utils.environment import is_nbsphinx as _is_nbsphinx
from beamme.utils.environment import is_testing as _is_testing


def add_base_vectors_to_grid(grid: _pv.UnstructuredGrid) -> None:
    """Add base vectors for Cosserat nodes to a PyVista grid.

    Args:
        grid: The PyVista grid to which the base vectors will be added.
            The base vectors are calculated from the point data field "rotation_vector"
            of the grid. Only points that have Cosserat flag in "point_type" will have
            the base vectors added. The base vectors are added as new point data fields
            named "base_vector_1", "base_vector_2", and "base_vector_3".
    """
    if "rotation_vector" not in grid.point_data:
        # In this case we have no rotation vector data, typically because the grid only
        # contains solid elements. In this case we return empty base vectors.
        for i in range(3):
            grid.point_data[f"base_vector_{i + 1}"] = _np.zeros((grid.n_points, 3))
        return

    cosserat_mask = _np.isin(
        grid.point_data["point_type"],
        [_bme.node_type.cosserat.value, _bme.node_type.space_time_cosserat.value],
    )
    quaternions = _quaternion.from_rotation_vector(
        grid.point_data["rotation_vector"][cosserat_mask]
    )
    rotation_matrices = _quaternion.as_rotation_matrix(quaternions)
    for i in range(3):
        base_vector = _np.zeros((grid.n_points, 3))
        base_vector[cosserat_mask] = rotation_matrices[:, :, i]
        grid.point_data[f"base_vector_{i + 1}"] = base_vector


def show_plotter(plotter: _pv.Plotter, *, nbsphinx_export_3d_view: bool = True) -> None:
    """Show a PyVista plotter.

    This function displays the plotter according to the current environment:
    - For local development, it will directly show the plotter.
    - For nbsphinx documentation, it will export a static image of the plotter
      and optionally an interactive 3D view that can be embedded in the
      documentation.

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


def display_pyvista(
    grid: _pv.UnstructuredGrid,
    *,
    beam_nodes=True,
    beam_tube=True,
    beam_cross_section_directors=True,
    resolution=20,
    parallel_projection=False,
):
    """Display the grid in pyvista.

    If this is called in a GitHub testing run, nothing will be shown, instead
    the _pv.plotter object will be returned.

    Args:
        beam_nodes: bool
            If the beam nodes should be displayed. The start and end nodes of each
            beam will be shown in green, possible middle nodes inside the element
            are shown in cyan.
        beam_tube: bool
            If the beam should be rendered as a tube
        beam_cross_section_directors: bool
            If the cross section directors should be displayed (at each node)
        resolution: int
            Indicates how many triangulations will be performed to visualize arrows,
            tubes and spheres.
        parallel_projection: bool
            Flag to change camera view to parallel projection.
    """
    plotter = _pv.Plotter()
    plotter.renderer.add_axes()

    if parallel_projection:
        plotter.enable_parallel_projection()

    beam_mask = grid.cell_data["beamme_type"] == _bme.element_type.beam.value
    if _np.any(beam_mask):
        beam_grid = grid.extract_cells(beam_mask).cell_data_to_point_data()

        # Plot the nodes
        beam_finite_element_nodes = beam_grid.cast_to_poly_points()
        node_radius_scaling_factor = 1.5
        if beam_nodes:
            sphere = _pv.Sphere(
                radius=1.0,
                theta_resolution=resolution,
                phi_resolution=resolution,
            )
            nodes_glyph = beam_finite_element_nodes.glyph(
                geom=sphere,
                scale="cross_section_radius",
                factor=node_radius_scaling_factor,
                orient=False,
            )
            plotter.add_mesh(
                nodes_glyph.threshold(scalars="node_value", value=(0.9, 1.1)),
                color="green",
            )
            middle_nodes = nodes_glyph.threshold(scalars="node_value", value=(0.4, 0.6))
            if len(middle_nodes.points) > 0:
                plotter.add_mesh(middle_nodes, color="cyan")

        # Plot the beams
        beam_color = [0.5, 0.5, 0.5]
        if beam_tube:
            # Check that all beams have a defined cross-section radius.
            if _np.any(beam_grid.point_data["cross_section_radius"] < 0):
                raise ValueError("All beams must have a defined cross-section radius.")

            # TODO: The `algorithm=None` argument is used to avoid a warning from
            # pyvista regarding changed default arguments. For future pyvista
            # versions, this argument can be removed.
            beam_tube_grid = beam_grid.extract_surface(algorithm=None)
            beam_tube_grid = beam_tube_grid.tube(
                scalars="cross_section_radius",
                absolute=True,
                n_sides=resolution,
            )
            plotter.add_mesh(beam_tube_grid, color=beam_color)
        else:
            plotter.add_mesh(beam_grid, color=beam_color, line_width=4)

        # Plot the directors of the beam cross-section
        if beam_cross_section_directors:
            director_radius_scaling_factor = 3.5
            arrow = _pv.Arrow(tip_resolution=resolution, shaft_resolution=resolution)
            directors = [
                beam_finite_element_nodes.glyph(
                    geom=arrow,
                    orient=f"base_vector_{i + 1}",
                    scale="cross_section_radius",
                    factor=director_radius_scaling_factor,
                )
                for i in range(3)
            ]
            colors = ["white", "blue", "red"]
            for i, arrow in enumerate(directors):
                plotter.add_mesh(arrow, color=colors[i])

    solid_mask = grid.cell_data["beamme_type"] == _bme.element_type.solid.value
    if _np.any(solid_mask):
        solid_grid = grid.extract_cells(solid_mask)
        plotter.add_mesh(solid_grid, color="white", show_edges=True, opacity=0.5)

    if not _is_testing():
        show_plotter(plotter)
    else:
        return plotter
