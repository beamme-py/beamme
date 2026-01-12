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
"""This file contains utility functions for the first example."""

# We need to import vtk before pyvista for the TeX labels to work
import vtk  # noqa: F401, I001

import numpy as np
import pyvista as pv
import sys

from beamme.utils.environment import is_testing
import beamme.utils.visualization as visualization


def print_matrix(name, matrix):
    """Print a matrix to the console."""
    print(f"{name}:\n{np.round(matrix, decimals=10)}")


def print_rotation_matrix(name, rotation):
    """Print the rotation matrix to the console."""
    print_matrix(name, rotation.get_rotation_matrix())


def add_cube_plot(plotter, row, col, rotation, text, *, plot_outlines=True):
    """Add a cube to the plotter."""

    plotter.subplot(row, col)

    # Define and optionally plot the original cube
    cube_original = pv.Cube()
    cube_original = cube_original.scale([3, 2, 1])
    if plot_outlines:
        plotter.add_mesh(
            cube_original, scalars=None, show_edges=True, style="wireframe"
        )

    # Rotate the cube
    rotation_vector = rotation.get_rotation_vector()
    cube = cube_original.rotate_vector(
        vector=rotation_vector, angle=np.linalg.norm(rotation_vector) * 180 / np.pi
    )

    # Define colors for each face of the cube (Rubik's cube style)
    face_colors = np.array(
        [
            [255, 165, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 255, 255],
        ]
    )
    cube.cell_data["colors"] = face_colors

    # Plot the cube and the title text
    plotter.add_mesh(cube, scalars="colors", show_edges=True, rgb=True)
    plotter.add_text(text, font_size=12)

    # Plot the axes
    for i_dir, (color, name) in enumerate(
        zip(["red", "green", "blue"], ["x", "y", "z"])
    ):
        direction = np.zeros(3)
        direction[i_dir] = 3.0
        factor = 0.5
        arrow = pv.Arrow(
            start=[0, 0, 0],
            direction=direction,
            scale="auto",
            tip_radius=0.1 * factor,
            shaft_radius=0.05 * factor,
            tip_length=0.25 * factor,
        )
        plotter.add_mesh(arrow, color=color)
        plotter.add_point_labels(
            [1.075 * direction],
            [name],
            shadow=True,
            shape=None,
            show_points=False,
            font_size=32,
            justification_horizontal="center",
            justification_vertical="center",
        )

    plotter.show_axes()


class PyVistaPlotter:
    """Class to handle pyvista plotters in the examples."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # PyVista changes the printout, so store the original print out
        # and restore it when done.
        self.stdout = sys.stdout

    def __enter__(self):
        """Return the plotter with the given arguments."""

        self.plotter = pv.Plotter(*self.args, **self.kwargs)
        return self.plotter

    def __exit__(self, exc_type, exc_value, traceback):
        """When exiting the with statement, call this function.

        We show the plotter (except during testing and we reset the
        console print out).
        """
        if not is_testing():
            visualization.show_plotter(self.plotter, nbsphinx_export_3d_view=False)
        sys.stdout = self.stdout
