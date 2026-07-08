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
"""This file has functions to create a honeycomb beam mesh."""

import numpy as _np

from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.rotation import Rotation as _Rotation
from beamme.mesh_creation_functions.beam_line import (
    create_beam_mesh_line as _create_beam_mesh_line,
)
from beamme.utils.nodes import get_min_max_nodes as _get_min_max_nodes


def create_beam_mesh_honeycomb_flat(
    mesh,
    beam_class,
    material,
    width,
    n_width,
    n_height,
    *,
    n_el=1,
    closed_width=True,
    closed_height=True,
    create_couplings=True,
):
    """Add a flat honeycomb structure.

    The structure will be created in the x-y plane.

    Args
    ----
    mesh: Mesh
        Mesh that the honeycomb will be added to.
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    width: float
        Width of one honeycomb.
    n_width: int
        Number of honeycombs in x-direction.
    n_height: int
        Number of honeycombs in y-direction.
    n_el: int
        Number of elements per beam line.
    closed_width: bool
        If the last honeycombs in x-direction will be closed.
    closed_height: bool
        If the last vertical lines in y-direction will be created.
    create_couplings: bool
        If the nodes will be connected in this function.

    Return
    ----
    return_set: GeometryName
        Set with nodes on the north, south, east and west boundaries. Those
        sets only contains end nodes of lines, not the middle ones. The set
        'all' contains all nodes.
    """

    def add_line(pointa, pointb):
        """Shortcut to add line."""
        return _create_beam_mesh_line(
            mesh_honeycomb, beam_class, material, pointa, pointb, n_el=n_el
        )

    # Geometrical shortcuts.
    sin30 = _np.sin(_np.pi / 6)
    cos30 = _np.sin(2 * _np.pi / 6)
    a = width * 0.5 / cos30
    nx = _np.array([1.0, 0.0, 0.0])
    ny = _np.array([0.0, 1.0, 0.0])
    zig_zag_x = nx * width * 0.5
    zig_zag_y = ny * a * sin30 * 0.5

    # Create the honeycomb structure
    mesh_honeycomb = _Mesh()
    origin = _np.array([0, a * 0.5 * sin30, 0])
    for i_height in range(n_height + 1):
        # Start point for this zig-zag line.
        base_row = origin + (2 * zig_zag_y + a * ny) * i_height

        # If the first node is up or down of base_row.
        if i_height % 2 == 0:
            direction = 1
        else:
            direction = -1

        for i_width in range(n_width + 1):
            base_zig_zag = base_row + direction * zig_zag_y + width * i_width * nx

            # Do not add a zig-zag line on the last run (that one is only
            # for the remaining vertical lines).
            if i_width < n_width:
                add_line(
                    base_zig_zag, base_zig_zag + zig_zag_x - 2 * direction * zig_zag_y
                )
                add_line(
                    base_zig_zag + zig_zag_x - 2 * direction * zig_zag_y,
                    base_zig_zag + nx * width,
                )

            # Check where the vertical lines start.
            if i_height % 2 == 0:
                base_vert = base_zig_zag
            else:
                base_vert = base_zig_zag + zig_zag_x - 2 * direction * zig_zag_y

            # Only add vertical lines at the end if closed_width.
            if (i_width < n_width) or (direction == 1 and closed_width):
                # Check if the vertical lines at the top should be added.
                if not (i_height == n_height) or (not closed_height):
                    add_line(base_vert, base_vert + ny * a)

    # List of nodes from the honeycomb that are candidates for connections.
    honeycomb_nodes = [node for node in mesh_honeycomb.nodes if node.is_end_node]

    # Add connections for the nodes with same positions.
    if create_couplings:
        mesh_honeycomb.couple_nodes(nodes=honeycomb_nodes)

    # Get min and max nodes of the honeycomb.
    min_max_nodes = _get_min_max_nodes(honeycomb_nodes)

    # Return the geometry set.
    return_set = _GeometryName()
    return_set["north"] = min_max_nodes["y_max"]
    return_set["east"] = min_max_nodes["x_max"]
    return_set["south"] = min_max_nodes["y_min"]
    return_set["west"] = min_max_nodes["x_min"]
    return_set["all"] = _GeometrySet(mesh_honeycomb.elements)

    mesh.add(mesh_honeycomb)

    return return_set


def create_beam_mesh_honeycomb(
    mesh,
    beam_class,
    material,
    diameter,
    n_circumference,
    n_axis,
    *,
    n_el=1,
    closed_top=True,
    vertical=True,
):
    """Wrap a honeycomb structure around a cylinder.

    The cylinder axis will be the z-axis.

    Args
    ----
    mesh: Mesh
        Mesh that the honeycomb will be added to.
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    diameter: float
        Diameter of the cylinder.
    n_circumference: int
        Number of honeycombs around the diameter. If vertical is False this
        has to be an odd number.
    n_axis: int
        Number of honeycombs in axial-direction.
    n_el: int
        Number of elements per beam line.
    closed_top: bool
        If the last honeycombs in axial-direction will be closed.
    vertical: bool
        If there are vertical lines in the honeycomb or horizontal.

    Return
    ----
    return_set: GeometryName
        Set with nodes on the top and bottom boundaries. Those sets only
        contains end nodes of lines, not the middle ones. The set "all"
        contains all nodes.
    """
    # Calculate the input values for the flat honeycomb mesh.
    if vertical:
        width = diameter * _np.pi / n_circumference
        closed_width = False
        closed_height = closed_top
        rotation = _Rotation([0, 0, 1], _np.pi / 2) * _Rotation([1, 0, 0], _np.pi / 2)
        n_height = n_axis
        n_width = n_circumference
    else:
        if not n_circumference % 2 == 0:
            raise ValueError(
                "There has to be an even number of elements along the diameter in horizontal mode. "
                "Given: {}!".format(n_circumference)
            )
        H = diameter * _np.pi / n_circumference
        r = H / (1 + _np.sin(_np.pi / 6))
        width = 2 * r * _np.cos(_np.pi / 6)
        closed_width = closed_top
        closed_height = False
        rotation = _Rotation([0, 1, 0], -0.5 * _np.pi)
        n_height = n_circumference - 1
        n_width = n_axis

    # Create the flat mesh, do not create couplings, as they will be added
    # later in this function, where also the diameter nodes will be
    # connected.
    mesh_temp = _Mesh()
    honeycomb_sets = create_beam_mesh_honeycomb_flat(
        mesh_temp,
        beam_class,
        material,
        width,
        n_width,
        n_height,
        n_el=n_el,
        closed_width=closed_width,
        closed_height=closed_height,
        create_couplings=False,
    )

    # Move the mesh to the correct position.
    mesh_temp.rotate(rotation)
    mesh_temp.translate([diameter / 2, 0, 0])
    mesh_temp.wrap_around_cylinder()

    # Add connections for the nodes with same positions.
    honeycomb_nodes = [node for node in mesh_temp.nodes if node.is_end_node]
    mesh_temp.couple_nodes(nodes=honeycomb_nodes)

    # Return the geometry set'
    return_set = _GeometryName()
    return_set["all"] = honeycomb_sets["all"]
    if vertical:
        return_set["top"] = honeycomb_sets["north"]
        return_set["bottom"] = honeycomb_sets["south"]
    else:
        return_set["top"] = honeycomb_sets["east"]
        return_set["bottom"] = honeycomb_sets["west"]

    # Add to this mesh
    mesh.add_mesh(mesh_temp)

    return return_set
