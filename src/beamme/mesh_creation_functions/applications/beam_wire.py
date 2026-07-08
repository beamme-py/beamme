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
"""Create a steel wire."""

import numpy as _np

from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.mesh_creation_functions.beam_line import (
    create_beam_mesh_line as _create_beam_mesh_line,
)
from beamme.utils.nodes import check_node_by_coordinate as _check_node_by_coordinate


def create_wire_fibers(
    mesh, beam_class, material, length, *, radius=None, layers=1, n_el=1
):
    """Create a steel wire consisting of multiple filaments.

    The wire will be oriented in x-direction.

    Args
    ----
    mesh: Mesh
        Mesh that the line will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    length: float
        Length of the wire.
    radius: float
        If this parameter is given, not the beam cross section radius will be
        taken, but this one.
    layers: int
        Number of layers to be used for the wire.
    n_el: int
        Number of beam elements per filament.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' nodes of the wire. Also a 'all' set
        with all nodes of the wire.
    """
    if len(mesh.nodes) != 0:
        raise ValueError(
            "The create_wire_fibers function can only be used with an empty mesh."
        )

    if radius is None:
        wire_beam_radius = material.radius
    else:
        wire_beam_radius = radius

    def create_line(pos_yz):
        """Create a line starting at the yz-plane with the 2D coordinates pos_yz."""
        _create_beam_mesh_line(
            mesh,
            beam_class,
            material,
            [0.0, pos_yz[0], pos_yz[1]],
            [length, pos_yz[0], pos_yz[1]],
            n_el=n_el,
        )

    # Create the center filament.
    create_line([0.0, 0.0])

    # Create the filaments in the layers.
    for i_angle in range(6):
        angle = i_angle * _np.pi / 3.0
        direction_radial = _np.array([_np.cos(angle), _np.sin(angle)])
        angle = i_angle * _np.pi / 3.0 + 2.0 * _np.pi / 3.0
        direction_tangential = _np.array([_np.cos(angle), _np.sin(angle)])
        for i_layer in range(layers):
            for i_tangent in range(i_layer + 1):
                pos = (
                    2.0
                    * wire_beam_radius
                    * (
                        direction_radial * (i_layer + 1)
                        + direction_tangential * i_tangent
                    )
                )
                create_line(pos)

    # Create the sets to return.
    return_set = _GeometryName()
    start_nodes = mesh.get_nodes_by_function(_check_node_by_coordinate, 0, 0.0)
    end_nodes = mesh.get_nodes_by_function(_check_node_by_coordinate, 0, length)
    return_set["start"] = _GeometrySet(start_nodes)
    return_set["end"] = _GeometrySet(end_nodes)
    return_set["all"] = _GeometrySet(mesh.elements)
    return return_set
