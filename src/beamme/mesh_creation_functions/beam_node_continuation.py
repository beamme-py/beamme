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
"""Functions to create meshed beam geometries at a node."""

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_axis as _create_beam_mesh_arc_segment_via_axis,
)
from beamme.mesh_creation_functions.beam_line import (
    create_beam_mesh_line as _create_beam_mesh_line,
)
from beamme.utils.nodes import get_single_node as _get_single_node


def create_beam_mesh_line_at_node(
    mesh, beam_class, material, start_node, length, **kwargs
):
    """Generate a straight line at a given node.

    The tangent will be the same as at that node.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    start_node: _np.array, list
        Point where the arc will continue.
    length: float
        Length of the line.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. Mutually exclusive with n_el.
        Be aware, that this length might not be achieved, if the elements are
        warped after they are created.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """
    if length < 0:
        raise ValueError("Length has to be positive!")

    # Create the line starting from the given node
    start_node = _get_single_node(start_node)
    tangent = start_node.rotation * [1, 0, 0]
    start_position = start_node.coordinates
    end_position = start_position + tangent * length

    return _create_beam_mesh_line(
        mesh,
        beam_class,
        material,
        start_position,
        end_position,
        start_node=start_node,
        **kwargs,
    )


def create_beam_mesh_arc_at_node(
    mesh, beam_class, material, start_node, arc_axis_normal, radius, angle, **kwargs
):
    """Generate a circular segment starting at a given node.

    The arc will be tangent to the given node.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    start_node: _np.array, list
        Point where the arc will continue.
    arc_axis_normal: 3d-vector
        Rotation axis for the created arc.
    radius: float
        The radius of the arc segment.
    angle: float
        Angle of the arc. If the angle is negative, the arc will point in the
        opposite direction, i.e., as if the arc_axis_normal would change sign.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. Mutually exclusive with n_el.
        Be aware, that this length might not be achieved, if the elements are
        warped after they are created.

    Return
    ----
    return_set: GeometryName
        Set with the 'start' and 'end' node of the line. Also a 'line' set
        with all nodes of the line.
    """
    # If the angle is negative, the normal is switched
    arc_axis_normal = _np.asarray(arc_axis_normal)
    if angle < 0:
        arc_axis_normal = -1.0 * arc_axis_normal

    # The normal has to be perpendicular to the start point tangent
    start_node = _get_single_node(start_node)
    tangent = start_node.rotation * [1, 0, 0]
    if _np.abs(_np.dot(tangent, arc_axis_normal)) > _bme.eps_pos:
        raise ValueError(
            "The normal has to be perpendicular to the tangent in the start node!"
        )

    # Get the center of the arc
    center_direction = _np.cross(tangent, arc_axis_normal)
    center_direction *= 1.0 / _np.linalg.norm(center_direction)
    center = start_node.coordinates - center_direction * radius

    return _create_beam_mesh_arc_segment_via_axis(
        mesh,
        beam_class,
        material,
        arc_axis_normal,
        center,
        start_node.coordinates,
        _np.abs(angle),
        start_node=start_node,
        **kwargs,
    )
