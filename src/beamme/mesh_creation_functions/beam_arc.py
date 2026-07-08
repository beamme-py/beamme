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
"""Functions to create beam meshes along arcs."""

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.rotation import Rotation as _Rotation
from beamme.mesh_creation_functions.beam_generic import (
    create_beam_mesh_generic as _create_beam_mesh_generic,
)


def create_beam_mesh_arc_segment_via_rotation(
    mesh, beam_class, material, center, axis_rotation, radius, angle, **kwargs
):
    """Generate a circular segment of beam elements.

    The circular segment is defined via a rotation, specifying the "initial"
    triad of the beam at the beginning of the arc.

    This function exists for compatibility reasons with older BeamMe implementations.
    The user is encouraged to use the newer implementation create_beam_mesh_arc_segment_via_axis

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    center: _np.array, list
        Center of the arc.
    axis_rotation: Rotation
        This rotation defines the spatial orientation of the arc.
        The 3rd base vector of this rotation is the rotation axis of the arc
        segment. The segment starts in the direction of the 1st basis vector
        and the starting point is along the 2nd basis vector.
    radius: float
        The radius of the segment.
    angle: float
        The central angle of this segment in radians.

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
    # Convert the input to the one for create_beam_mesh_arc_segment_via_axis
    axis = axis_rotation * [0, 0, 1]
    start_point = center + radius * (axis_rotation * [0, -1, 0])
    return create_beam_mesh_arc_segment_via_axis(
        mesh, beam_class, material, axis, center, start_point, angle, **kwargs
    )


def create_beam_mesh_arc_segment_via_axis(
    mesh,
    beam_class,
    material,
    axis,
    axis_point,
    start_point,
    angle,
    **kwargs,
):
    """Generate a circular segment of beam elements.

    The arc is defined via a rotation axis, a point on the rotation axis a starting
    point, as well as the angle of the arc segment.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    axis: _np.array, list
        Rotation axis of the arc.
    axis_point: _np.array, list
        Point lying on the rotation axis. Does not have to be the center of the arc.
    start_point: _np.array, list
        Start point of the arc.
    angle: float
        The central angle of this segment in radians.

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
    # The angle can not be negative with the current implementation.
    if angle <= 0.0:
        raise ValueError(
            "The angle for a beam arc segment has to be a positive number!"
        )

    # Shortest distance from the given point to the axis of rotation gives
    # the "center" of the arc
    axis = _np.asarray(axis)
    axis_point = _np.asarray(axis_point)
    start_point = _np.asarray(start_point)

    axis = axis / _np.linalg.norm(axis)
    diff = start_point - axis_point
    distance = diff - _np.dot(_np.dot(diff, axis), axis)
    radius = _np.linalg.norm(distance)
    center = start_point - distance

    # Get the rotation at the start
    # No need to check the start node here, as eventual rotation offsets in
    # tangential direction will be covered by the create beam functionality.
    tangent = _np.cross(axis, distance)
    tangent /= _np.linalg.norm(tangent)
    start_rotation = _Rotation.from_rotation_matrix(
        _np.transpose(_np.array([tangent, -distance / radius, axis]))
    )

    def beam_function(phi: float) -> tuple[_np.ndarray, _Rotation, float | None]:
        """Return a point on the beams axis for a given interval coordinate."""
        arc_rotation = _Rotation(axis, phi)
        rot = arc_rotation * start_rotation
        pos = center + arc_rotation * distance
        return (pos, rot, phi * radius)

    # Create the beam in the mesh
    return _create_beam_mesh_generic(
        mesh,
        beam_class=beam_class,
        material=material,
        beam_function=beam_function,
        interval=[0.0, angle],
        interval_length=angle * radius,
        **kwargs,
    )


def create_beam_mesh_arc_segment_2d(
    mesh, beam_class, material, center, radius, phi_start, phi_end, **kwargs
):
    """Generate a circular segment of beam elements in the x-y plane.

    Args
    ----
    mesh: Mesh
        Mesh that the arc segment will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    center: _np.array, list
        Center of the arc. If the z component is not 0, an error will be
        thrown.
    radius: float
        The radius of the segment.
    phi_start, phi_end: float
        The start and end angles of the arc w.r.t the x-axis. If the start
        angle is larger than the end angle the beam faces in counter-clockwise
        direction, and if the start angle is smaller than the end angle, the
        beam faces in clockwise direction.

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
    # The center point has to be on the x-y plane.
    if _np.abs(center[2]) > _bme.eps_pos:
        raise ValueError("The z-value of center has to be 0!")

    # Check if the beam is in clockwise or counter clockwise direction.
    angle = phi_end - phi_start
    axis = _np.array([0, 0, 1])
    start_point = center + radius * (_Rotation(axis, phi_start) * [1, 0, 0])

    counter_clockwise = _np.sign(angle) == 1
    if not counter_clockwise:
        # If the beam is not in counter clockwise direction, we have to flip
        # the rotation axis.
        axis = -1.0 * axis

    return create_beam_mesh_arc_segment_via_axis(
        mesh,
        beam_class,
        material,
        axis,
        center,
        start_point,
        _np.abs(angle),
        **kwargs,
    )
