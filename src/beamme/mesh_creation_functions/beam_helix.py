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
"""Functions to create beam meshes along helical paths."""

import warnings as _warnings

import numpy as _np

from beamme.core.mesh import Mesh as _Mesh
from beamme.core.rotation import Rotation as _Rotation

from .beam_line import create_beam_mesh_line as _create_beam_mesh_line


def create_beam_mesh_helix(
    mesh,
    beam_class,
    material,
    axis_vector,
    axis_point,
    start_point,
    *,
    helix_angle=None,
    height_helix=None,
    turns=None,
    warning_straight_line=True,
    **kwargs,
):
    """Generate a helical segment starting at a given start point around a predefined
    axis (defined by axis_vector and axis_point). The helical segment is defined by a
    start_point and exactly two of the basic helical quantities [helix_angle,
    height_helix, turns].

    Args
    ----
    mesh: Mesh
        Mesh that the helical segment will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this segment.
    axis_vector: _np.array, list
        Vector for the orientation of the helical center axis.
    axis_point: _np.array, list
        Point lying on the helical center axis. Does not need to align with
        bottom plane of helix.
    start_point: _np.array, list
        Start point of the helix. Defines the radius.
    helix_angle: float
        Angle of the helix (synonyms in literature: twist angle or pitch
        angle).
    height_helix: float
        Height of helix.
    turns: float
        Number of turns.
    warning_straight_line: bool
        Warn if radius of helix is zero or helix angle is 90 degrees and
        simple line is returned.

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
    if [helix_angle, height_helix, turns].count(None) != 1:
        raise ValueError(
            "Exactly two arguments of [helix_angle, height_helix, turns]"
            " must be provided!"
        )

    if helix_angle is not None and _np.isclose(_np.sin(helix_angle), 0.0):
        raise ValueError(
            "Helix angle of helix is 0 degrees! "
            + "Change angle for feasible helix geometry!"
        )

    if height_helix is not None and _np.isclose(height_helix, 0.0):
        raise ValueError(
            "Height of helix is 0! Change height for feasible helix geometry!"
        )

    # determine radius of helix
    axis_vector = _np.asarray(axis_vector)
    axis_point = _np.asarray(axis_point)
    start_point = _np.asarray(start_point)

    axis_vector = axis_vector / _np.linalg.norm(axis_vector)
    origin = axis_point + _np.dot(
        _np.dot(start_point - axis_point, axis_vector), axis_vector
    )
    start_point_origin_vec = start_point - origin
    radius = _np.linalg.norm(start_point_origin_vec)

    # create temporary mesh to not alter original mesh
    mesh_temp = _Mesh()

    # return line if radius of helix is 0, helix angle is pi/2 or turns is 0
    if (
        _np.isclose(radius, 0)
        or (helix_angle is not None and _np.isclose(_np.cos(helix_angle), 0.0))
        or (turns is not None and _np.isclose(turns, 0.0))
    ):
        if height_helix is None:
            raise ValueError(
                "Radius of helix is 0, helix angle is 90 degrees or turns is 0! "
                + "Fallback to simple line geometry but height cannot be "
                + "determined based on helix angle and turns! Either switch one "
                + "helix parameter to height of helix or change radius!"
            )

        if warning_straight_line:
            _warnings.warn(
                "Radius of helix is 0, helix angle is 90 degrees or turns is 0! "
                + "Simple line geometry is returned!"
            )

        if helix_angle is not None and height_helix is not None:
            end_point = start_point + height_helix * axis_vector * _np.sign(
                _np.sin(helix_angle)
            )
        elif height_helix is not None and turns is not None:
            end_point = start_point + height_helix * axis_vector

        line_sets = _create_beam_mesh_line(
            mesh_temp,
            beam_class,
            material,
            start_point=start_point,
            end_point=end_point,
            **kwargs,
        )

        # add line to mesh
        mesh.add_mesh(mesh_temp)

        return line_sets

    # generate simple helix
    if helix_angle and height_helix:
        end_point = _np.array(
            [
                radius,
                _np.sign(_np.sin(helix_angle)) * height_helix / _np.tan(helix_angle),
                _np.sign(_np.sin(helix_angle)) * height_helix,
            ]
        )
    elif helix_angle and turns:
        end_point = _np.array(
            [
                radius,
                _np.sign(_np.cos(helix_angle)) * 2 * _np.pi * radius * turns,
                _np.sign(_np.cos(helix_angle))
                * 2
                * _np.pi
                * radius
                * _np.abs(turns)
                * _np.tan(helix_angle),
            ]
        )
    elif height_helix and turns:
        end_point = _np.array(
            [
                radius,
                2 * _np.pi * radius * turns,
                height_helix,
            ]
        )

    helix_sets = _create_beam_mesh_line(
        mesh_temp,
        beam_class,
        material,
        start_point=[radius, 0, 0],
        end_point=end_point,
        **kwargs,
    )

    mesh_temp.wrap_around_cylinder()

    # rotate and translate simple helix to align with necessary axis and starting point
    mesh_temp.rotate(
        _Rotation.from_basis(start_point_origin_vec, axis_vector)
        * _Rotation([1, 0, 0], -_np.pi * 0.5)
    )
    mesh_temp.translate(-mesh_temp.nodes[0].coordinates + start_point)

    # add helix to mesh
    mesh.add_mesh(mesh_temp)

    return helix_sets
