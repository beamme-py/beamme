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
"""This file has utility functions for handling NURBS."""

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.rotation import Rotation as _Rotation


def ensure_3d_splinepy_object(splinepy_obj) -> None:
    """Ensure that a given splinepy object has 3D control points.

    Args:
        splinepy_obj:
            Splinepy object to ensure 3D, if this is an object with fewer than
            3 dimensions, it will be converted to 3D by adding the missing
            coordinates with a value of 0.
    """
    control_points_dim = splinepy_obj.control_points.shape[1]
    if control_points_dim == 3:
        pass
    elif control_points_dim < 3:
        # Add "empty" coordinates to make control points 3D
        control_points_3d = _np.zeros([len(splinepy_obj.control_points), 3])
        control_points_3d[:, :control_points_dim] = splinepy_obj.control_points
        splinepy_obj.control_points = control_points_3d
    else:
        raise ValueError(
            f"Splinepy object must be 1D, 2D or 3D, but has {control_points_dim} dimensions."
        )


def translate_splinepy(splinepy_obj, vector) -> None:
    """Translate a splinepy object by a vector.

    Args:
        vector: _np.array, list
            2D/3D vector to translate the splinepy object.
    """
    if not len(vector) == splinepy_obj.control_points.shape[1]:
        raise ValueError(
            f"Dimensions of translation vector and splinepy object do not match: {len(vector)} != {splinepy_obj.control_points.shape[1]}"
        )

    for point in splinepy_obj.control_points:
        point += vector


def rotate_splinepy(splinepy_obj, rotation: _Rotation, origin=None) -> None:
    """Rotate a splinepy object by a rotation object."""
    rotation_matrix = rotation.get_rotation_matrix()

    dimension = splinepy_obj.control_points.shape[1]
    if dimension == 2:
        if not _np.allclose(
            rotation_matrix[2, :], [0, 0, 1], rtol=0, atol=_bme.eps_quaternion
        ) or not _np.allclose(
            rotation_matrix[:, 2], [0, 0, 1], rtol=0, atol=_bme.eps_quaternion
        ):
            raise ValueError(
                "Rotation vector must be in the x-y plane for 2D splinepy objects."
            )
        rotation_matrix = rotation_matrix[:2, :2]

    if origin is None:
        origin = _np.zeros(dimension)

    for i_point, point in enumerate(splinepy_obj.control_points):
        point_new = _np.dot(rotation_matrix, point - origin) + origin
        splinepy_obj.control_points[i_point] = point_new
