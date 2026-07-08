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
"""Unit tests for the NURBS utils functions."""

import numpy as np
import pytest

from beamme.core.rotation import Rotation
from beamme.mesh_creation_functions.nurbs_utils import (
    ensure_3d_splinepy_object,
    rotate_splinepy,
    translate_splinepy,
)


class MockSpline:
    """Minimal mock class to mimic a splinepy object with control points.

    By using this mock, we can test that the translate and rotate functions only modify
    the control points.
    """

    def __init__(self, control_points):
        self.control_points = np.array(control_points)


def test_beamme_mesh_creation_functions_nurbs_utils_translate_splinepy_2d(
    assert_results_close,
):
    """Test translation of a 2D splinepy object."""
    spline = MockSpline([[0.0, 0.0], [1.0, 1.0]])
    translation = [2.0, -1.0]
    translate_splinepy(spline, translation)
    expected = np.array([[2.0, -1.0], [3.0, 0.0]])
    assert_results_close(spline.control_points, expected)


def test_beamme_mesh_creation_functions_nurbs_utils_translate_splinepy_3d(
    assert_results_close,
):
    """Test translation of a 3D splinepy object."""
    spline = MockSpline([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    translation = [1.0, 2.0, 3.0]
    translate_splinepy(spline, translation)
    expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    assert_results_close(spline.control_points, expected)


def test_beamme_mesh_creation_functions_nurbs_utils_translate_invalid_dimension():
    """Test that translation with a vector of incorrect dimension raises error."""
    spline = MockSpline([[0.0, 0.0]])
    with pytest.raises(ValueError, match="Dimensions of translation"):
        translate_splinepy(spline, [1.0, 2.0, 3.0])


def test_beamme_mesh_creation_functions_nurbs_utils_rotate_splinepy_2d(
    assert_results_close,
):
    """Test rotation of a 2D splinepy object."""
    points = [[1.0, 0.0], [0.0, 1.0]]
    spline = MockSpline(points)
    rotation = Rotation([0, 0, 1], np.pi / 2)
    rotate_splinepy(spline, rotation)
    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert_results_close(spline.control_points, expected)


def test_beamme_mesh_creation_functions_nurbs_utils_rotate_splinepy_invalid_2d_rotation():
    """Test that a 2D splinepy object can only be rotated by a rotation around the
    z-axis."""
    spline = MockSpline([[1.0, 0.0]])
    rotation = Rotation([1, 1, 0], np.pi / 3)
    with pytest.raises(ValueError, match="Rotation vector must be in"):
        rotate_splinepy(spline, rotation)


@pytest.mark.parametrize("origin", [None, [1.0, 2.0, 3.0]])
def test_beamme_mesh_creation_functions_nurbs_utils_rotate_splinepy_3d_origin(
    origin, assert_results_close
):
    """Test rotation of a 3D splinepy object with and without given origin argument."""
    points = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    spline = MockSpline(points)
    rotation = Rotation([1.0, 2.0, 3.0], np.pi / 3.0)
    rotate_splinepy(spline, rotation, origin=origin)
    origin_for_reference = np.zeros(3) if origin is None else np.array(origin)
    rotation_matrix = rotation.get_rotation_matrix()
    expected = np.array(
        [
            rotation_matrix @ (p - origin_for_reference) + origin_for_reference
            for p in points
        ]
    )
    assert_results_close(spline.control_points, expected)


def test_beamme_mesh_creation_functions_nurbs_utils_ensure_3d_splinepy_object_already_3d():
    """Ensure that a 3D splinepy object is left unchanged."""
    original_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    spline = MockSpline(original_points)
    spline_cp_initial = spline.control_points
    ensure_3d_splinepy_object(spline)
    # Control point object should be same identical object
    assert spline.control_points is spline_cp_initial


def test_beamme_mesh_creation_functions_nurbs_utils_ensure_3d_splinepy_object_from_1d(
    assert_results_close,
):
    """Ensure that a 1D splinepy object is converted to 3D by adding y=0 and z=0."""
    original_points = np.array([[1.0], [3.0]])
    spline = MockSpline(original_points)
    ensure_3d_splinepy_object(spline)
    expected = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert_results_close(spline.control_points, expected)


def test_beamme_mesh_creation_functions_nurbs_utils_ensure_3d_splinepy_object_from_2d(
    assert_results_close,
):
    """Ensure that a 2D splinepy object is converted to 3D by adding z=0."""
    original_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    spline = MockSpline(original_points)
    ensure_3d_splinepy_object(spline)
    expected = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    assert_results_close(spline.control_points, expected)


def test_beamme_mesh_creation_functions_nurbs_utils_ensure_3d_splinepy_object_invalid_dimension():
    """Ensure that an invalid spline dimension raises ValueError."""
    spline = MockSpline(np.zeros((2, 4)))  # 4D control points
    with pytest.raises(ValueError, match="Splinepy object must be 1D, 2D or 3D"):
        ensure_3d_splinepy_object(spline)
