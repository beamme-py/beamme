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
"""This script is used to unittest the functionality of the Rotation class."""

import numpy as np
import pytest

from beamme.core.conf import bme
from beamme.core.rotation import Rotation, get_rotation_vector_series, smallest_rotation


def get_rotation_matrix(axis, alpha):
    """Create a rotation about one of the Cartesian axis.

    Args
    ----
    axis: int
        0 - x
        1 - y
        2 - z
    angle: double rotation angle

    Return
    ----
    rot3D: array(3x3)
        Rotation matrix for this rotation
    """
    c, s = np.cos(alpha), np.sin(alpha)
    rot2D = np.array(((c, -s), (s, c)))
    index = [np.mod(j, 3) for j in range(axis, axis + 3) if not j == axis]
    rot3D = np.eye(3)
    rot3D[np.ix_(index, index)] = rot2D
    return rot3D


def test_beamme_core_rotations_cartesian_rotations(assert_results_close):
    """Create a rotation in all 3 directions.

    And compare with the rotation matrix.
    """
    theta = 1.0
    # Loop per direction.
    for i in range(3):
        rot3D = get_rotation_matrix(i, theta)
        axis = np.zeros(3)
        axis[i] = 1
        angle = theta
        rotation = Rotation(axis, angle)

        # Check if the rotation is the same if it is created from its own
        # quaternion and then created from its own rotation matrix.
        rotation = Rotation.from_quaternion(rotation.get_quaternion())
        rotation_matrix = Rotation.from_rotation_matrix(rotation.get_rotation_matrix())

        assert_results_close(rot3D, rotation_matrix.get_rotation_matrix())


def test_beamme_core_rotations_euler_angles(assert_results_close):
    """Create a rotation with Euler angles and compare to known results."""
    # Euler angles.
    alpha = 1.1
    beta = 1.2 * np.pi * 10
    gamma = -2.5

    # Create the rotation with rotation matrices.
    Rx = get_rotation_matrix(0, alpha)
    Ry = get_rotation_matrix(1, beta)
    Rz = get_rotation_matrix(2, gamma)
    R_euler = Rz.dot(Ry.dot(Rx))

    # Create the rotation with the Rotation object.
    rotation_x = Rotation([1, 0, 0], alpha)
    rotation_y = Rotation([0, 1, 0], beta)
    rotation_z = Rotation([0, 0, 1], gamma)
    rotation_euler = rotation_z * rotation_y * rotation_x
    assert_results_close(R_euler, rotation_euler.get_rotation_matrix())
    assert rotation_euler == Rotation.from_rotation_matrix(R_euler)

    # Direct formula for quaternions for Euler angles.
    quaternion = np.zeros(4)
    cy = np.cos(gamma * 0.5)
    sy = np.sin(gamma * 0.5)
    cr = np.cos(alpha * 0.5)
    sr = np.sin(alpha * 0.5)
    cp = np.cos(beta * 0.5)
    sp = np.sin(beta * 0.5)
    quaternion[0] = cy * cr * cp + sy * sr * sp
    quaternion[1] = cy * sr * cp - sy * cr * sp
    quaternion[2] = cy * cr * sp + sy * sr * cp
    quaternion[3] = sy * cr * cp - cy * sr * sp
    assert Rotation.from_quaternion(quaternion) == rotation_euler
    assert Rotation.from_quaternion(quaternion) == Rotation.from_quaternion(
        rotation_euler.get_quaternion()
    )
    assert Rotation.from_quaternion(quaternion) == Rotation.from_rotation_matrix(
        R_euler
    )


def test_beamme_core_rotations_negative_angles():
    """Check if a rotation is created correctly if a negative angle or a large angle is
    given."""
    vector = 10 * np.array([-1.234243, -2.334343, -1.123123])
    phi = -12.152101868665
    rot = Rotation(vector, phi)
    for i in range(2):
        assert rot == Rotation(vector, phi + 2 * i * np.pi)

    rot = Rotation.from_rotation_vector(vector)
    q = rot.q
    assert rot == Rotation.from_quaternion(-q)
    assert Rotation.from_quaternion(q) == Rotation.from_quaternion(-q)


def test_beamme_core_rotations_inverse_rotation():
    """Test the inv() function for rotations."""
    # Define test rotation.
    rot = Rotation([1, 2, 3], 2)

    # Check if inverse rotation gets identity rotation. Use two different
    # constructors for identity rotation.
    assert Rotation.from_rotation_vector([0, 0, 0]) == rot * rot.inv()
    assert Rotation() == rot * rot.inv()

    # Check that there is no warning or error when getting the vector for
    # an identity rotation.
    (rot * rot.inv()).get_rotation_vector()


def test_beamme_core_rotations_rotation_vector(assert_results_close):
    """Test if the rotation vector functions give a correct result."""
    # Calculate rotation vector and quaternion.
    axis = np.array([1.36568, -2.96784, 3.23346878])
    angle = 0.7189467
    rotation_vector = angle * axis / np.linalg.norm(axis)
    q = np.zeros(4)
    q[0] = np.cos(angle / 2)
    q[1:] = np.sin(angle / 2) * axis / np.linalg.norm(axis)

    # Check that the rotation object from the quaternion and rotation
    # vector are equal.
    rotation_from_vec = Rotation.from_rotation_vector(rotation_vector)
    assert Rotation.from_quaternion(q) == rotation_from_vec
    assert Rotation(axis, angle) == rotation_from_vec

    # Check that the same rotation vector is returned after being converted
    # to a quaternion.
    assert_results_close(rotation_vector, rotation_from_vec.get_rotation_vector())


def test_beamme_core_rotations_rotation_operator_overload(assert_results_close):
    """Test if the operator overloading gives a correct result."""
    # Calculate rotation and vector.
    axis = np.array([1.36568, -2.96784, 3.23346878])
    angle = 0.7189467
    rot = Rotation(axis, angle)
    vector = [2.234234, -4.213234, 6.345234]

    # Check the result of the operator overloading.
    result_vector = np.dot(rot.get_rotation_matrix(), vector)
    assert_results_close(result_vector, rot * vector)
    assert_results_close(result_vector, rot * np.array(vector))


@pytest.mark.parametrize(
    "vectors",
    [
        [[1, 0, 0], [0, -1, 0]],
        [[0, 0, 1], [0, 1, 0]],
        [[-1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1]],
    ],
)
def test_beamme_core_rotations_from_rotation_matrix(vectors, assert_results_close):
    """Test if the correct quaternions are generated from a rotation matrix.

    The from_rotation_matrix function has different branches, the input vectors to this
    functions are chosen to trigger all branches.
    """
    t1, t2 = vectors
    rot = Rotation().from_basis(t1, t2)
    t1_rot = rot * [1, 0, 0]
    t2_rot = rot * [0, 1, 0]
    assert_results_close(t1, t1_rot)
    assert_results_close(t2, t2_rot)


def test_beamme_core_rotations_from_basis(assert_results_close):
    """Test the from_basis function for general input values."""
    t1 = [1, 2, 3]
    t2 = [-1, 3, 1]
    rot = Rotation().from_basis(t1, t2)
    t1_rot = rot * [1, 0, 0]
    t3_rot = rot * [0, 0, 1]

    # The first basis vector has to be in t1 direction
    assert_results_close(t1 / np.linalg.norm(t1), t1_rot)

    # The third basis vector has to be orthogonal to t1 and t2
    for vector in (t1, t2):
        assert_results_close(np.dot(vector, t3_rot), 0.0)

    # Check that an error is raised for wrong input parameters
    with pytest.raises(ValueError, match="The given vector t1"):
        Rotation().from_basis([0, 0, 0], [2, 4, 6])
    with pytest.raises(ValueError, match="Got two vectors"):
        Rotation().from_basis([1, 2, 3], [2, 4, 6])
    with pytest.raises(ValueError, match="Got two vectors"):
        Rotation().from_basis([1, 2, 3], [0, 0, 0])


def test_beamme_core_rotations_transformation_matrix(assert_results_close):
    """Test that the transformation matrix is computed correctly."""
    rotation_vector_large = [1.0, 2.0, np.pi / 5.0]
    rotation_large = Rotation.from_rotation_vector(rotation_vector_large)
    rotation_vector_small = (
        rotation_vector_large
        / np.linalg.norm(rotation_vector_large)
        / 10.0
        * bme.eps_quaternion
    )
    rotation_small = Rotation.from_rotation_vector(rotation_vector_small)

    # Test transformation matrix
    transformation_matrix_large_reference = np.array(
        [
            [0.5959488405656389, 0.49803685445056006, -0.9422331516950085],
            [-0.13028167626739845, 0.8717652242030102, 0.6155336966099826],
            [1.0577668483049911, -0.3844663033900173, 0.5403060272710478],
        ]
    )
    assert_results_close(
        rotation_large.get_transformation_matrix(),
        transformation_matrix_large_reference,
    )
    assert_results_close(rotation_small.get_transformation_matrix(), np.identity(3))

    # Test transformation matrix inverse
    transformation_matrix_inverse_large_reference = np.array(
        [
            [0.44154375784863675, 0.05812896596538626, 0.7037804689849043],
            [0.4501610612860693, 0.8227612782872283, -0.15228520755418612],
            [-0.5440964474342915, 0.4716532506554118, 0.36463746593568075],
        ]
    )
    assert_results_close(
        rotation_large.get_transformation_matrix_inv(),
        transformation_matrix_inverse_large_reference,
    )
    assert_results_close(rotation_small.get_transformation_matrix_inv(), np.identity(3))


def test_beamme_core_rotations_smallest_rotation_triad(assert_results_close):
    """Test that the smallest rotation triad is calculated correctly."""
    # Get the triad obtained by a smallest rotation from an arbitrary triad
    # onto an arbitrary tangent vector.
    rot = Rotation([1, 2, 3], 0.431 * np.pi)
    tan = [2.0, 3.0, -1.0]
    rot_smallest = smallest_rotation(rot, tan)

    rot_smallest_ref = [
        0.853329730651268,
        0.19771093216880734,
        0.25192421451158936,
        0.4114279380770031,
    ]
    assert_results_close(rot_smallest.q, rot_smallest_ref)


def test_beamme_core_rotations_error_accumulation_multiplication(assert_results_close):
    """Test that error accumulation of successive multiplications of rotations does not
    affect the results."""
    rotation_1 = Rotation([1, 2, 3], 0.3)
    rotation_2 = Rotation([1, -1, -2], np.pi / 6)
    rotation_3 = Rotation([-1, -2, -3], 7 * np.pi / 17)
    rotation = Rotation()
    for _ in range(100):
        rotation = rotation_1 * rotation * rotation_2
        rotation = rotation * rotation_3

    q_ref = [
        -0.38478914485223104,
        -0.0385171948379694,
        -0.49122781649072017,
        -0.780479962594468,
    ]
    assert_results_close(q_ref, rotation.q)


def test_beamme_core_rotations_error_accumulation_smallest_rotation(
    assert_results_close,
):
    """Test that error accumulation of successive smallest rotation mappings does not
    affect the results.

    Calculate the smallest rotation onto a vector and then rotate that vector "away" to
    calculate the next smallest rotation and so on...
    """
    tangent = [0.9, 0.1, -0.3]
    rotation_old = Rotation([1, 2, 3], 0.3)

    for _ in range(50):
        rotation_new = smallest_rotation(rotation_old, tangent)
        tangent = rotation_new * rotation_old.inv() * tangent
        rotation_old = rotation_new

    q_ref = [
        0.6329069205124062,
        0.13331392718187732,
        -0.5128773537467728,
        0.5644581887089211,
    ]
    assert_results_close(q_ref, rotation_new.q)


def test_beamme_core_rotations_rotations_rotation_vector_series(assert_results_close):
    """Test the function get_rotation_vector_series."""
    director = np.array([1, 2, 3])
    director = director / np.linalg.norm(director)
    director_2 = np.array([1.1, 2.5, 3.1])
    director_2 = 1.1 * director_2 / np.linalg.norm(director_2)

    # First rotation is 0, second is also zero but with a full 2pi rotation.
    rotation_vectors = [
        [0, 0, 0],
        2.0 * np.pi * director,
    ]
    rotation_vectors_ref = [[0, 0, 0], [0, 0, 0]]
    assert_results_close(
        get_rotation_vector_series(rotation_vectors), rotation_vectors_ref
    )

    # First rotation is close to 2pi, second is 0.
    rotation_vectors = [1.9 * np.pi * director, [0, 0, 0]]
    rotation_vectors_ref = [1.9 * np.pi * director, 2.0 * np.pi * director]
    assert_results_close(
        get_rotation_vector_series(rotation_vectors), rotation_vectors_ref
    )

    # Two similar rotations, no jump over 2pi.
    rotation_vectors = [director, director_2]
    rotation_vectors_ref = rotation_vectors
    assert_results_close(
        get_rotation_vector_series(rotation_vectors), rotation_vectors_ref
    )

    # Standard case, two similar rotations, without jump over 2pi
    rotation_vectors = [director, director_2]
    rotation_vectors_ref = rotation_vectors
    assert_results_close(
        get_rotation_vector_series(rotation_vectors), rotation_vectors_ref
    )

    # Standard case, two similar rotations, with jump over 2pi
    rotation_vectors = [
        director,
        (2 * np.pi + np.linalg.norm(director_2))
        * director_2
        / np.linalg.norm(director_2),
    ]
    rotation_vectors_ref = [director, director_2]
    assert_results_close(
        get_rotation_vector_series(rotation_vectors), rotation_vectors_ref
    )

    # Standard case, two similar rotations, with jump over 2pi, using rotation
    # objects.
    # With the rotation objects, there is no jump over 2pi, ut in this case we
    # expect the same result as for the previous case.
    # In theory it can be, that the rotation object leads to a different
    # "initial" rotation vector and then also a different series.
    rotation_vectors = [
        Rotation.from_rotation_vector(item) for item in rotation_vectors
    ]
    assert_results_close(
        get_rotation_vector_series(rotation_vectors), rotation_vectors_ref
    )
