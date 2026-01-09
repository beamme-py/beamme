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
"""This module defines a class that represents a rotation in 3D."""

import numpy as _np
import quaternion as _quaternion
from numpy.typing import NDArray as _NDArray

from beamme.core.conf import bme as _bme


def skew_matrix(vector):
    """Return the skew matrix for the vector."""
    skew = _np.zeros([3, 3])
    skew[0, 1] = -vector[2]
    skew[0, 2] = vector[1]
    skew[1, 0] = vector[2]
    skew[1, 2] = -vector[0]
    skew[2, 0] = -vector[1]
    skew[2, 1] = vector[0]
    return skew


class Rotation:
    """A class that represents a rotation of a coordinate system.

    Internally the rotations are stored as quaternions.
    """

    def __init__(self, *args):
        """Initialize the rotation object.

        Args
        ----
        *args:
            - Rotation()
                Create a identity rotation.
            - Rotation(axis, phi)
                Create a rotation around the vector axis with the angle phi.
        """

        self.q = _np.zeros(4)

        if len(args) == 0:
            # Identity element.
            self.q[0] = 1
        elif len(args) == 2:
            # Set from rotation axis and rotation angle.
            axis = _np.asarray(args[0])
            phi = args[1]
            norm = _np.linalg.norm(axis)
            if norm < _bme.eps_quaternion:
                raise ValueError("The rotation axis can not be a zero vector!")
            self.q[0] = _np.cos(0.5 * phi)
            self.q[1:] = _np.sin(0.5 * phi) * axis / norm
        else:
            raise ValueError(f"The given arguments {args} are invalid!")

    @classmethod
    def from_quaternion(cls, q, *, normalized=False):
        """Create the object from a quaternion float array (4x1)

        Args
        ----
        q: Quaternion, q0, qx,qy,qz
        normalized: Flag if the input quaternion is normalized. If so, no
            normalization is performed which can potentially improve performance.
            Skipping the normalization should only be done in very special cases
            where we can be sure that the input quaternion is normalized to avoid
            error accumulation.
        """
        if isinstance(q, _quaternion.quaternion):
            q = _quaternion.as_float_array(q)

        rotation = object.__new__(cls)
        if normalized:
            rotation.q = _np.array(q)
        else:
            rotation.q = _np.asarray(q) / _np.linalg.norm(q)
        if (not rotation.q.ndim == 1) or (not len(rotation.q) == 4):
            raise ValueError("Got quaternion array with unexpected dimensions")
        return rotation

    @classmethod
    def from_rotation_matrix(cls, R):
        """Create the object from a rotation matrix.

        The code is based on Spurriers algorithm:
            R. A. Spurrier (1978): “Comment on “Singularity-free extraction of a quaternion from a
            direction-cosine matrix”
        """

        R = _np.asarray(R)
        q = _np.zeros(4)
        trace = _np.trace(R)
        values = [R[i, i] for i in range(3)]
        values.append(trace)
        arg_max = _np.argmax(values)
        if arg_max == 3:
            q[0] = _np.sqrt(trace + 1) * 0.5
            q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
            q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
            q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
        else:
            i_index = arg_max
            j_index = (i_index + 1) % 3
            k_index = (i_index + 2) % 3
            q_i = _np.sqrt(R[i_index, i_index] * 0.5 + (1 - trace) * 0.25)
            q[0] = (R[k_index, j_index] - R[j_index, k_index]) / (4 * q_i)
            q[i_index + 1] = q_i
            q[j_index + 1] = (R[j_index, i_index] + R[i_index, j_index]) / (4 * q_i)
            q[k_index + 1] = (R[k_index, i_index] + R[i_index, k_index]) / (4 * q_i)

        return cls.from_quaternion(q)

    @classmethod
    def from_basis(cls, t1, t2):
        """Create the object from two basis vectors t1, t2.

        t2 will be orthogonalized on t1, and t3 will be calculated with
        the cross product.
        """

        t1_norm = _np.linalg.norm(t1)
        if t1_norm < _bme.eps_quaternion:
            raise ValueError(f"The given vector t1 can not be a zero vector, got {t1}.")
        t1_normal = t1 / t1_norm

        t2_ortho = t2 - t1_normal * _np.dot(t1_normal, t2)
        t2_ortho_norm = _np.linalg.norm(t2_ortho)
        if t2_ortho_norm < _bme.eps_quaternion:
            raise ValueError(
                f"Got two vectors {t1} and {t2} that are not linear independent."
            )
        t2_normal = t2_ortho / t2_ortho_norm
        t3_normal = _np.cross(t1_normal, t2_normal)

        R = _np.transpose([t1_normal, t2_normal, t3_normal])
        return cls.from_rotation_matrix(R)

    @classmethod
    def from_rotation_vector(cls, rotation_vector):
        """Create the object from a rotation vector."""

        q = _np.zeros(4)
        rotation_vector = _np.asarray(rotation_vector)
        phi = _np.linalg.norm(rotation_vector)
        q[0] = _np.cos(0.5 * phi)
        if phi < _bme.eps_quaternion:
            # This is the Taylor series expansion of sin(phi/2)/phi around phi=0
            q[1:] = 0.5 * rotation_vector
        else:
            q[1:] = _np.sin(0.5 * phi) / phi * rotation_vector
        return cls.from_quaternion(q)

    def check(self):
        """Perform all checks for the rotation."""
        self.check_uniqueness()
        self.check_quaternion_constraint()

    def check_uniqueness(self):
        """We always want q0 to be positive -> the range for the rotational
        angle is 0 <= phi <= pi."""

        if self.q[0] < 0:
            self.q *= -1

    def check_quaternion_constraint(self):
        """We want to check that q.q = 1."""

        if _np.abs(1 - _np.linalg.norm(self.q)) > _bme.eps_quaternion:
            raise ValueError(
                f"The rotation object is corrupted. q.q does not equal 1! q={self.q}"
            )

    def get_rotation_matrix(self):
        """Return the rotation matrix for this rotation.

        (Krenk (3.50))
        """
        q_skew = skew_matrix(self.q[1:])
        R = (
            (self.q[0] ** 2 - _np.dot(self.q[1:], self.q[1:])) * _np.eye(3)
            + 2 * self.q[0] * q_skew
            + 2 * ([self.q[1:]] * _np.transpose([self.q[1:]]))
        )

        return R

    def get_quaternion(self):
        """Return the quaternion for this rotation, as numpy array (copy)."""
        return _np.array(self.q)

    def get_numpy_quaternion(self):
        """Return a numpy quaternion object representing this rotation
        (copy)."""
        return _quaternion.from_float_array(_np.array(self.q))

    def get_rotation_vector(self):
        """Return the rotation vector for this object."""

        self.check()

        norm = _np.linalg.norm(self.q[1:])
        phi = 2 * _np.arctan2(norm, self.q[0])

        if phi < _bme.eps_quaternion:
            # For small angles return the Taylor series expansion of phi/sin(phi/2)
            scale_factor = 2
        else:
            scale_factor = phi / _np.sin(phi / 2)
            if _np.abs(_np.abs(phi) - _np.pi) < _bme.eps_quaternion:
                # For rotations of exactly +-pi, numerical issues might occur, resulting in
                # a rotation vector that is non-deterministic. The result is correct, but
                # the sign can switch due to different implementation of basic underlying
                # math functions. This is especially triggered when using this code with
                # different OS. To avoid this, we scale the rotation axis in such a way,
                # for a rotation angle of +-pi, the first component of the rotation axis
                # that is not 0 is positive.
                for i_dir in range(3):
                    if _np.abs(self.q[1 + i_dir]) > _bme.eps_quaternion:
                        if self.q[1 + i_dir] < 0:
                            scale_factor *= -1
                        break
        return self.q[1:] * scale_factor

    def get_transformation_matrix(self):
        """Return the transformation matrix for this rotation.

        The transformation matrix maps the (infinitesimal)
        multiplicative rotational increments onto the additive ones.
        """

        omega = self.get_rotation_vector()
        omega_norm = _np.linalg.norm(omega)

        # We have to take the inverse of the the rotation angle here, therefore,
        # we have a branch for small angles where the singularity is not present.
        if omega_norm**2 > _bme.eps_quaternion:
            # Taken from Jelenic and Crisfield (1999) Equation (2.5)
            omega_dir = omega / omega_norm
            omega_skew = skew_matrix(omega)
            transformation_matrix = (
                _np.outer(omega_dir, omega_dir)
                - 0.5 * omega_skew
                + 0.5
                * omega_norm
                / _np.tan(0.5 * omega_norm)
                * (_np.identity(3) - _np.outer(omega_dir, omega_dir))
            )
        else:
            # This is the constant part of the Taylor series expansion. If this
            # function is used with automatic differentiation, higher order
            # terms have to be added!
            transformation_matrix = _np.identity(3)
        return transformation_matrix

    def get_transformation_matrix_inv(self):
        """Return the inverse of the transformation matrix for this rotation.

        The inverse of the transformation matrix maps the
        (infinitesimal) additive rotational increments onto the
        multiplicative ones.
        """

        omega = self.get_rotation_vector()
        omega_norm = _np.linalg.norm(omega)

        # We have to take the inverse of the the rotation angle here, therefore,
        # we have a branch for small angles where the singularity is not present.
        if omega_norm**2 > _bme.eps_quaternion:
            # Taken from Jelenic and Crisfield (1999) Equation (2.5)
            omega_dir = omega / omega_norm
            omega_skew = skew_matrix(omega)
            transformation_matrix_inverse = (
                (1.0 - _np.sin(omega_norm) / omega_norm)
                * _np.outer(omega_dir, omega_dir)
                + _np.sin(omega_norm) / omega_norm * _np.identity(3)
                + (1.0 - _np.cos(omega_norm)) / omega_norm**2 * omega_skew
            )
        else:
            # This is the constant part of the Taylor series expansion. If this
            # function is used with automatic differentiation, higher order
            # terms have to be added!
            transformation_matrix_inverse = _np.identity(3)
        return transformation_matrix_inverse

    def inv(self):
        """Return the inverse of this rotation."""

        tmp_quaternion = self.q.copy()
        tmp_quaternion[0] *= -1.0
        return Rotation.from_quaternion(tmp_quaternion)

    def __mul__(self, other):
        """Add this rotation to another, or apply it on a vector."""

        # Check if the other object is also a rotation.
        if isinstance(other, Rotation):
            # Get quaternions of the two objects.
            p = self.q
            q = other.q
            # Add the rotations.
            added_rotation = _np.zeros_like(self.q)
            added_rotation[0] = p[0] * q[0] - _np.dot(p[1:], q[1:])
            added_rotation[1:] = p[0] * q[1:] + q[0] * p[1:] + _np.cross(p[1:], q[1:])
            return Rotation.from_quaternion(added_rotation)
        elif isinstance(other, (list, _np.ndarray)) and len(other) == 3:
            # Apply rotation to vector.
            return _np.dot(self.get_rotation_matrix(), _np.asarray(other))
        raise NotImplementedError("Error, not implemented, does not make sense anyway!")

    def __eq__(self, other):
        """Check if the other rotation is equal to this one."""

        if isinstance(other, Rotation):
            return bool(
                (_np.linalg.norm(self.q - other.q) < _bme.eps_quaternion)
                or (_np.linalg.norm(self.q + other.q) < _bme.eps_quaternion)
            )
        else:
            return object.__eq__(self, other)

    def copy(self):
        """Return a copy of this object."""
        return Rotation.from_quaternion(self.q, normalized=True)

    def __str__(self):
        """String representation of object."""

        self.check()
        return f"Rotation:\n    q0: {self.q[0]}\n    q: {self.q[1:]}"


def add_rotations(
    rotation_21: Rotation | _NDArray[_quaternion.quaternion],
    rotation_10: Rotation | _NDArray[_quaternion.quaternion],
) -> _NDArray[_quaternion.quaternion]:
    """Multiply rotations onto another.

    Args:
        rotation_10: The first rotation(s) that are applied.
        rotation_21: The second rotation(s) that are applied.

    Returns:
        An array with the compound quaternions.
    """

    # Transpose the arrays, to work with the following code.
    if isinstance(rotation_10, Rotation):
        rot1 = rotation_10.get_quaternion().transpose()
    else:
        rot1 = _np.transpose(rotation_10)
    if isinstance(rotation_21, Rotation):
        rot2 = rotation_21.get_quaternion().transpose()
    else:
        rot2 = _np.transpose(rotation_21)

    if rot1.size > rot2.size:
        rotnew = _np.zeros_like(rot1)
    else:
        rotnew = _np.zeros_like(rot2)

    # Multiply the two rotations (code is taken from /utility/rotation.nb).
    rotnew[0] = (
        rot1[0] * rot2[0] - rot1[1] * rot2[1] - rot1[2] * rot2[2] - rot1[3] * rot2[3]
    )
    rotnew[1] = (
        rot1[1] * rot2[0] + rot1[0] * rot2[1] + rot1[3] * rot2[2] - rot1[2] * rot2[3]
    )
    rotnew[2] = (
        rot1[2] * rot2[0] - rot1[3] * rot2[1] + rot1[0] * rot2[2] + rot1[1] * rot2[3]
    )
    rotnew[3] = (
        rot1[3] * rot2[0] + rot1[2] * rot2[1] - rot1[1] * rot2[2] + rot1[0] * rot2[3]
    )

    return rotnew.transpose()


def rotate_coordinates(
    coordinates: _NDArray,
    rotation: Rotation | _NDArray[_quaternion.quaternion],
    *,
    origin=None,
):
    """Rotate all given coordinates.

    Args:
        coordinates: Array of 3D coordinates to be rotated
        rotation: The rotation(s) that will be applied to the coordinates. If
            this is an array it has to hold a quaternion for each coordinate.
    origin (3D vector):  If this is given, the mesh is rotated about this
        point. Defaults to (0, 0, 0).
    """

    if isinstance(rotation, Rotation):
        rotation = rotation.get_quaternion().transpose()

    # Check if origin has to be added
    if origin is None:
        origin = [0.0, 0.0, 0.0]

    # New position array
    coordinates_new = _np.zeros_like(coordinates)

    # Evaluate the new positions using the numpy data structure
    # (code is taken from /utility/rotation.nb)
    rotation = rotation.transpose()

    q0_q0 = _np.square(rotation[0])
    q0_q1_2 = 2.0 * rotation[0] * rotation[1]
    q0_q2_2 = 2.0 * rotation[0] * rotation[2]
    q0_q3_2 = 2.0 * rotation[0] * rotation[3]

    q1_q1 = _np.square(rotation[1])
    q1_q2_2 = 2.0 * rotation[1] * rotation[2]
    q1_q3_2 = 2.0 * rotation[1] * rotation[3]

    q2_q2 = _np.square(rotation[2])
    q2_q3_2 = 2.0 * rotation[2] * rotation[3]

    q3_q3 = _np.square(rotation[3])

    coordinates_new[:, 0] = (
        (q0_q0 + q1_q1 - q2_q2 - q3_q3) * (coordinates[:, 0] - origin[0])
        + (q1_q2_2 - q0_q3_2) * (coordinates[:, 1] - origin[1])
        + (q0_q2_2 + q1_q3_2) * (coordinates[:, 2] - origin[2])
    )
    coordinates_new[:, 1] = (
        (q1_q2_2 + q0_q3_2) * (coordinates[:, 0] - origin[0])
        + (q0_q0 - q1_q1 + q2_q2 - q3_q3) * (coordinates[:, 1] - origin[1])
        + (-q0_q1_2 + q2_q3_2) * (coordinates[:, 2] - origin[2])
    )
    coordinates_new[:, 2] = (
        (-q0_q2_2 + q1_q3_2) * (coordinates[:, 0] - origin[0])
        + (q0_q1_2 + q2_q3_2) * (coordinates[:, 1] - origin[1])
        + (q0_q0 - q1_q1 - q2_q2 + q3_q3) * (coordinates[:, 2] - origin[2])
    )

    if origin is not None:
        coordinates_new += origin

    return coordinates_new


def smallest_rotation(q: Rotation, t):
    """Get the triad that results from the smallest rotation (rotation without
    twist) from the triad q such that the rotated first basis vector aligns
    with t. For more details see Christoph Meier's dissertation chapter 2.1.2.

    Args
    ----
    q: Rotation
        Starting triad.
    t: Vector in R3
        Direction of the first basis of the rotated triad.
    Return
    ----
    q_sr: Rotation
        The triad that results from a smallest rotation.
    """

    R_old = q.get_rotation_matrix()
    g1_old = R_old[:, 0]
    g1 = _np.asarray(t) / _np.linalg.norm(t)

    # Quaternion components of relative rotation
    q_rel = _np.zeros(4)

    # The scalar quaternion part is cos(alpha/2) this is equal to
    q_rel[0] = _np.linalg.norm(0.5 * (g1_old + g1))

    # Vector part of the quaternion is sin(alpha/2)*axis
    q_rel[1:] = _np.cross(g1_old, g1) / (2.0 * q_rel[0])

    return Rotation.from_quaternion(q_rel) * q


def get_rotation_vector_series(
    rotation_vectors: list | _NDArray | list[Rotation],
) -> _NDArray:
    """Return an array containing the rotation vectors representing the given
    rotation vectors.

    The main feature of this function is, that the returned rotation
    vectors don't have jumps when the rotation angle exceeds 2*pi. We
    return a "continuous" series of rotation vectors that can be used to
    interpolate between them.

    Note: Interpolating between the returned rotation vectors will in general
    result in a non-objective interpolation.

    Args:
        rotation_vectors: A list or array containing the rotation vectors, has
            to be in order.

    Returns:
        An array containing the "continuous" rotation vectors.
    """

    if isinstance(rotation_vectors, list):
        rotation_vector_array = _np.zeros((len(rotation_vectors), 3))
        for i_rotation, rotation_entry in enumerate(rotation_vectors):
            if isinstance(rotation_entry, Rotation):
                rotation_vector_array[i_rotation] = rotation_entry.get_rotation_vector()
            else:
                rotation_vector_array[i_rotation] = rotation_entry

    def closest_multiple_of_two_pi(x: float) -> float:
        """Given the value x, return the multiple of 2*pi that is closest to
        it."""
        return 2.0 * _np.pi * round(x / (2 * _np.pi))

    rotation_vectors_continuous = _np.zeros((len(rotation_vector_array), 3))
    rotation_vectors_continuous[0, :] = rotation_vector_array[0]

    for i, current_rotation_vector in enumerate(rotation_vector_array[1:]):
        rotation_vector_last = rotation_vector_array[i]
        theta = _np.linalg.norm(current_rotation_vector)

        if theta < _bme.eps_quaternion:
            # The current rotation is an identity rotation.
            theta_last = _np.linalg.norm(rotation_vector_last)
            if theta_last < _bme.eps_quaternion:
                # The last rotation was also an identity rotation, so simply add
                # an empty rotation vector
                rotation_vectors_continuous[i + 1] = [0.0, 0.0, 0.0]
            else:
                # The last rotation was not a zero rotation. In this case we simply
                # take the direction of the last rotation vector and set it to to the
                # closest length which is a multiple of 2*pi.
                rotation_vectors_continuous[i + 1] = (
                    rotation_vector_last
                    / theta_last
                    * closest_multiple_of_two_pi(theta_last)
                )
        else:
            axis = current_rotation_vector / theta
            # This is the offset from the current rotation vector to the previous one
            # in the sense of a multiple of 2*pi.
            multiple_of_two_pi = closest_multiple_of_two_pi(
                (_np.dot(axis, rotation_vector_last) - theta)
            )
            rotation_vectors_continuous[i + 1] = (multiple_of_two_pi + theta) * axis
    return rotation_vectors_continuous
