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
"""This file has functions to create a beam from a parametric curve."""

from typing import Callable as _Callable
from typing import Type as _Type

import numpy as _np
import scipy.integrate as _integrate
from autograd import jacobian as _jacobian
from scipy.interpolate import interp1d as _interp1d

from beamme.core.element_beam import Beam as _Beam
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.material import MaterialBeamBase as _MaterialBeamBase
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.rotation import Rotation as _Rotation
from beamme.core.rotation import smallest_rotation as _smallest_rotation
from beamme.mesh_creation_functions.beam_generic import (
    create_beam_mesh_generic as _create_beam_mesh_generic,
)


class _ArcLengthEvaluation:
    """Class to allow evaluation of the arc length S(t) and the inverse mapping
    t(S).

    This class uses precomputed samples to interpolate between arc
    length values. This is much more efficient than root finding
    algorithms and should provide a suitable accuracy.
    """

    def __init__(
        self,
        ds: _Callable,
        interval: tuple[float, float],
        n_intervals: int = 100,
        method: str = "arc-length",
    ) -> None:
        """Initialize the arc length evaluator and precomputed samples.

        Args:
            ds:
                Function that returns the increment along the curve for a given
                parameter value t.
            interval:
                Start and end values for the parameter coordinate of the curve,
                must be in ascending order.
            n_intervals:
                Number of intervals to use for the precomputation. More intervals
                lead to higher accuracy.
            method:
                Method to use for evaluation of nodal positions along the curve:
                - "arc-length": Uniform spacing along the arc-length of the curve. This means
                    that elements along a curve will have equal length in space.
                - "parametric": Uniform spacing along the parameter coordinate of the curve.
                    This means that elements along a curve will have equal length in parameter
                    space, but not necessarily in physical space.
                - "parametric_consistent_middle_nodes": Same as `parametric`, but element
                    middle nodes are adjusted to be consistent with the arc-length mapping.
                    This means that elements along a curve will have equal length in parameter
                    space, but the middle nodes are placed such that the elements themselves are
                    not distorted.
        """

        self.ds = ds
        self.interval = interval
        self.n_intervals = n_intervals
        self.method = method

        self._compute_samples()
        self._compute_interpolation_functions()

    def _compute_samples(self) -> None:
        """Compute the samples for the arc length mapping.

        This function computes the arc length S(t) at a set of sample
        points along the parameter coordinate t with the accumulative
        Simpson integration.
        """

        # Uniform grid in t for sampling.
        self.t_grid = _np.linspace(self.interval[0], self.interval[1], self.n_intervals)

        # Evaluate ds at all grid points.
        # TODO: This could be vectorized for better performance.
        ds_at_t_samples = _np.empty_like(self.t_grid)
        for i, t in enumerate(self.t_grid):
            ds_at_t_samples[i] = self.ds(t)

        self.S_grid = _integrate.cumulative_simpson(
            y=ds_at_t_samples, x=self.t_grid, initial=0.0
        )

    def _compute_interpolation_functions(self) -> None:
        """Setup the interpolation functions for S(t) and t(S)."""
        self.S_from_t = _interp1d(
            self.t_grid,
            self.S_grid,
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        self.t_from_S = _interp1d(
            self.S_grid,
            self.t_grid,
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )

    def approximate_total_arc_length(self) -> float:
        """Approximate the total arc length along the curve.

        This value is only needed to choose the number of elements along
        the curve.
        """
        return self.S_grid[-1]

    def get_total_arc_length(self) -> float:
        """Get the total arc length along the curve.

        This function might return a different arc-length than `approximate_total_arc_length`,
        if the integral is adaptively refined in `evaluate_all`.
        """
        return self.S_grid[-1]

    def evaluate_all(
        self, evaluation_points: _np.ndarray, middle_node_flags: _np.ndarray
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """Evaluate the parameter coordinates corresponding to each node.

        Args:
            evaluation_points:
                Evaluation points in the interval [0, 1]. Depending on the integration method,
                this can be in normalized parameter space or arc-length space.
            middle_node_flags:
                Boolean array indicating which of the evaluation points
                are middle nodes (True) and which are nodal points (False).

        Returns:
            t_evaluate:
                Parameter coordinates along the curve for each evaluation point.
            S_evaluate:
                Arc-length coordinates along the curve for each evaluation point.
        """

        # Todo: Check if it makes sense to adaptively refine the arc-length
        # integration here.

        if self.method == "arc-length":
            S_evaluate = evaluation_points * self.get_total_arc_length()
            t_evaluate = self.t_from_S(S_evaluate)

        elif self.method == "parametric":
            interval_length = self.interval[1] - self.interval[0]
            t_evaluate = self.interval[0] + evaluation_points * interval_length
            S_evaluate = self.S_from_t(t_evaluate)

        elif self.method == "parametric_consistent_middle_nodes":
            interval_length = self.interval[1] - self.interval[0]
            is_nodal_point = ~middle_node_flags
            nodal_evaluation_points = evaluation_points[is_nodal_point]

            n_el = len(nodal_evaluation_points) - 1
            middle_nodes = int(((len(evaluation_points) - 1) / n_el) - 1)

            t_evaluate = _np.zeros_like(evaluation_points)
            S_evaluate = _np.zeros_like(evaluation_points)

            # Evaluate the nodal points based on direct parametric mapping.
            t_evaluate[is_nodal_point] = (
                self.interval[0] + nodal_evaluation_points * interval_length
            )
            S_evaluate[is_nodal_point] = self.S_from_t(t_evaluate[is_nodal_point])

            # For the middle nodes, do an interpolation in arc-length space.
            for i_interval in range(n_el):
                eval_a = nodal_evaluation_points[i_interval]
                eval_b = nodal_evaluation_points[i_interval + 1]

                S_a = S_evaluate[i_interval * (middle_nodes + 1)]
                S_b = S_evaluate[(i_interval + 1) * (middle_nodes + 1)]

                for i_middle in range(middle_nodes):
                    index_node = i_interval * (middle_nodes + 1) + i_middle + 1
                    eval_middle_node = evaluation_points[index_node]
                    factor = (eval_middle_node - eval_a) / (eval_b - eval_a)
                    S = S_a + factor * (S_b - S_a)
                    t_evaluate[index_node] = self.t_from_S(S)
                    S_evaluate[index_node] = S

        else:
            raise ValueError(f"Unknown method {self.method} for arc-length evaluation!")

        return (t_evaluate, S_evaluate)


def create_beam_mesh_parametric_curve(
    mesh: _Mesh,
    beam_class: _Type[_Beam],
    material: _MaterialBeamBase,
    function: _Callable,
    interval: tuple[float, float],
    *,
    output_length: bool | None = False,
    function_derivative: _Callable | None = None,
    function_rotation: _Callable | None = None,
    arc_length_integrator_kwargs: dict | None = None,
    **kwargs,
) -> _GeometryName | tuple[_GeometryName, float]:
    """Generate a beam from a parametric curve. Integration along the beam is
    performed with scipy, and if the gradient is not explicitly provided, it is
    calculated with the numpy wrapper autograd.

    Args
    ----
    mesh: Mesh
        Mesh that the curve will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    function: function
        3D-parametric curve that represents the beam axis. If only a 2D
        point is returned, the triad creation is simplified. If
        mathematical functions are used, they have to come from the wrapper
        autograd.numpy.
    interval: [start end]
        Start and end values for the parameter of the curve, must be in ascending
        order.
    output_length: bool
        If this is true, the function returns a tuple containing the created
        sets and the total arc length along the integrated function.
    function_derivative: function -> R3
        Explicitly provide the jacobian of the centerline position.
    function_rotation: function -> Rotation
        If this argument is given, the triads are computed with this
        function, on the same interval as the position function. Must
        return a Rotation object.
        If no function_rotation is given, the rotation of the first node
        is calculated automatically and all subsequent nodal rotations
        are calculated based on a smallest rotation mapping onto the curve
        tangent vector.
    arc_length_integrator_kwargs:
        Additional arguments for the arc-length integrator.

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
        Set with the 'start' and 'end' node of the curve. Also a 'line' set
        with all nodes of the curve.
    """

    # Set default values for optional arguments.
    if arc_length_integrator_kwargs is None:
        arc_length_integrator_kwargs = {}

    # To avoid issues with automatic differentiation, we need to ensure that the interval
    # values are of type float.
    interval_array = _np.asarray(interval, dtype=float)

    # Validate interval shape, length and order.
    if interval_array.ndim != 1:
        raise ValueError(
            f"Interval must be a 1D sequence of exactly two values, got array with shape {interval_array.shape}."
        )
    if interval_array.size != 2:
        raise ValueError(
            f"Interval must contain exactly two values, got {interval_array.size}."
        )
    if interval_array[0] >= interval_array[1]:
        raise ValueError(f"Interval must be in ascending order, got {interval_array}.")

    # Check size and type of position function
    test_evaluation_of_function = function(interval_array[0])

    if len(test_evaluation_of_function) == 2:
        is_3d_curve = False
    elif len(test_evaluation_of_function) == 3:
        is_3d_curve = True
    else:
        raise ValueError("Function must return either 2d or 3d curve!")

    if not isinstance(test_evaluation_of_function, _np.ndarray):
        raise TypeError(
            "Function return value must be of type np.ndarray, got {}!".format(
                type(test_evaluation_of_function)
            )
        )

    # Get the derivative of the position function and the increment along
    # the curve.
    if function_derivative is None:
        rp = _jacobian(function)
    else:
        rp = function_derivative

    def ds(t: float) -> float:
        """Increment along the curve."""
        return _np.linalg.norm(rp(t))

    # Setup the arc length integration object.
    arc_length_evaluator = _ArcLengthEvaluation(
        ds, interval_array, **arc_length_integrator_kwargs
    )

    class _BeamFunctionGenerator:
        """This class manages the creation the actual beam nodes and
        rotations."""

        def __init__(self):
            """Initialize the object and define a starting triad."""
            if is_3d_curve:
                r_prime = rp(interval_array[0])
                if abs(_np.dot(r_prime, [0, 0, 1])) < abs(_np.dot(r_prime, [0, 1, 0])):
                    t2_temp = [0, 0, 1]
                else:
                    t2_temp = [0, 1, 0]
                self.last_created_triad = _Rotation.from_basis(r_prime, t2_temp)

        def evaluate_positions_and_rotations(
            self, evaluation_positions: _np.ndarray, middle_node_flags: _np.ndarray
        ) -> tuple[_np.ndarray, list[_Rotation], _np.ndarray]:
            """This function evaluates the positions and rotations for given
            node positions within the interval [0,1].

            Args:
                evaluation_positions:
                    Node positions in the interval [0, 1]. Depending on the integration method,
                    this can be in normalized parameter space or arc-length space.
                middle_node_flags:
                    Boolean array indicating which of the node positions
                    are middle nodes (True) and which are inter element nodes (False).

            Returns:
                coordinates:
                    Physical coordinates of all nodes points along the curve.
                rotations:
                    Rotations at all nodes points along the curve.
            """

            # Get the nodal parameter coordinates and the nodal arc-lengths.
            t_evaluate, S_evaluate = arc_length_evaluator.evaluate_all(
                evaluation_positions, middle_node_flags
            )

            # Position at S.
            coordinates = _np.zeros((len(evaluation_positions), 3))
            if is_3d_curve:
                for i, t in enumerate(t_evaluate):
                    coordinates[i, :] = function(t)
            else:
                for i, t in enumerate(t_evaluate):
                    coordinates[i, :2] = function(t)

            # Rotation at S.
            rotations = []
            if function_rotation is not None:
                for t in t_evaluate:
                    rotations.append(function_rotation(t))

            else:
                for t in t_evaluate:
                    r_prime = rp(t)
                    if is_3d_curve:
                        # Create the next triad via the smallest rotation mapping based
                        # on the last triad.
                        rot = _smallest_rotation(self.last_created_triad, r_prime)
                        self.last_created_triad = rot.copy()
                    else:
                        # The rotation simplifies in the 2d case.
                        rot = _Rotation([0, 0, 1], _np.arctan2(r_prime[1], r_prime[0]))
                    rotations.append(rot)

            # Return the needed values for beam creation.
            return (coordinates, rotations, S_evaluate)

    # Create the beam in the mesh
    created_sets = _create_beam_mesh_generic(
        mesh,
        beam_class=beam_class,
        material=material,
        beam_function_evaluate_positions_and_rotations=True,
        beam_function=_BeamFunctionGenerator(),
        interval=(0.0, 1.0),
        interval_length=arc_length_evaluator.approximate_total_arc_length(),
        **kwargs,
    )

    if output_length:
        return (created_sets, arc_length_evaluator.get_total_arc_length())
    else:
        return created_sets
