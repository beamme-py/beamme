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
"""This file has functions to generate multiple parallel fibers within a rectangle.

This can for example be used to create fiber reinforced composite plates.
"""

import numpy as _np

from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.mesh_creation_functions.beam_line import (
    create_beam_mesh_line as _create_beam_mesh_line,
)
from beamme.utils.nodes import check_node_by_coordinate as _check_node_by_coordinate


def _intersect_line_with_rectangle(
    length, width, start_line, direction_line, fail_if_no_intersection=True
):
    """Calculate the intersection points between a line and a rectangle.

    Args
    ----
    length: scalar
        Rectangle length in x direction.
    width: scalar
        Rectangle width in y direction.
    start_line: 2d-list
        Point on the line.
    direction_line: 2d-list
        Direction of the line.
    fail_if_no_intersection: bool
        If this is true and no intersections are found, an error will be
        thrown.

    Return
    ----
    (start, end, projection_found)
    start: 2D vector
        Start point of intersected line.
    end: 2D vector
        End point of intersected line.
    projection_found: bool
        True if intersection is valid.
    """
    # Convert the input values to np.arrays.
    start_line = _np.asarray(start_line)
    direction_line = _np.asarray(direction_line)

    # Set definition for the boundary lines of the rectangle. The director is
    # chosen in a way, that the values [0, 1] for the line parameters alpha are
    # valid.
    # boundary_lines = [..., [start, dir], ...]
    boundary_lines = [
        [[0, 0], [length, 0]],
        [[0, 0], [0, width]],
        [[0, width], [length, 0]],
        [[length, 0], [0, width]],
    ]
    # Convert to numpy arrays.
    boundary_lines = [
        [_np.array(item) for item in boundary] for boundary in boundary_lines
    ]

    # Loop over the boundaries.
    alpha_list = []
    for start_boundary, direction_boundary in boundary_lines:
        # Set up the linear system to solve the intersection problem.
        A = _np.transpose(_np.array([direction_line, -direction_boundary]))

        # Check if the system is solvable.
        if _np.abs(_np.linalg.det(A)) > 1e-10:
            alpha = _np.linalg.solve(A, start_boundary - start_line)
            if 0 <= alpha[1] and alpha[1] <= 1:
                alpha_list.append(alpha[0])

    # Check that intersections were found.
    if len(alpha_list) < 2:
        if fail_if_no_intersection:
            raise ValueError("No intersections found!")
        return (None, None, False)

    # Return the start and end point on the line.
    return (
        start_line + min(alpha_list) * direction_line,
        start_line + max(alpha_list) * direction_line,
        True,
    )


def create_fibers_in_rectangle(
    mesh,
    beam_class,
    material,
    length,
    width,
    angle,
    fiber_normal_distance,
    fiber_element_length,
    *,
    reference_point=None,
    fiber_element_length_min=None,
):
    """Create multiple fibers in a rectangle.

    Args
    ----
    mesh: Mesh
        Mesh that the fibers will be added to.
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    length: float
        Length of the rectangle in x direction (starting at x=0)
    width: float
        Width of the rectangle in y direction (starting at y=0)
    angle: float
        Angle of the fibers in degree.
    fiber_normal_distance: float
        Normal distance between the parallel fibers.
    fiber_element_length: float
        Length of a single beam element. In general it will not be possible to
        exactly achieve this length.
    reference_point: [float, float]
        Specify a single point inside the rectangle that one of the fibers will pass through.
        Per default the reference point is in the middle of the rectangle.
    fiber_element_length_min: float
        Minimum fiber length. If a fiber is shorter than this value, it will not be created.
        The default value is half of fiber_element_length.
    """
    if reference_point is None:
        reference_point = 0.5 * _np.array([length, width])
    else:
        if (
            reference_point[0] < 0.0
            or reference_point[0] > length
            or reference_point[1] < 0.0
            or reference_point[1] > width
        ):
            raise ValueError("The reference point has to lie within the rectangle")

    if fiber_element_length_min is None:
        fiber_element_length_min = 0.5 * fiber_element_length
    elif fiber_element_length_min < 0.0:
        raise ValueError("fiber_element_length_min must be positive!")

    # Get the fiber angle in rad.
    fiber_angle = angle * _np.pi / 180.0
    sin = _np.sin(fiber_angle)
    cos = _np.cos(fiber_angle)

    # Direction and normal vector of the fibers.
    fiber_direction = _np.array([cos, sin])
    fiber_normal = _np.array([-sin, cos])

    # Get an upper bound of the number of fibers in this layer.
    diagonal = _np.sqrt(length**2 + width**2)
    fiber_n_max = int(_np.ceil(diagonal / fiber_normal_distance)) + 1

    # Go in both directions from the start point.
    for direction_sign, n_start in [[-1, 1], [1, 0]]:
        # Create a fiber as long as an intersection is found.
        for i_fiber in range(n_start, fiber_n_max):
            # Get the start and end point of the line.
            start, end, projection_found = _intersect_line_with_rectangle(
                length,
                width,
                reference_point
                + fiber_normal * i_fiber * fiber_normal_distance * direction_sign,
                fiber_direction,
                fail_if_no_intersection=False,
            )

            if projection_found:
                # Calculate the length of the line.
                fiber_length = _np.linalg.norm(end - start)

                # Create the beams if the length is not smaller than the fiber
                # distance.
                if fiber_length >= fiber_element_length_min:
                    # Calculate the number of elements in this fiber.
                    fiber_nel = int(_np.round(fiber_length / fiber_element_length))
                    fiber_nel = _np.max([fiber_nel, 1])
                    _create_beam_mesh_line(
                        mesh,
                        beam_class,
                        material,
                        _np.append(start, 0.0),
                        _np.append(end, 0.0),
                        n_el=fiber_nel,
                    )
            else:
                # The current search position is already outside of the rectangle, no need to continue.
                break

    return_set = _GeometryName()
    return_set["north"] = _GeometrySet(
        mesh.get_nodes_by_function(_check_node_by_coordinate, 1, width),
    )
    return_set["east"] = _GeometrySet(
        mesh.get_nodes_by_function(_check_node_by_coordinate, 0, length),
    )
    return_set["south"] = _GeometrySet(
        mesh.get_nodes_by_function(_check_node_by_coordinate, 1, 0)
    )
    return_set["west"] = _GeometrySet(
        mesh.get_nodes_by_function(_check_node_by_coordinate, 0, 0)
    )
    return_set["all"] = _GeometrySet(mesh.elements)
    return return_set
