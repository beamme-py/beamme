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
"""Generic function for beam creation."""

from typing import Any as _Any
from typing import Callable as _Callable
from typing import Type as _Type

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.element_beam import Beam as _Beam
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.material import MaterialBeamBase as _MaterialBeamBase
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.node import NodeCosserat as _NodeCosserat
from beamme.core.rotation import Rotation as _Rotation
from beamme.utils.nodes import get_single_node as _get_single_node


def _get_interval_node_positions_of_elements(
    interval: tuple[float, float],
    n_el: int | None,
    l_el: float | None,
    node_positions_of_elements: list[float] | None,
    interval_length: float | None,
) -> _np.ndarray:
    """Get the node positions within the interval [0,1].

    Args:
        interval:
            Start and end values for interval that will be used to create the
            beam filament.
        n_el:
            Number of equally spaced beam elements along the line. Defaults to 1.
            Mutually exclusive with l_el
        l_el:
            Desired length of beam elements. This requires the option interval_length
            to be set. Mutually exclusive with n_el. Be aware, that this length
            might not be achieved, if the elements are warped after they are
            created.
        node_positions_of_elements:
            A list of normalized positions (within [0,1] and in ascending order)
            that define the boundaries of beam elements along the created curve.
            The given values will be mapped to the actual `interval` given as an
            argument to this function. These values specify where elements start
            and end, additional internal nodes (such as midpoints in higher-order
            elements) may be placed automatically.
        interval_length:
            Approximation of the total length of the interval. Is required when
            the option `l_el` is given.

    Returns:
        Numpy array with the node positions within the interval.
    """

    # Check for mutually exclusive parameters
    n_given_arguments = sum(
        1
        for argument in [n_el, l_el, node_positions_of_elements]
        if argument is not None
    )
    if n_given_arguments == 0:
        # No arguments were given, use a single element per default
        n_el = 1
    elif n_given_arguments > 1:
        raise ValueError(
            'The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive'
        )

    # Cases where we have equally spaced elements
    if n_el is not None or l_el is not None:
        if l_el is not None:
            # Calculate the number of elements in case a desired element length is provided
            if interval_length is None:
                raise ValueError(
                    'The parameter "l_el" requires "interval_length" to be set.'
                )
            n_el = max([1, round(interval_length / l_el)])
        elif n_el is None:
            raise ValueError("n_el should not be None at this point")

        node_positions_of_elements = [i_node / n_el for i_node in range(n_el + 1)]
    # A list for the element node positions was provided
    elif node_positions_of_elements is not None:
        # Check that the given positions are in ascending order and start with 0 and end with 1
        for index, value, name in zip([0, -1], [0, 1], ["First", "Last"]):
            if not _np.isclose(
                value,
                node_positions_of_elements[index],
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{name} entry of node_positions_of_elements must be {value}, got {node_positions_of_elements[index]}"
                )
        if not all(
            x < y
            for x, y in zip(node_positions_of_elements, node_positions_of_elements[1:])
        ):
            raise ValueError(
                f"The given node_positions_of_elements must be in ascending order. Got {node_positions_of_elements}"
            )
    else:
        raise ValueError(
            'One of the parameters "n_el", "l_el" or "node_positions_of_elements" has to be provided.'
        )

    return interval[0] + (interval[1] - interval[0]) * _np.asarray(
        node_positions_of_elements
    )


def _get_interval_nodal_positions(
    interval_node_positions_of_elements: _np.ndarray, nodes_create: list[float]
) -> tuple[_np.ndarray, _np.ndarray]:
    """Return the nodal positions along the interval depending on the element
    formulation.

    Args:
        interval_node_positions_of_elements:
            Numpy array with the element boundary node positions within the interval of the filament
        nodes_create:
            List with the FE parameter coordinates (in the interval [-1,1]) of the
            element nodes.

    Returns:
        evaluation_positions:
            Numpy array with the interval positions of all nodes along the beam filament (ordered).
        middle_node_flags:
            Numpy array with flags that indicate if a node is an element internal node.
    """

    if (
        _np.abs(nodes_create[0] + 1.0) > _bme.eps_parameter_space
        or _np.abs(nodes_create[-1] - 1.0) > _bme.eps_parameter_space
    ):
        raise ValueError(
            "The first and last entry of nodes_create must be -1 and 1, respectively."
        )

    middle_node_coordinates = nodes_create[1:-1]
    n_middle_nodes = len(middle_node_coordinates)
    n_el = len(interval_node_positions_of_elements) - 1
    n_nodes = n_el * n_middle_nodes + (n_el + 1)

    evaluation_positions = _np.zeros(n_nodes)
    evaluation_positions[:: n_middle_nodes + 1] = interval_node_positions_of_elements

    interval_start_positions = interval_node_positions_of_elements[:-1]
    interval_end_positions = interval_node_positions_of_elements[1:]
    interval_length = interval_end_positions - interval_start_positions

    for i in range(n_middle_nodes):
        nodes_create_position = 0.5 * (middle_node_coordinates[i] + 1.0)
        evaluation_positions[i + 1 :: n_middle_nodes + 1] = (
            interval_start_positions + nodes_create_position * interval_length
        )

    middle_node_flags = _np.ones(n_nodes, dtype=bool)
    middle_node_flags[:: n_middle_nodes + 1] = False

    return evaluation_positions, middle_node_flags


def _evaluate_positions_and_rotations(
    beam_function: _Callable[[float], tuple[_np.ndarray, _Rotation, float | None]],
    evaluation_positions: _np.ndarray,
) -> tuple[_np.ndarray, list[_Rotation], _np.ndarray]:
    """Evaluate positions, rotations and arc lengths along the filament, also
    return a flag indicating middle nodes.

    Args:
        beam_function:
            The `beam_function` has to take one variable s (from `evaluation_positions`)
            and return the position, rotation and arc-length along the beam.
        evaluation_positions:
            Numpy array with the node positions within the interval of the filament.

    Returns:
        coordinates:
            Numpy array with the coordinates of all nodes along the beam.
        rotations:
            List with the rotations of all nodes along the beam.
        arc_lengths:
            Numpy array with the arc lengths of all nodes along the beam.
    """

    n_nodes = len(evaluation_positions)
    coordinates = _np.zeros((n_nodes, 3))
    rotations: list[_Rotation] = []
    arc_lengths = _np.zeros(n_nodes)

    for i_node, evaluation_position in enumerate(evaluation_positions):
        position, rotation, arc_length = beam_function(evaluation_position)
        coordinates[i_node, :] = position
        rotations.append(rotation)
        arc_lengths[i_node] = arc_length

    return coordinates, rotations, arc_lengths


def _check_given_node_and_return_relative_twist(
    mesh: _Mesh,
    node: _NodeCosserat,
    position_from_function: _np.ndarray,
    rotation_from_function: _Rotation,
    name: str,
) -> _Rotation | None:
    """Perform some checks for given nodes and return relative twist if
    necessary.

    If the rotations do not match, check if the first basis vector of the triads is the same. If that is the case, a simple relative twist can be applied to ensure that the triad field is continuous. This relative twist can lead to issues if the beam cross-section is not double symmetric.

    Args:
        mesh: Mesh in which to check if the given nodes already exist.
        node: Given node that should be used at the start or end of the beam.
        position_from_function: Position at the start or end of the beam as given
            by the beam function.
        rotation_from_function: Rotation at the start or end of the beam as given
            by the beam function.
        name: Name of the node ("start" or "end") for better error messages.

    Returns:
        relative_twist:
            If the rotation of the given node does not match with the one from the
            function, but the tangent is the same, the relative twist that has to
            be applied to the rotation field is returned. If no relative twist is
            necessary, None is returned.
    """

    if node not in mesh.nodes:
        raise ValueError("The given node is not in the current mesh")

    if _np.linalg.norm(position_from_function - node.coordinates) > _bme.eps_pos:
        raise ValueError(
            f"The position of the given {name} node does not match with the position from the function!"
        )

    if rotation_from_function == node.rotation:
        return None
    elif not _bme.allow_beam_rotation:
        raise ValueError(
            f"The rotation of the given {name} node does not match with the rotation from the function!"
        )
    else:
        # Evaluate the relative rotation
        # First check if the first basis vector is the same
        relative_basis_1 = node.rotation.inv() * rotation_from_function * [1, 0, 0]
        if _np.linalg.norm(relative_basis_1 - [1, 0, 0]) < _bme.eps_quaternion:
            # Calculate the relative rotation
            return rotation_from_function.inv() * node.rotation
        else:
            raise ValueError(
                f"The tangent of the {name} node does not match with the given function!"
            )


def create_beam_mesh_generic(
    mesh: _Mesh,
    *,
    beam_class: _Type[_Beam],
    material: _MaterialBeamBase,
    beam_function: _Any,
    interval: tuple[float, float],
    beam_function_evaluate_positions_and_rotations: bool = False,
    n_el: int | None = None,
    l_el: float | None = None,
    node_positions_of_elements: list[float] | None = None,
    interval_length: float | None = None,
    set_nodal_arc_length: bool = False,
    nodal_arc_length_offset: float | None = None,
    start_node: _NodeCosserat | _GeometrySet | None = None,
    end_node: _NodeCosserat | _GeometrySet | None = None,
    close_beam: bool = False,
    vtk_cell_data: dict[str, tuple] | None = None,
) -> _GeometryName:
    """Generic beam creation function.

    Remark for given start and/or end nodes:
        If the rotation does not match, but the tangent vector is the same,
        the created beams triads are rotated so the physical problem stays
        the same (for axi-symmetric beam cross-sections) but the nodes can
        be reused.

    Args:
        mesh:
            Mesh that the created beam(s) should be added to.
        beam_class:
            Class of beam that will be used for this line.
        material:
            Material for this line.
        beam_function:
            The beam_function has to return the position along the beam centerline
            for any point in the given `interval`.

            Usually, the Jacobian of the returned position field should be a unit
            vector. Otherwise, the nodes may be spaced in an undesired way. All
            standard mesh creation functions fulfill this property.
        interval:
            Start and end values for interval that will be used to create the
            beam.
        beam_function_evaluate_positions_and_rotations:
            Flag to indicate if the beam_function already provides an efficient
            evaluation of all positions and rotations (and arc lengths) at once.
            If this is True, the beam_function has to provide a method
            `evaluate_positions_and_rotations(evaluation_positions, middle_node_flags)`
            that returns the positions, rotations and arc lengths for all given
            evaluation positions at once. This can speed up the creation of beams
            significantly, especially for complex beam functions.
        n_el:
            Number of equally spaced beam elements along the line. Defaults to 1.
            Mutually exclusive with l_el
        l_el:
            Desired length of beam elements. This requires the option `interval_length`
            to be set. Mutually exclusive with n_el. Be aware, that this length
            might not be achieved, if the elements are warped after they are
            created.
        node_positions_of_elements:
            A list of normalized positions (within [0,1] and in ascending order)
            that define the boundaries of beam elements along the created curve.
            The given values will be mapped to the actual `interval` given as an
            argument to this function. These values specify where elements start
            and end, additional internal nodes (such as midpoints in higher-order
            elements) may be placed automatically.
        interval_length:
            Approximation of the total length of the interval. Is required when
            the option `l_el` is given.
        set_nodal_arc_length:
            Flag if the arc length along the beam filament is set in the created
            nodes. It is ensured that the arc length is consistent with possible
            given start/end nodes.
        nodal_arc_length_offset:
            Offset of the stored nodal arc length w.r.t. to the one generated by
            the function. Defaults to 0, the arc length set in the start node, or
            the arc length in the end node minus total length (such that the arc
            length at the end node matches).
        start_node:
            Node to use as the first node for this line. Use this if the line
            is connected to other lines (angles have to be the same, otherwise
            connections should be used). If a geometry set is given, it can
            contain one, and one node only.
        end_node:
            If this is a Node or GeometrySet, the last node of the created beam
            is set to that node.
            If it is True the created beam is closed within itself.
        close_beam:
            If it is True the created beam is closed within itself (mutually
            exclusive with end_node).
        vtk_cell_data:
            With this argument, a vtk cell data can be set for the elements
            created within this function. This can be used to check which
            elements are created by which function.

    Returns:
        Geometry sets with the 'start' and 'end' node of the curve. Also a 'line' set
        with all nodes of the curve.
    """

    if close_beam and end_node is not None:
        raise ValueError(
            'The arguments "close_beam" and "end_node" are mutually exclusive'
        )

    if set_nodal_arc_length:
        if close_beam:
            raise ValueError(
                "The flags 'set_nodal_arc_length' and 'close_beam' are mutually exclusive."
            )
    elif nodal_arc_length_offset is not None:
        raise ValueError(
            'Providing the argument "nodal_arc_length_offset" without setting '
            '"set_nodal_arc_length" to True does not make sense.'
        )

    # Get element boundary node positions within the given interval
    interval_node_positions_of_elements = _get_interval_node_positions_of_elements(
        interval, n_el, l_el, node_positions_of_elements, interval_length
    )
    n_el = len(interval_node_positions_of_elements) - 1

    # Get the nodal positions in the interval for all nodes (depending on the element formulation).
    evaluation_positions, middle_node_flags = _get_interval_nodal_positions(
        interval_node_positions_of_elements, beam_class.nodes_create
    )

    # Evaluate the centerline position and the rotation for all beam nodes
    if not beam_function_evaluate_positions_and_rotations:
        coordinates, rotations, arc_lengths = _evaluate_positions_and_rotations(
            beam_function, evaluation_positions
        )
    else:
        coordinates, rotations, arc_lengths = (
            beam_function.evaluate_positions_and_rotations(
                evaluation_positions, middle_node_flags
            )
        )

    # Make sure the material is in the mesh.
    mesh.add_material(material)

    # Inspect given nodes and get relative twists if necessary
    relative_twist_start = None
    if start_node is not None:
        start_node = _get_single_node(start_node)
        relative_twist_start = _check_given_node_and_return_relative_twist(
            mesh, start_node, coordinates[0], rotations[0], "start"
        )

    # If an end node is given, check what behavior is wanted.
    relative_twist_end = None
    if end_node is not None:
        end_node = _get_single_node(end_node)
        relative_twist_end = _check_given_node_and_return_relative_twist(
            mesh, end_node, coordinates[-1], rotations[-1], "end"
        )

    # Check if a relative twist has to be applied
    relative_twist_list = [
        twist
        for twist in [relative_twist_start, relative_twist_end]
        if twist is not None
    ]
    if len(relative_twist_list) == 2:
        if not relative_twist_list[0] == relative_twist_list[1]:
            raise ValueError(
                "The relative twist required for the start and end node do not match"
            )
    if len(relative_twist_list) > 0:
        relative_twist = relative_twist_list[0]
        for i_rot, rotation in enumerate(rotations):
            rotations[i_rot] = rotation * relative_twist

    # Get the start value for the arc length functionality
    if set_nodal_arc_length:
        if nodal_arc_length_offset is not None:
            # Let's use the given value, the later check will detect if this
            # does not match the given nodes.
            pass
        elif start_node is not None and start_node.arc_length is not None:
            nodal_arc_length_offset = start_node.arc_length
        elif end_node is not None and end_node.arc_length is not None:
            nodal_arc_length_offset = end_node.arc_length - arc_lengths[-1]
        else:
            # Default value
            nodal_arc_length_offset = 0.0
        arc_lengths += nodal_arc_length_offset

        if start_node is not None:
            if _np.abs(start_node.arc_length - arc_lengths[0]) > _bme.eps_pos:
                raise ValueError(
                    "The arc length at the start node does not match with "
                    "the calculated one!"
                )
        if end_node is not None:
            if _np.abs(end_node.arc_length - arc_lengths[-1]) > _bme.eps_pos:
                raise ValueError(
                    "The arc length at the end node does not match with "
                    "the calculated one!"
                )
    else:
        arc_lengths = [None] * len(arc_lengths)

    # Create the nodes and add the new ones to the mesh
    nodes = [
        _NodeCosserat(pos, rot, is_middle_node=middle_node_flag, arc_length=arc_length)
        for pos, rot, arc_length, middle_node_flag in zip(
            coordinates, rotations, arc_lengths, middle_node_flags
        )
    ]
    if start_node is not None:
        nodes[0] = start_node
    if close_beam:
        nodes[-1] = nodes[0]
    elif end_node is not None:
        nodes[-1] = end_node
    start_slice = 1 if start_node is not None else None
    end_slice = -1 if end_node is not None or close_beam else None
    mesh.nodes.extend(nodes[start_slice:end_slice])

    # Create the beam elements and assign the nodes
    nodes_per_element = len(beam_class.nodes_create)
    elements: list[_Beam] = []
    for i_el in range(n_el):
        beam = beam_class(
            material=material,
            nodes=nodes[
                i_el * (nodes_per_element - 1) : (i_el + 1) * (nodes_per_element - 1)
                + 1
            ],
        )
        elements.append(beam)

    # Set vtk cell data on created elements.
    if vtk_cell_data is not None:
        for data_name, data_value in vtk_cell_data.items():
            for element in elements:
                if data_name in element.vtk_cell_data.keys():
                    raise KeyError(
                        'The cell data "{}" already exists!'.format(data_name)
                    )
                element.vtk_cell_data[data_name] = data_value

    # Add items to the mesh
    mesh.elements.extend(elements)

    # Set the nodes that are at the beginning and end of line (for search
    # of overlapping points)
    nodes[0].is_end_node = True
    nodes[-1].is_end_node = True

    # Create geometry sets that will be returned.
    return_set = _GeometryName()
    return_set["start"] = _GeometrySet(nodes[0])
    return_set["end"] = _GeometrySet(nodes[-1])
    return_set["line"] = _GeometrySet(elements)

    return return_set
