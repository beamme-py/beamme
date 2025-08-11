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
"""This file defines the base beam element."""

from typing import Any as _Any
from typing import Callable as _Callable
from typing import List as _List
from typing import Optional as _Optional

import numpy as _np
import vtk as _vtk

from beamme.core.conf import bme as _bme
from beamme.core.element import Element as _Element
from beamme.core.node import NodeCosserat as _NodeCosserat
from beamme.core.rotation import Rotation as _Rotation
from beamme.core.vtk_writer import add_point_data_node_sets as _add_point_data_node_sets


class Beam(_Element):
    """A base class for a beam element."""

    # An array that defines the parameter positions of the element nodes,
    # in ascending order.
    nodes_create: _Any = []

    # A list of valid material types for this element.
    valid_materials: _Any = []

    # Coupling strings.
    coupling_fix_string: _Optional[str] = None
    coupling_joint_string: _Optional[str] = None

    def __init__(self, material=None, nodes=None):
        super().__init__(nodes=nodes, material=material)

    def create_beam(
        self,
        beam_function: _Callable,
        *,
        start_node: _Optional[_NodeCosserat] = None,
        end_node: _Optional[_NodeCosserat] = None,
        relative_twist: _Optional[_Rotation] = None,
        set_nodal_arc_length: bool = False,
        nodal_arc_length_offset: _Optional[float] = None,
    ) -> _List[_NodeCosserat]:
        """Create the nodes for this beam element. The function returns a list
        with the created nodes.

        In the case of start_node and end_node, it is checked, that the
        function and the node have the same coordinates and rotations.

        Args:
            beam_function: function(xi)
                Returns the position, rotation and (optionally) arc length of the
                beam along the local coordinate xi. If no arc lengths is provided,
                the returned value should simply be None.
            start_node: Node
                If this argument is given, this is the node of the beam at xi=-1.
            end_node: Node
                If this argument is given, this is the node of the beam at xi=1.
            relative_twist: Rotation
                Apply this relative rotation to all created nodes. This can be used to
                "twist" the created beam to match the rotation of given nodes.
            set_nodal_arc_length:
                Flag if the arc length in the created nodes should be set.
            nodal_arc_length_offset:
                Offset of the stored nodal arc length w.r.t. to the one generated
                by the function.
        """

        if len(self.nodes) > 0:
            raise ValueError("The beam should not have any local nodes yet!")

        def check_node(node, pos, rot, arc_length, name):
            """Check if the given node matches with the position and rotation
            and optionally also the arc length."""

            if _np.linalg.norm(pos - node.coordinates) > _bme.eps_pos:
                raise ValueError(
                    f"{name} position does not match with function! Got {pos} from function but "
                    + f"given node value is {node.coordinates}"
                )
            if not node.rotation == rot:
                raise ValueError(f"{name} rotation does not match with function!")

            if arc_length is not None:
                if _np.abs(node.arc_length - arc_length) > _bme.eps_pos:
                    raise ValueError(
                        f"Arc lengths don't match, got {node.arc_length} and {arc_length}"
                    )

        # Flags if nodes are given
        has_start_node = start_node is not None
        has_end_node = end_node is not None

        # Loop over local nodes.
        arc_length = None
        for i, xi in enumerate(self.nodes_create):
            # Get the position and rotation at xi
            pos, rot, arc_length_from_function = beam_function(xi)
            if relative_twist is not None:
                rot = rot * relative_twist
            if set_nodal_arc_length:
                arc_length = arc_length_from_function + nodal_arc_length_offset

            # Check if the position and rotation match existing nodes
            if i == 0 and has_start_node:
                check_node(start_node, pos, rot, arc_length, "start_node")
                self.nodes = [start_node]
            elif (i == len(self.nodes_create) - 1) and has_end_node:
                check_node(end_node, pos, rot, arc_length, "end_node")

            # Create the node
            if (i > 0 or not has_start_node) and (
                i < len(self.nodes_create) - 1 or not has_end_node
            ):
                is_middle_node = 0 < i < len(self.nodes_create) - 1
                self.nodes.append(
                    _NodeCosserat(
                        pos, rot, is_middle_node=is_middle_node, arc_length=arc_length
                    )
                )

        # Get a list with the created nodes.
        if has_start_node:
            created_nodes = self.nodes[1:]
        else:
            created_nodes = self.nodes.copy()

        # Add the end node to the beam.
        if has_end_node:
            self.nodes.append(end_node)

        # Return the created nodes.
        return created_nodes

    @classmethod
    def get_coupling_dict(cls, coupling_dof_type):
        """Return the dict to couple this beam to another beam."""

        match coupling_dof_type:
            case _bme.coupling_dof.joint:
                if cls.coupling_joint_dict is None:
                    raise ValueError(f"Joint coupling is not implemented for {cls}")
                return cls.coupling_joint_dict
            case _bme.coupling_dof.fix:
                if cls.coupling_fix_dict is None:
                    raise ValueError("Fix coupling is not implemented for {cls}")
                return cls.coupling_fix_dict
            case _:
                raise ValueError(
                    f'Coupling_dof_type "{coupling_dof_type}" is not implemented!'
                )

    def flip(self):
        """Reverse the nodes of this element.

        This is usually used when reflected.
        """
        self.nodes = [self.nodes[-1 - i] for i in range(len(self.nodes))]

    def _check_material(self):
        """Check if the linked material is valid for this type of beam
        element."""
        for material_type in type(self).valid_materials:
            if isinstance(self.material, material_type):
                break
        else:
            raise TypeError(
                f"Beam of type {type(self)} can not have a material of type {type(self.material)}!"
            )

    def get_vtk(
        self,
        vtk_writer_beam,
        vtk_writer_solid,
        *,
        beam_centerline_visualization_segments=1,
        **kwargs,
    ):
        """Add the representation of this element to the VTK writer as a poly
        line.

        Args
        ----
        vtk_writer_beam:
            VTK writer for the beams.
        vtk_writer_solid:
            VTK writer for solid elements, not used in this method.
        beam_centerline_visualization_segments: int
            Number of segments to be used for visualization of beam centerline between successive
            nodes. Default is 1, which means a straight line is drawn between the beam nodes. For
            Values greater than 1, a Hermite interpolation of the centerline is assumed for
            visualization purposes.
        """

        n_nodes = len(self.nodes)
        n_segments = n_nodes - 1
        n_additional_points_per_segment = beam_centerline_visualization_segments - 1
        # Number of points (in addition to the nodes) to be used for output
        n_additional_points = n_segments * n_additional_points_per_segment
        n_points = n_nodes + n_additional_points

        # Dictionary with cell data.
        cell_data = self.vtk_cell_data.copy()
        cell_data["cross_section_radius"] = self.material.radius

        # Dictionary with point data.
        point_data = {}
        point_data["node_value"] = _np.zeros(n_points)
        point_data["base_vector_1"] = _np.zeros((n_points, 3))
        point_data["base_vector_2"] = _np.zeros((n_points, 3))
        point_data["base_vector_3"] = _np.zeros((n_points, 3))

        coordinates = _np.zeros((n_points, 3))
        nodal_rotation_matrices = [
            node.rotation.get_rotation_matrix() for node in self.nodes
        ]

        for i_node, (node, rotation_matrix) in enumerate(
            zip(self.nodes, nodal_rotation_matrices)
        ):
            coordinates[i_node, :] = node.coordinates
            if node.is_middle_node:
                point_data["node_value"][i_node] = 0.5
            else:
                point_data["node_value"][i_node] = 1.0

            point_data["base_vector_1"][i_node] = rotation_matrix[:, 0]
            point_data["base_vector_2"][i_node] = rotation_matrix[:, 1]
            point_data["base_vector_3"][i_node] = rotation_matrix[:, 2]

        # We can output the element partner index, which is a debug quantity to help find elements
        # with matching middle nodes. This is usually an indicator for an issue with the mesh.
        # TODO: Check if we really need this partner index output any more
        element_partner_indices = list(
            [
                node.element_partner_index
                for node in self.nodes
                if node.element_partner_index is not None
            ]
        )
        if len(element_partner_indices) > 1:
            raise ValueError(
                "More than one element partner indices are currently not supported in the output"
                "functionality"
            )
        elif len(element_partner_indices) == 1:
            cell_data["partner_index"] = element_partner_indices[0] + 1

        # Check if we have everything we need to write output or if we need to calculate additional
        # points for a smooth beam visualization.
        if beam_centerline_visualization_segments == 1:
            point_connectivity = _np.arange(n_nodes)
        else:
            # We need the centerline shape function matrices, so calculate them once and use for
            # all segments that we need. Drop the first and last value, since they represent the
            # nodes which we have already added above.
            xi = _np.linspace(-1, 1, beam_centerline_visualization_segments + 1)[1:-1]
            hermite_shape_functions_pos = _np.array(
                [
                    0.25 * (2.0 + xi) * (1.0 - xi) ** 2,
                    0.25 * (2.0 - xi) * (1.0 + xi) ** 2,
                ]
            ).transpose()
            hermite_shape_functions_tan = _np.array(
                [
                    0.125 * (1.0 + xi) * (1.0 - xi) ** 2,
                    -0.125 * (1.0 - xi) * (1.0 + xi) ** 2,
                ]
            ).transpose()

            point_connectivity = _np.zeros(n_points, dtype=int)

            for i_segment in range(n_segments):
                positions = _np.array(
                    [
                        self.nodes[i_node].coordinates
                        for i_node in [i_segment, i_segment + 1]
                    ]
                )
                tangents = _np.array(
                    [
                        nodal_rotation_matrices[i_node][:, 0]
                        for i_node in [i_segment, i_segment + 1]
                    ]
                )
                length_factor = _np.linalg.norm(positions[1] - positions[0])
                interpolated_coordinates = _np.dot(
                    hermite_shape_functions_pos, positions
                ) + length_factor * _np.dot(hermite_shape_functions_tan, tangents)

                index_first_point = (
                    n_nodes + i_segment * n_additional_points_per_segment
                )
                index_last_point = (
                    n_nodes + (i_segment + 1) * n_additional_points_per_segment
                )

                coordinates[index_first_point:index_last_point] = (
                    interpolated_coordinates
                )
                point_connectivity[
                    i_segment * beam_centerline_visualization_segments
                ] = i_segment
                point_connectivity[
                    (i_segment + 1) * beam_centerline_visualization_segments
                ] = i_segment + 1
                point_connectivity[
                    i_segment * beam_centerline_visualization_segments + 1 : (
                        i_segment + 1
                    )
                    * beam_centerline_visualization_segments
                ] = _np.arange(index_first_point, index_last_point)

        # Get the point data sets and add everything to the output file.
        _add_point_data_node_sets(
            point_data, self.nodes, extra_points=n_additional_points
        )
        indices = vtk_writer_beam.add_points(coordinates, point_data=point_data)
        vtk_writer_beam.add_cell(
            _vtk.vtkPolyLine, indices[point_connectivity], cell_data=cell_data
        )


def generate_beam_class(n_nodes: int):
    """Return a class representing a general beam with n_nodes in BeamMe.

    Args:
        n_nodes: Number of equally spaced nodes along the beam centerline.

    Returns:
        A beam object that has n_nodes along the centerline.
    """

    # Define the class variable responsible for creating the nodes.
    nodes_create = _np.linspace(-1, 1, num=n_nodes)

    # Create the beam class which inherits from the base beam class.
    return type(f"Beam{n_nodes}", (Beam,), {"nodes_create": nodes_create})


Beam2 = generate_beam_class(2)
Beam3 = generate_beam_class(3)
Beam4 = generate_beam_class(4)
Beam5 = generate_beam_class(5)
