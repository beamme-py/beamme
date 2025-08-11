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
"""This module defines the class that is used to create an input file for
Abaqus."""

from enum import Enum as _Enum
from enum import auto as _auto

import numpy as _np

from beamme.core.conf import INPUT_FILE_HEADER as _INPUT_FILE_HEADER
from beamme.core.conf import bme as _bme
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_utils import (
    get_coupled_nodes_to_master_map as _get_coupled_nodes_to_master_map,
)
from beamme.core.rotation import smallest_rotation as _smallest_rotation

# Format template for different number types.
F_INT = "{:6d}"
F_FLOAT = "{: .14e}"


def set_i_global(data_list, *, start_index=0):
    """Set i_global in every item of data_list.

    Args
    ----
    data_list:
        List containing the items that should be numbered
    start_index: int
        Starting index of the numbering
    """

    # A check is performed that every entry in data_list is unique.
    if len(data_list) != len(set(data_list)):
        raise ValueError("Elements in data_list are not unique!")

    # Set the values for i_global.
    for i, item in enumerate(data_list):
        item.i_global = i + start_index


def get_set_lines(set_type, items, name):
    """Get the Abaqus input file lines for a set of items (max 16 items per
    row)"""
    max_entries_per_line = 16
    lines = ["*{}, {}={}".format(set_type, set_type.lower(), name)]
    set_ids = [item.i_global + 1 for item in items]
    set_ids.sort()
    set_ids = [
        set_ids[i : i + max_entries_per_line]
        for i in range(0, len(set_ids), max_entries_per_line)
    ]
    for ids in set_ids:
        lines.append(", ".join([F_INT.format(id) for id in ids]))
    return lines


class AbaqusBeamNormalDefinition(_Enum):
    """Enum for different ways to define the beam cross-section normal.

    For more information see the Abaqus documentation on: "Beam element cross-section orientation"
    and the function `AbaqusInputFile.calculate_cross_section_normal_data`.
    """

    normal_and_extra_node = _auto()
    """Create an extra node and the nodal normal information for each node."""

    normal = _auto()
    """Create the nodal normal information for each node."""


class AbaqusInputFile(object):
    """This class represents an Abaqus input file."""

    def __init__(self, mesh: _Mesh):
        """Initialize the input file.

        Args
        ----
        mesh: Mesh()
            Mesh to be used in this input file.
        """
        self.mesh = mesh

    def write_input_file(
        self,
        file_path,
        *,
        normal_definition=AbaqusBeamNormalDefinition.normal_and_extra_node,
    ):
        """Write the ASCII input file to disk.

        Args
        ----
        file_path: path
            Path on the disk, where the input file should be stored.
        normal_definition: AbaqusBeamNormalDefinition
            How the beam cross-section should be defined.
        """

        # Write the input file to disk
        with open(file_path, "w") as input_file:
            input_file.write(self.get_input_file_string(normal_definition))
            input_file.write("\n")

    def get_input_file_string(self, normal_definition):
        """Generate the string for the Abaqus input file."""

        # Perform some checks on the mesh.
        if _bme.check_overlapping_elements:
            self.mesh.check_overlapping_elements()

        # Assign global indices to all materials
        set_i_global(self.mesh.materials)

        # Calculate the required cross-section normal data
        self.calculate_cross_section_normal_data(normal_definition)

        # Add the lines to the input file
        input_file_lines = []
        input_file_lines.extend(["** " + line for line in _INPUT_FILE_HEADER])
        input_file_lines.extend(self.get_nodes_lines())
        input_file_lines.extend(self.get_element_lines())
        input_file_lines.extend(self.get_material_lines())
        input_file_lines.extend(self.get_set_lines())
        return "\n".join(input_file_lines)

    def calculate_cross_section_normal_data(self, normal_definition):
        """Evaluate all data that is required to fully specify the cross-
        section orientation in Abaqus. The evaluated data is stored in the
        elements.

        For more information see the Abaqus documentation on: "Beam element cross-section orientation"

        Args
        ----
        normal_definition: AbaqusBeamNormalDefinition
            How the beam cross-section should be defined.
        """

        def normalize(vector):
            """Normalize a vector."""
            return vector / _np.linalg.norm(vector)

        # Reset possibly existing data stored in the elements
        # element.n1_orientation_node: list(float)
        #     The coordinates of an additional (dummy) node connected to the
        #     element to define its approximate n1 direction. It this is None,
        #     no additional node will be added to the input file.
        # element.n1_node_id: str
        #     The global ID in the input file for the additional orientation
        #     node.
        # element.n2: list(list(float)):
        #     A list containing possible explicit normal definitions for each
        #     element node. All entries that are not None will be added to the
        #     *NORMAL section of the input file.

        for element in self.mesh.elements:
            element.n1_position = None
            element.n1_node_id = None
            element.n2 = [None for i_node in range(len(element.nodes))]

        if (
            normal_definition == AbaqusBeamNormalDefinition.normal
            or normal_definition == AbaqusBeamNormalDefinition.normal_and_extra_node
        ):
            # In this case we take the beam tangent from the first to the second node
            # and calculate an ortho-normal triad based on this direction. We do this
            # via a smallest rotation mapping from the triad of the first node onto
            # the tangent.

            for element in self.mesh.elements:
                node_1 = element.nodes[0].coordinates
                node_2 = element.nodes[1].coordinates
                t = normalize(node_2 - node_1)

                rotation = element.nodes[0].rotation
                cross_section_rotation = _smallest_rotation(rotation, t)

                if (
                    normal_definition
                    == AbaqusBeamNormalDefinition.normal_and_extra_node
                ):
                    element.n1_position = node_1 + cross_section_rotation * [
                        0.0,
                        1.0,
                        0.0,
                    ]
                element.n2[0] = cross_section_rotation * [0.0, 0.0, 1.0]
        else:
            raise ValueError(f"Got unexpected normal_definition {normal_definition}")

    def get_nodes_lines(self):
        """Get the lines for the input file that represent the nodes."""

        # The nodes require postprocessing, as we have to identify coupled nodes in Abaqus.
        # Internally in Abaqus, coupled nodes are a single node with different normals for the
        # connected element. Therefore, for nodes which are coupled to each other, we keep the
        # same global ID while still keeping the individual nodes.
        _, unique_nodes = _get_coupled_nodes_to_master_map(
            self.mesh, assign_i_global=True
        )

        # Number the remaining nodes and create nodes for the input file
        input_file_lines = ["*Node"]
        for node in unique_nodes:
            input_file_lines.append(
                (", ".join([F_INT] + 3 * [F_FLOAT])).format(
                    node.i_global + 1, *node.coordinates
                )
            )

        # Check if we need to write additional nodes for the element cross-section directions
        node_counter = len(unique_nodes)
        for element in self.mesh.elements:
            if element.n1_position is not None:
                node_counter += 1
                input_file_lines.append(
                    (", ".join([F_INT] + 3 * [F_FLOAT])).format(
                        node_counter, *element.n1_position
                    )
                )
                element.n1_node_id = node_counter

        return input_file_lines

    def get_element_lines(self):
        """Get the lines for the input file that represent the elements."""

        # Sort the elements after their types.
        element_types = {}
        for element in self.mesh.elements:
            element_type = element.beam_type
            if element_type in element_types.keys():
                element_types[element_type].append(element)
            else:
                element_types[element_type] = [element]

        # Write the element connectivity.
        element_count = 0
        element_lines = []
        normal_lines = ["*Normal, type=element"]
        for element_type, elements in element_types.items():
            # Number the elements of this type
            set_i_global(elements, start_index=element_count)

            # Set the element connectivity, possibly including the n1 direction node
            element_lines.append("*Element, type={}".format(element_type))
            for element in elements:
                node_ids = [node.i_global + 1 for node in element.nodes]
                if element.n1_node_id is not None:
                    node_ids.append(element.n1_node_id)
                line_ids = [element.i_global + 1] + node_ids
                element_lines.append(", ".join(F_INT.format(i) for i in line_ids))

                # Set explicit normal definitions for the nodes
                for i_node, n2 in enumerate(element.n2):
                    if n2 is not None:
                        node = element.nodes[i_node]
                        normal_lines.append(
                            (", ".join(2 * [F_INT] + 3 * [F_FLOAT])).format(
                                element.i_global + 1, node.i_global + 1, *n2
                            )
                        )

            element_count += len(elements)

        if len(normal_lines) > 1:
            return element_lines + normal_lines
        else:
            return element_lines

    def get_material_lines(self):
        """Get the lines for the input file that represent the element sets
        with the same material."""

        materials = {}
        for element in self.mesh.elements:
            element_material = element.material
            if element_material in materials.keys():
                materials[element_material].append(element)
            else:
                materials[element_material] = [element]

        # Create the element sets for the different materials.
        input_file_lines = []
        for material, elements in materials.items():
            material_name = material.dump_to_list()[0]
            input_file_lines.extend(get_set_lines("Elset", elements, material_name))
        return input_file_lines

    def get_set_lines(self):
        """Add lines to the input file that represent node and element sets."""

        input_file_lines = []
        for point_set in self.mesh.geometry_sets[_bme.geo.point]:
            if point_set.name is None:
                raise ValueError("Sets added to the mesh have to have a valid name!")
            input_file_lines.extend(
                get_set_lines("Nset", point_set.get_points(), point_set.name)
            )
        for line_set in self.mesh.geometry_sets[_bme.geo.line]:
            if line_set.name is None:
                raise ValueError("Sets added to the mesh have to have a valid name!")
            if isinstance(line_set, _GeometrySet):
                input_file_lines.extend(
                    get_set_lines(
                        "Elset", line_set.geometry_objects[_bme.geo.line], line_set.name
                    )
                )
            else:
                raise ValueError(
                    "Line sets can only be exported to Abaqus if they are defined with the beam elements"
                )
        return input_file_lines
