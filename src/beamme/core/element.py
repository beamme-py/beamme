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
"""This module implements the class that represents one element in the Mesh."""

from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem


class Element(_BaseMeshItem):
    """A base class for an FEM element in the mesh."""

    def __init__(self, nodes=None, material=None, **kwargs):
        super().__init__(**kwargs)

        # List of nodes that are connected to the element.
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes.copy()

        # Material of this element.
        self.material = material

        # VTK cell data for this element.
        self.vtk_cell_data = {}

    def flip(self):
        """Reverse the nodes of this element.

        This is usually used when reflected.
        """
        raise NotImplementedError(
            f"The flip method is not implemented for {self.__class__}"
        )

    def replace_node(self, old_node, new_node):
        """Replace old_node with new_node."""

        # Look for old_node and replace it. If it is not found, throw error.
        for i, node in enumerate(self.nodes):
            if node == old_node:
                self.nodes[i] = new_node
                break
        else:
            raise ValueError(
                "The node that should be replaced is not in the current element"
            )

    def dump_element_specific_section(self, yaml_dict):
        """Add information of this element to specific section (e.g. STRUCTURE
        KNOTVECTORS for NURBS elements)."""

    def get_vtk(self, vtk_writer_beam, vtk_writer_solid, **kwargs):
        """Add representation of this element to the vtk_writers for solid and
        beam."""
        raise NotImplementedError("VTK output has to be implemented in the class!")
