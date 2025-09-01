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
"""This file defines the base volume element."""

import numpy as _np
import vtk as _vtk

from beamme.core.element import Element as _Element
from beamme.core.vtk_writer import add_point_data_node_sets as _add_point_data_node_sets


class VolumeElement(_Element):
    """A base class for a volume element."""

    # This class variables stores the information about the element shape in
    # vtk. And the connectivity to the nodes.
    vtk_cell_type = None
    vtk_topology: list = []

    def __init__(self, nodes=None, data={}, **kwargs):
        super().__init__(nodes=nodes, material=None, **kwargs)
        self.data = data

    def get_vtk(self, vtk_writer_beam, vtk_writer_solid, **kwargs):
        """Add the representation of this element to the VTK writer as a
        quad."""

        # Check that the element has a valid vtk cell type.
        if self.vtk_cell_type is None:
            raise TypeError(f"vtk_cell_type for {type(self)} not set!")

        # Dictionary with cell data.
        cell_data = {}

        # Dictionary with point data.
        point_data = {}

        # Array with nodal coordinates.
        coordinates = _np.zeros([len(self.nodes), 3])
        for i, node in enumerate(self.nodes):
            coordinates[i, :] = node.coordinates

        # Add the node sets connected to this element.
        _add_point_data_node_sets(point_data, self.nodes)

        # Add cell to writer.
        indices = vtk_writer_solid.add_points(coordinates, point_data=point_data)
        vtk_writer_solid.add_cell(
            self.vtk_cell_type, indices[self.vtk_topology], cell_data=cell_data
        )


class VolumeWEDGE6(VolumeElement):
    """A WEDGE6 volume element."""

    vtk_cell_type = _vtk.vtkWedge
    vtk_topology = list(range(6))


class VolumeHEX8(VolumeElement):
    """A HEX8 volume element."""

    vtk_cell_type = _vtk.vtkHexahedron
    vtk_topology = list(range(8))


class VolumeTET4(VolumeElement):
    """A TET4 volume element."""

    vtk_cell_type = _vtk.vtkTetra
    vtk_topology = list(range(4))


class VolumeTET10(VolumeElement):
    """A TET10 volume element."""

    vtk_cell_type = _vtk.vtkQuadraticTetra
    vtk_topology = list(range(10))


class VolumeHEX20(VolumeElement):
    """A HEX20 volume element."""

    vtk_cell_type = _vtk.vtkQuadraticHexahedron
    vtk_topology = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
    ]


class VolumeHEX27(VolumeElement):
    """A HEX27 volume element."""

    vtk_cell_type = _vtk.vtkTriQuadraticHexahedron
    vtk_topology = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
        24,
        22,
        21,
        23,
        20,
        25,
        26,
    ]
