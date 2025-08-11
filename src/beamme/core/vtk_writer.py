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
"""This module provides a class that is used to write VTK files."""

import numbers as _numbers
import os as _os
import warnings as _warnings
from enum import Enum as _Enum
from enum import auto as _auto

import numpy as _np
import vtk as _vtk

from beamme.core.conf import bme as _bme

# Number of digits for node set output (this will be set in the
# Mesh.get_unique_geometry_sets() method.
VTK_NODE_SET_FORMAT = "{:05}"

# Nan values for vtk data, since we currently can't set nan explicitly.
VTK_NAN_INT = -1
VTK_NAN_FLOAT = 0.0


class VTKGeometry(_Enum):
    """Enum for VTK geometry types (for now cells and points)."""

    point = _auto()
    cell = _auto()


class VTKType(_Enum):
    """Enum for VTK value types."""

    int = _auto()
    float = _auto()


class VTKTensor(_Enum):
    """Enum for VTK tensor types."""

    scalar = _auto()
    vector = _auto()


def add_point_data_node_sets(point_data, nodes, *, extra_points=0):
    """Add the information if a node is part of a set to the point_data vector
    for all nodes in the list 'nodes'.

    The extra_points argument specifies how many additional
    visualization points there are, i.e., points that are not based on
    nodes, but are only used for visualization purposes.
    """

    # Get list with node set indices of the given nodes
    geometry_set_list = []
    for node in nodes:
        geometry_set_list.extend(node.node_sets_link)

    # Remove double entries of list.
    geometry_set_list = list(set(geometry_set_list))

    # Loop over the geometry sets.
    n_nodes = len(nodes)
    for geometry_set in geometry_set_list:
        # Check which nodes are connected to a geometry set.
        data_vector = _np.zeros(n_nodes + extra_points)
        for i, node in enumerate(nodes):
            if geometry_set in node.node_sets_link:
                data_vector[i] = 1
            else:
                data_vector[i] = VTK_NAN_INT
        for i in range(extra_points):
            data_vector[n_nodes + i] = (
                1 if geometry_set.geometry_type is _bme.geo.line else VTK_NAN_INT
            )

        # Get the name of the geometry type.
        if geometry_set.geometry_type is _bme.geo.point:
            geometry_name = "geometry_point"
        elif geometry_set.geometry_type is _bme.geo.line:
            geometry_name = "geometry_line"
        elif geometry_set.geometry_type is _bme.geo.surface:
            geometry_name = "geometry_surface"
        elif geometry_set.geometry_type is _bme.geo.volume:
            geometry_name = "geometry_volume"
        else:
            raise TypeError("The geometry type is wrong!")

        # Add the data vector.
        set_name = f"{geometry_name}_set_{_bme.vtk_node_set_format.format(geometry_set.i_global + 1)}"
        point_data[set_name] = (data_vector, VTKType.int)


def _get_data_value_and_type(data):
    """Return the data and its type if one was given.

    The default type, if none was given is float.
    """
    if isinstance(data, tuple):
        return data[0], data[1]
    else:
        return data, VTKType.float


def _get_vtk_array_type(data):
    """Return the corresponding beamme type."""
    data_type = data.GetDataTypeAsString()
    if data_type == "int":
        return VTKType.int
    elif data_type == "double":
        return VTKType.float
    raise ValueError(f'Got unexpected type "{data_type}"!')


class VTKWriter:
    """A class that manages VTK cells and data and can also create them."""

    def __init__(self):
        # Initialize VTK objects.
        self.points = _vtk.vtkPoints()
        self.points.SetDataTypeToDouble()
        self.grid = _vtk.vtkUnstructuredGrid()

        # Link points to grid.
        self.grid.SetPoints(self.points)

        # Container for output data.
        self.data = {}
        for key1 in VTKGeometry:
            for key2 in VTKTensor:
                self.data[key1, key2] = {}

    def add_points(self, points, *, point_data=None):
        """Add points to the data stored in this object.

        Args
        ----
        points: [3d vector]
            Coordinates of points for this cell.
        point_data: dic
            A dictionary containing data that will be added for the newly added points.
            If a field exists in the global data but not in the one added here, that field
            will be set to bme.vtk_nan for the newly added points.

        Return:
        ----
        indices: [int]
            A list with the global indices of the added points.
        """

        n_points = len(points)

        # Check if point data containers are of the correct size
        if point_data is not None:
            for key, item_value in point_data.items():
                value, _data_type = _get_data_value_and_type(item_value)
                if not len(value) == n_points:
                    raise IndexError(
                        f"The length of coordinates is {n_points},"
                        f"the length of {key} is {len(value)}, does not match!"
                    )

        # Add point data
        self._add_data(point_data, VTKGeometry.point, n_new_items=n_points)

        # Add point coordinates
        n_grid_points = self.points.GetNumberOfPoints()
        for point in points:
            # Add the coordinate to the global list of coordinates.
            self.points.InsertNextPoint(*point)

        return _np.array(
            [n_grid_points + i_point for i_point in range(len(points))], dtype=int
        )

    def add_cell(self, cell_type, topology, *, cell_data=None):
        """Create a cell and add it to the global array.

        Args
        ----
        cell_type: VTK_type
            Type of cell that will be created.
        topology: [int]
            The connectivity between the cell and the global points.
        cell_data: dic
            A dictionary containing data that will be added for the newly added cell.
            If a field exists in the global data but not in the one added here, that field
            will be set to bme.vtk_nan for the newly added cell.
        """

        # Add the data entries.
        self._add_data(cell_data, VTKGeometry.cell)

        # Create the cell.
        geometry_item = cell_type()
        geometry_item.GetPointIds().SetNumberOfIds(len(topology))

        # Set the connectivity
        for i_local, i_global in enumerate(topology):
            geometry_item.GetPointIds().SetId(i_local, i_global)

        # Add to global cells
        self.grid.InsertNextCell(
            geometry_item.GetCellType(), geometry_item.GetPointIds()
        )

    def _add_data(self, data_container, vtk_geom_type, *, n_new_items=1):
        """Add a data container to the existing global data container of this
        object.

        Args
        ----
        data_container: see self.add_cell
        vtk_geom_type: bme.vtk_geo
            Type of data container that is added
        n_new_items: int
            Number of new items added. This is needed to fill up data fields that are in the
            global data but not in the one that is added.
        """

        # Check if data container already exists. If not, add it and also add
        # previous entries.
        if data_container is not None:
            if vtk_geom_type == VTKGeometry.cell:
                n_items = self.grid.GetNumberOfCells()
            else:
                n_items = self.grid.GetNumberOfPoints()

            for key, item_value in data_container.items():
                # Get the data and the value type (int or float).
                value, data_type = _get_data_value_and_type(item_value)

                # Data type.
                if vtk_geom_type == VTKGeometry.cell:
                    vtk_tensor_type = self._get_vtk_data_type(value)
                else:
                    for item in value:
                        vtk_tensor_type = self._get_vtk_data_type(item)

                # Check if key already exists.
                if key not in self.data[vtk_geom_type, vtk_tensor_type].keys():
                    # Set up the VTK data array.
                    if data_type is VTKType.float:
                        data = _vtk.vtkDoubleArray()
                    else:
                        data = _vtk.vtkIntArray()
                    data.SetName(key)
                    if vtk_tensor_type == VTKTensor.scalar:
                        data.SetNumberOfComponents(1)
                    else:
                        data.SetNumberOfComponents(3)

                    # Add the empty values for all previous cells / points.

                    for i in range(n_items):
                        self._add_single_data_item(data, vtk_tensor_type)
                    self.data[vtk_geom_type, vtk_tensor_type][key] = data

                else:
                    # In this case we just check that the already existing
                    # data has the same type.
                    data_array = self.data[vtk_geom_type, vtk_tensor_type][key]
                    if not _get_vtk_array_type(data_array) == data_type:
                        raise ValueError(
                            (
                                'The existing data with the key "{}"'
                                + ' is of type "{}", but the type you tried to add'
                                + ' is "{}"!'
                            ).format(key, data_array.GetDataTypeAsString(), data_type)
                        )

        # Add to global data. Check if there is something to be added. If not an empty value
        # is added.
        for key_tensor in VTKTensor:
            global_data = self.data[vtk_geom_type, key_tensor]
            if data_container is None:
                data_container = {}

            for key, value in global_data.items():
                # Check if an existing field is also given for this function.
                if key in data_container.keys():
                    # Get the data and the value type (int or float).
                    data_values, _ = _get_data_value_and_type(data_container[key])

                    # Add the given data.
                    if vtk_geom_type == VTKGeometry.cell:
                        self._add_single_data_item(
                            value, key_tensor, non_zero_data=data_values
                        )
                    else:
                        for item in data_values:
                            self._add_single_data_item(
                                value, key_tensor, non_zero_data=item
                            )
                else:
                    # Add empty data.
                    if vtk_geom_type == VTKGeometry.cell:
                        self._add_single_data_item(value, key_tensor)
                    else:
                        for item in range(n_new_items):
                            self._add_single_data_item(value, key_tensor)

    @staticmethod
    def _get_vtk_data_type(data):
        """Return the type of data.

        Check if data matches an expected case.
        """

        if isinstance(data, (list, _np.ndarray)):
            if len(data) == 3:
                return VTKTensor.vector
            raise IndexError(
                f"Only 3d vectors are implemented yet! Got len(data) = {len(data)}"
            )
        elif isinstance(data, _numbers.Number):
            return VTKTensor.scalar

        raise ValueError(f"Data {data} did not match any expected case!")

    @staticmethod
    def _add_single_data_item(data, vtk_tensor_type, non_zero_data=None):
        """Add data to a VTK data array."""

        if _get_vtk_array_type(data) == VTKType.int:
            nan_value = VTK_NAN_INT
        elif _get_vtk_array_type(data) == VTKType.float:
            nan_value = VTK_NAN_FLOAT

        if vtk_tensor_type == VTKTensor.scalar:
            if non_zero_data is None:
                data.InsertNextTuple1(nan_value)
            else:
                data.InsertNextTuple1(non_zero_data)
        else:
            if non_zero_data is None:
                data.InsertNextTuple3(nan_value, nan_value, nan_value)
            else:
                data.InsertNextTuple3(
                    non_zero_data[0], non_zero_data[1], non_zero_data[2]
                )

    def complete_data(self):
        """Add the stored data to the vtk grid."""
        for (key_geom, _key_data), value in self.data.items():
            for vtk_data in value.values():
                if key_geom == VTKGeometry.cell:
                    self.grid.GetCellData().AddArray(vtk_data)
                else:
                    self.grid.GetPointData().AddArray(vtk_data)

    def write_vtk(self, filepath, *, binary=True):
        """Write the VTK geometry and data to a file.

        Args
        ----
        filepath: str
            Path to output file. The file extension should be vtu.
        binary: bool
            If the data should be written encoded in binary or in human readable text.
        """

        # Check if directory for file exits.
        file_directory = _os.path.dirname(filepath)
        if not _os.path.isdir(file_directory):
            raise ValueError(f"Directory {file_directory} does not exist!".format())

        # Initialize VTK writer.
        writer = _vtk.vtkXMLUnstructuredGridWriter()

        # Set the ascii flag.
        if not binary:
            writer.SetDataModeToAscii()

        # Check the file extension.
        _filename, file_extension = _os.path.splitext(filepath)
        if not file_extension.lower() == ".vtu":
            _warnings.warn(f'The extension should be "vtu", got {file_extension}!')

        # Write geometry and data to file.
        writer.SetFileName(filepath)
        writer.SetInputData(self.grid)
        writer.Write()
