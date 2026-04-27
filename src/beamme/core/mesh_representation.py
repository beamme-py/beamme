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
"""This module defines the mesh representation data structure."""

from dataclasses import dataclass as _dataclass
from itertools import repeat as _repeat
from typing import Any as _Any
from typing import Iterable as _Iterable

import numpy as _np
from numpy.typing import NDArray as _NDArray

from beamme.core.conf import Geometry as _Geometry
from beamme.core.conf import bme as _bme

MESH_REPRESENTATION_MAPPINGS: dict[str, _Any] = {}
# fmt: off
MESH_REPRESENTATION_MAPPINGS[
    "element_type_and_n_nodes_to_connectivity_mapping_beamme_to_vtk"
] = {

    # Only list the non-standard mappings
    (_bme.element_type.solid, 20):
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15],
    (_bme.element_type.solid, 27):
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15,
         24, 22, 21, 23, 20, 25, 26],
}
# fmt: on
MESH_REPRESENTATION_MAPPINGS[
    "element_type_and_n_nodes_to_connectivity_mapping_vtk_to_beamme"
] = {
    # Only list the non-standard mappings
    (_bme.element_type.solid, 20): _np.argsort(
        MESH_REPRESENTATION_MAPPINGS[
            "element_type_and_n_nodes_to_connectivity_mapping_beamme_to_vtk"
        ][(_bme.element_type.solid, 20)]
    ),
    (_bme.element_type.solid, 27): _np.argsort(
        MESH_REPRESENTATION_MAPPINGS[
            "element_type_and_n_nodes_to_connectivity_mapping_beamme_to_vtk"
        ][(_bme.element_type.solid, 27)]
    ),
}

GEOMETRY_SET_INFO_PREFIX = "geometry_set"


@_dataclass
class GeometrySetInfo:
    """Data class that contains information of a geometry set."""

    geometry_type: _Geometry
    i_global: int
    name: str | None = None
    point_flag_vector: _NDArray | None = None
    cell_flag_vector: _NDArray | None = None

    def __str__(self):
        """Return a string representation of this geometry set info - the node and cell flags are not included."""
        name = self.name
        return (
            f"{GEOMETRY_SET_INFO_PREFIX}_{self.i_global}_{self.geometry_type.name}"
            + (f"_{name}" if name is not None else "")
        )


def string_to_geometry_set_info(name: str) -> GeometrySetInfo | None:
    """Extract the geometry set information from a given string."""
    if not name.startswith(GEOMETRY_SET_INFO_PREFIX):
        return None

    name_without_prefix = name[len(GEOMETRY_SET_INFO_PREFIX) + 1 :]
    split = name_without_prefix.split("_", 2)
    i_global = int(split[0])
    geometry_type = _Geometry[split[1]]
    return GeometrySetInfo(
        i_global=i_global,
        geometry_type=geometry_type,
        name=split[2] if len(split) == 3 else None,
    )


class MeshRepresentation:
    """Class representing a generic mesh."""

    def __init__(
        self,
        cell_connectivity: _NDArray[_np.integer] | None = None,
        cell_types: _NDArray[_np.integer] | None = None,
        points: _NDArray[_np.floating] | None = None,
        geometry_sets: list[GeometrySetInfo] | None = None,
        cell_data: dict[str, _NDArray | None] | None = None,
        point_data: dict[str, _NDArray | None] | None = None,
    ):
        def _convert_argument_numpy(
            argument: _NDArray | None,
            default_shape: tuple[int, ...],
            dtype: type,
        ) -> _NDArray:
            """Convert given array arguments so we can store them in this
            object."""
            if argument is None:
                return _np.empty(default_shape, dtype=dtype)
            else:
                return _np.asarray(argument, dtype=dtype)

        def _filter_none_entries(
            argument: dict[str, _NDArray | None] | None, expected_size: int
        ) -> dict[str, _NDArray]:
            """Check if a dictionary is given, and if so, filter None entries
            from it."""
            if argument is None:
                return {}
            else:
                data = {
                    key: _np.asarray(value)
                    for key, value in argument.items()
                    if value is not None
                }
                for key, value in data.items():
                    if len(value) != expected_size:
                        raise ValueError(
                            f"Data field {key} has size {len(value)}, but expected size is {expected_size}."
                        )
                return data

        self.cell_connectivity = _convert_argument_numpy(
            cell_connectivity, default_shape=(0,), dtype=int
        )
        self.cell_types = _convert_argument_numpy(
            cell_types, default_shape=(0,), dtype=int
        )
        self.points = _convert_argument_numpy(points, default_shape=(0, 3), dtype=float)

        self.cell_data = _filter_none_entries(cell_data, self.n_cells)
        self.point_data = _filter_none_entries(point_data, self.n_points)

        def _store_geometry_set_data(
            data_field: str,
            name: str,
            field_vector: _NDArray | None,
            expected_size: int,
        ) -> None:
            """Store point or cell data defining a geometry set."""
            if field_vector is not None:
                if len(field_vector) != expected_size:
                    raise ValueError(
                        f"Geometry set {name} has size {len(field_vector)},"
                        f" but expected size is {expected_size}."
                    )
                getattr(self, data_field)[str(name)] = field_vector

        if geometry_sets is not None:
            for geometry_set in geometry_sets:
                geometry_set_name = str(geometry_set)
                _store_geometry_set_data(
                    "point_data",
                    geometry_set_name,
                    geometry_set.point_flag_vector,
                    self.n_points,
                )
                _store_geometry_set_data(
                    "cell_data",
                    geometry_set_name,
                    geometry_set.cell_flag_vector,
                    self.n_cells,
                )

        # Get the offset array so we can iterate over the connectivity in a
        # performant manner.
        self.cell_connectivity_offsets = _np.zeros(self.n_cells + 1, dtype=int)
        for i_cell in range(self.n_cells):
            last_offset = self.cell_connectivity_offsets[i_cell]
            if last_offset >= len(self.cell_connectivity):
                raise ValueError(
                    "Invalid offset, is larger than the size of the connectivity."
                )
            cell_size = self.cell_connectivity[last_offset]
            if cell_size < 1:
                raise ValueError("Invalid offset, has to be at least 1.")
            self.cell_connectivity_offsets[i_cell + 1] = (
                cell_size + self.cell_connectivity_offsets[i_cell] + 1
            )
        if self.cell_connectivity_offsets[-1] != len(self.cell_connectivity):
            raise ValueError("Invalid cell connectivity offsets.")

    @property
    def n_cells(self) -> int:
        """Return the number of cells in this mesh representation.

        Returns:
            Number of cells in this mesh representation.
        """
        return len(self.cell_types)

    @property
    def n_points(self) -> int:
        """Return the number of points in this mesh representation.

        Returns:
            Number of points in this mesh representation.
        """
        return len(self.points)

    def connectivity_iterator(self) -> _Iterable:
        """Return an iterator over the cell connectivity.

        Returns:
            An iterator that returns the connectivity array for each cell.
        """
        for i in range(self.n_cells):
            start = self.cell_connectivity_offsets[i] + 1
            end = self.cell_connectivity_offsets[i + 1]
            yield self.cell_connectivity[start:end]

    def data_iterator(self, data_field: str, data_name: str) -> _Iterable:
        """This method returns an iterator for the given data field and data
        name.

        This is useful, when looping over the data, as accessing the data field in each
        loop iteration can be expensive. If the data field is not present, a iterator
        is returned that will always return `None`.

        Args:
            data_field: The data field to get the iterator for. This can be either
                "point_data" or "cell_data".
            data_name: The name of the data to get the iterator for.

        Returns:
            An iterator to iterate through the data. If the data field is not present,
            a iterator is returned that will always return `None`.
        """
        data_dict = getattr(self, data_field)
        if data_name in data_dict:
            return data_dict[data_name]
        else:
            match data_field:
                case "point_data":
                    size = self.n_points
                case "cell_data":
                    size = self.n_cells
                case _:
                    raise ValueError(f"Invalid data field: {data_field}")
            return _repeat(None, size)

    def offset_indices(
        self,
        element_type_id_offset: int | None = None,
        material_offset: int | None = None,
        geometry_set_offset: int | None = None,
    ) -> None:
        """Add offsets to the internal indices of this mesh representation.

        This is required when merging multiple mesh representations together,
        to ensure that there are no index conflicts.

        Args:
            element_type_id_offset: The offset to add to the element type IDs.
            material_offset: The offset to add to the material IDs.
            geometry_set_offset: The offset to add to the geometry set IDs.
        """

        if element_type_id_offset is not None:
            if "element_type_id" in self.cell_data:
                self.cell_data["element_type_id"] += element_type_id_offset

        if material_offset is not None:
            if "material_id" in self.cell_data:
                # Add the offset to all material entries that are not -1. We use
                # -1 to indicate no assigned material and those values should not change.
                self.cell_data["material_id"][self.cell_data["material_id"] >= 0] += (
                    material_offset
                )

        if geometry_set_offset is not None:
            for field_type in ("point_data", "cell_data"):
                new_dict = {}
                data_field = getattr(self, field_type)
                # We change the keys in the loop, so we iterate over a copy of them.
                for name in list(data_field.keys()):
                    info = string_to_geometry_set_info(name)
                    if info is not None:
                        new_name = str(
                            GeometrySetInfo(
                                geometry_type=info.geometry_type,
                                i_global=info.i_global + geometry_set_offset,
                                name=info.name,
                            )
                        )
                        new_dict[new_name] = data_field.pop(name)
                # Add the renamed geometry sets back to the data field.
                setattr(self, field_type, {**data_field, **new_dict})


def merge_mesh_representations(
    mesh_representation_a: MeshRepresentation, mesh_representation_b: MeshRepresentation
) -> MeshRepresentation:
    """Merge two mesh representations.

    Args:
        mesh_representation_a: First mesh representation.
        mesh_representation_b: Second mesh representation.

    Returns:
        A merged mesh representation. Depending on the input data, this might
        be a reference to one of the input mesh representations.
    """

    def _is_empty(mesh_representation: MeshRepresentation) -> bool:
        """Check if the mesh representation contains any points or cells."""
        return mesh_representation.n_points == 0 and mesh_representation.n_cells == 0

    # If one of the two mesh representations is empty, we can simply return the other (which
    # could be empty as well).
    if _is_empty(mesh_representation_a):
        return mesh_representation_b
    elif _is_empty(mesh_representation_b):
        return mesh_representation_a

    # Create merged data entries
    #   Points
    merged_points = _np.vstack(
        (mesh_representation_a.points, mesh_representation_b.points)
    )

    #   Connectivity - before the connectivity can be merged, the IDs of the second mesh have
    #   to be increased by the size of the first mesh.
    #   First, we create a mask to identify the entries that have to be incremented.
    cell_connectivity_a = mesh_representation_a.cell_connectivity
    cell_connectivity_b = mesh_representation_b.cell_connectivity
    offsets_b = mesh_representation_b.cell_connectivity_offsets

    merged_cell_connectivity = _np.empty(
        cell_connectivity_a.size + cell_connectivity_b.size,
        dtype=cell_connectivity_a.dtype,
    )
    merged_cell_connectivity[: cell_connectivity_a.size] = cell_connectivity_a
    merged_cell_connectivity[cell_connectivity_a.size :] = cell_connectivity_b

    mask = _np.ones(cell_connectivity_b.size, dtype=bool)
    mask[offsets_b[:-1]] = False

    merged_cell_connectivity_view_part_b = merged_cell_connectivity[
        cell_connectivity_a.size :
    ]
    merged_cell_connectivity_view_part_b[mask] += mesh_representation_a.n_points

    #   The cell types can simply be merged.
    merged_cell_types = _np.concatenate(
        (
            mesh_representation_a.cell_types,
            mesh_representation_b.cell_types,
        )
    )

    #   Contained data
    def _merge_data_dicts(
        dict_a: dict[str, _NDArray],
        size_a: int,
        dict_b: dict[str, _NDArray],
        size_b: int,
    ) -> dict[str, _NDArray]:
        """Merge the given data dictionaries and fill in non-existing data."""

        def _ensure_array_size(size: int, reference_array: _NDArray) -> _NDArray:
            """Create an empty array that matches the columns of the reference
            array and has the given size, i.e., number of rows."""
            new_shape = reference_array.shape
            if len(new_shape) == 1:
                new_shape = (size,)
            elif len(new_shape) == 2:
                new_shape = (size, new_shape[1])
            else:
                raise ValueError(f"Got unexpected array shape {new_shape}.")
            return _np.zeros(new_shape, dtype=reference_array.dtype)

        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        merged_data = {}
        for key in all_keys:
            data_a = dict_a.get(key, None)
            data_b = dict_b.get(key, None)
            if data_a is None:
                data_a = _ensure_array_size(size_a, data_b)
            elif data_b is None:
                data_b = _ensure_array_size(size_b, data_a)
            merged_data[key] = _np.concatenate((data_a, data_b))
        return merged_data

    merged_cell_data = _merge_data_dicts(
        mesh_representation_a.cell_data,
        mesh_representation_a.n_cells,
        mesh_representation_b.cell_data,
        mesh_representation_b.n_cells,
    )
    merged_point_data = _merge_data_dicts(
        mesh_representation_a.point_data,
        mesh_representation_a.n_points,
        mesh_representation_b.point_data,
        mesh_representation_b.n_points,
    )

    return MeshRepresentation(
        cell_connectivity=merged_cell_connectivity,
        cell_types=merged_cell_types,
        points=merged_points,
        cell_data=merged_cell_data,
        point_data=merged_point_data,
    )
