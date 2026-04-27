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
"""This file defines functions to dump mesh items for 4C."""

import copy as _copy
from collections import defaultdict as _defaultdict
from typing import Any as _Any
from typing import KeysView as _KeysView
from typing import List as _List

import numpy as _np
from fourcipp.fourc_input import FourCInput as _FourCInput

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.conf import Geometry as _Geometry
from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.function import Function as _Function
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.material import Material as _Material
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.core.mesh_representation import (
    merge_mesh_representations as _merge_mesh_representations,
)
from beamme.core.mesh_representation import (
    string_to_geometry_set_info as _string_to_geometry_set_info,
)
from beamme.core.nurbs_patch import NURBSPatch as _NURBSPatch
from beamme.core.rotation import Rotation as _Rotation
from beamme.four_c.four_c_types import (
    BeamKirchhoffParametrizationType as _BeamKirchhoffParametrizationType,
)
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)


def dump_function(function: _Function, i_global: int) -> dict[str, _Any]:
    """Return the representation of a function in the 4C input file."""
    return {f"FUNCT{i_global + 1}": function.data}


def dump_coupling(coupling):
    """Return the input file representation of the coupling condition."""

    if isinstance(coupling.data, dict):
        data = coupling.data
    else:
        # In this case we have to check which beams are connected to the node.
        # TODO: Coupling also makes sense for different beam types, this can
        # be implemented at some point.
        nodes = coupling.geometry_set.get_points()
        connected_elements = [
            element for node in nodes for element in node.element_link
        ]
        element_types = {type(element) for element in connected_elements}
        if len(element_types) > 1:
            raise TypeError(
                f"Expected a single connected type of beam elements, got {element_types}"
            )
        element_type = element_types.pop()
        if element_type.beam_type is _BeamType.kirchhoff:
            unique_parametrization_flags = {
                type(element).kirchhoff_parametrization
                for element in connected_elements
            }
            if (
                len(unique_parametrization_flags) > 1
                or not unique_parametrization_flags.pop()
                == _BeamKirchhoffParametrizationType.rot
            ):
                raise TypeError(
                    "Couplings for Kirchhoff beams and tangent "
                    "based parametrization not yet implemented."
                )

        data = element_type.get_coupling_dict(coupling.data)

    return {"E": coupling.geometry_set, **data}


def dump_nurbs_patch_knotvectors(input_file, nurbs_patch: _NURBSPatch) -> None:
    """Set the knot vectors of the NURBS patch in the input file."""

    patch_data: dict[str, _Any] = {
        "KNOT_VECTORS": [],
    }

    for dir_manifold in range(nurbs_patch.get_nurbs_dimension()):
        knotvector = nurbs_patch.knot_vectors[dir_manifold]
        num_knots = len(knotvector)

        # Check the type of knot vector, in case that the multiplicity of the first and last
        # knot vectors is not p + 1, then it is a closed (periodic) knot vector, otherwise it
        # is an open (interpolated) knot vector.
        knotvector_type = "Interpolated"

        for i in range(nurbs_patch.polynomial_orders[dir_manifold] - 1):
            if (abs(knotvector[i] - knotvector[i + 1]) > _bme.eps_knot_vector) or (
                abs(knotvector[num_knots - 2 - i] - knotvector[num_knots - 1 - i])
                > _bme.eps_knot_vector
            ):
                knotvector_type = "Periodic"
                break

        patch_data["KNOT_VECTORS"].append(
            {
                "DEGREE": nurbs_patch.polynomial_orders[dir_manifold],
                "TYPE": knotvector_type,
                "KNOTS": [
                    knot_vector_val
                    for knot_vector_val in nurbs_patch.knot_vectors[dir_manifold]
                ],
            }
        )

    if "STRUCTURE KNOTVECTORS" in input_file:
        # Get all existing patches in the input file - they will be added to the
        # input file again at the end of this function. By doing it this way, the
        # FourCIPP type converter will be applied to the current patch.
        # This also means that we apply the type converter again already existing
        # patches. But, with the usual number of patches and data size, this
        # should not lead to a measurable performance impact.
        patches = input_file.pop("STRUCTURE KNOTVECTORS")["PATCHES"]
    else:
        patches = []

    patch_data["ID"] = nurbs_patch
    patches.append(patch_data)
    input_file.add({"STRUCTURE KNOTVECTORS": {"PATCHES": patches}})


def dump_item(item) -> dict[str, _Any]:
    """General function to dump items to a 4C input file."""
    if hasattr(item, "dump_to_list"):
        return item.dump_to_list()
    elif isinstance(item, _BoundaryCondition):
        return {"E": item.geometry_set, **item.data}
    elif isinstance(item, _Coupling):
        return dump_coupling(item)
    else:
        raise TypeError(f"Could not dump {item}")


def dump_mesh_to_input_file(input_file, mesh: _Mesh) -> None:
    """Add a mesh to the input file.

    Internally, we store the geometry information from the mesh in the mesh
    representation of the input file. All other information, e.g., element
    types, materials, boundary conditions and functions will be dumped to the
    FourCIPP input file.

    Args:
        input_file: The input file where we want to add the mesh information.
        mesh: The mesh to be added to the input file.
    """

    # Compute starting indices
    #   Element types
    start_index_element_types = len(input_file.element_type_id_to_data)

    #   NURBS patches
    nurbs_patches = input_file.sections.get("STRUCTURE KNOTVECTORS", {}).get(
        "PATCHES", []
    )
    start_index_nurbs_patches = len(nurbs_patches)

    #   Geometry sets
    start_index_geometry_set = max(
        (
            entry["d_id"]  # We don't need a + 1 here, as this index is index-1 based.
            for section_name in _INPUT_FILE_MAPPINGS[
                "geometry_sets_geometry_to_condition_name"
            ].values()
            for entry in input_file.sections.get(section_name, [])
        ),
        default=0,
    )
    for name in input_file.mesh_representation.point_data.keys():
        info = _string_to_geometry_set_info(name)
        if info is not None:
            start_index_geometry_set = max(start_index_geometry_set, info.i_global + 1)

    #   Functions
    start_index_functions = max(
        (
            int(section.split("FUNCT")[-1])
            for section in input_file.sections
            if section.startswith("FUNCT")
        ),
        default=0,
    )

    #   Materials
    start_index_materials = max(
        (material["MAT"] for material in input_file.sections.get("MATERIALS", [])),
        default=0,
    )

    # Get the mesh representation for the mesh.
    (
        mesh_representation,
        mesh_element_type_id_to_data,
        geometry_sets_to_i_global,
        material_to_i_global,
        nurbs_patch_to_i_global,
    ) = mesh.get_mesh_representation()
    mesh_representation.offset_indices(
        element_type_id_offset=start_index_element_types,
        material_offset=start_index_materials,
        geometry_set_offset=start_index_geometry_set,
    )

    # Add the new element types to the mapping in the input file.
    for element_type_id, element_type_data in mesh_element_type_id_to_data.items():
        input_file.element_type_id_to_data[
            element_type_id + start_index_element_types
        ] = element_type_data
    input_file.mesh_representation = _merge_mesh_representations(
        input_file.mesh_representation, mesh_representation
    )

    # Dump functions to the input file.
    function_to_i_global: dict[_Function, int] = {}
    for i_global, function in enumerate(mesh.functions, start=start_index_functions):
        function_to_i_global[function] = i_global
        input_file.add(dump_function(function, i_global))
    if len(mesh.functions) != len(function_to_i_global):
        raise ValueError("Functions are not unique!")

    # Adapt the FourCIPP type converters such that the correct indices will be dumped.
    input_file.fourc_input.type_converter.register_type(
        _GeometrySetBase,
        lambda _, obj: geometry_sets_to_i_global[obj] + 1 + start_index_geometry_set,
    )
    input_file.fourc_input.type_converter.register_type(
        _Function, lambda _, obj: function_to_i_global[obj] + 1
    )
    input_file.fourc_input.type_converter.register_type(
        _Material,
        lambda _, obj: material_to_i_global[obj] + 1 + start_index_materials,
    )
    input_file.fourc_input.type_converter.register_type(
        _NURBSPatch,
        lambda _, obj: nurbs_patch_to_i_global[obj] + 1 + start_index_nurbs_patches,
    )

    def _dump(section_name: str, items: _List | _KeysView) -> None:
        """Dump list of items to a section in the input file.

        This function ensures that the dumped items will be appended to the
        existing section in the input file, and that the full data is parsed
        through the FourCIPP type converter again.

        Args:
            section_name: Name of the section
            items: List of items to be dumped
        """
        if not items:
            return
        dumped: list[_Any] = []
        for item in items:
            dumped.append(dump_item(item))

        # Go through FourCIPP to convert to native types
        full_item_list = input_file.pop(section_name, [])
        full_item_list.extend(dumped)
        input_file.add({section_name: full_item_list})

    # Dump materials
    _dump("MATERIALS", material_to_i_global.keys())

    # Dump couplings
    #   If there are couplings in the mesh, set the link between the nodes
    #   and elements, so the couplings can decide which DOFs they couple,
    #   depending on the type of the connected beam element.
    if any(
        mesh.boundary_conditions.get((key, _bme.geo.point), [])
        for key in (_bme.bc.point_coupling, _bme.bc.point_coupling_penalty)
    ):
        is_linked_nodes = True
        mesh.set_node_links()
    else:
        is_linked_nodes = False

    #   Boundary conditions
    for (bc_key, geometry_key), bc_list in mesh.boundary_conditions.items():
        if bc_list:
            section = (
                bc_key
                if isinstance(bc_key, str)
                else _INPUT_FILE_MAPPINGS["boundary_conditions"][(bc_key, geometry_key)]
            )
            _dump(section, bc_list)

    #   If we have previously set the node links, we unlink them here.
    if is_linked_nodes:
        mesh.unlink_nodes()

    # Dump NURBS patch information.
    for element in mesh.elements:
        if isinstance(element, _NURBSPatch):
            dump_nurbs_patch_knotvectors(input_file, element)


def dump_mesh_representation_to_input_file_legacy(
    fourc_input: _FourCInput,
    mesh_representation: _MeshRepresentation,
    element_type_id_to_data: dict[int, dict],
) -> None:
    """Dump the information contained in the mesh representation to the
    FourCIPP input file, in legacy format.

    Args:
        fourc_input: FourCIPP input file where the mesh information data will be dumped to.
        mesh_representation: The mesh representation that is added to the input file.
        element_type_id_to_data: The mapping between element type ID and the element type data.
    """

    # Compute the starting indices for the nodes and elements entities.
    start_index_nodes = len(fourc_input.sections.get("NODE COORDS", []))
    start_index_elements = sum(
        len(fourc_input.sections.get(section, []))
        for section in ("FLUID ELEMENTS", "STRUCTURE ELEMENTS")
    )

    # Dump the geometry information from the mesh representation to the input file.
    def _dump(section_name: str, dictionary_list: _List):
        """Append the given list of dictionaries to the section in the input
        file."""
        full_item_list = fourc_input.pop(section_name, [])
        full_item_list.extend(dictionary_list)
        fourc_input.combine_sections({section_name: full_item_list})

    #   Dump nodes
    node_data_list = []
    for i_node, (point, point_type, cp_weight) in enumerate(
        zip(
            mesh_representation.points,
            mesh_representation.data_iterator("point_data", "point_type"),
            mesh_representation.data_iterator("point_data", "control_point_weight"),
        ),
        start=start_index_nodes,
    ):
        node_type = _bme.node_type(point_type)
        node_id = i_node + 1
        if node_type == _bme.node_type.node or node_type == _bme.node_type.cosserat:
            node_data_list.append(
                {
                    "id": node_id,
                    "COORD": point,
                    "data": {"type": "NODE"},
                }
            )
        elif node_type == _bme.node_type.control_point:
            node_data_list.append(
                {
                    "id": node_id,
                    "COORD": point,
                    "data": {
                        "type": "CP",
                        "weight": cp_weight,
                    },
                }
            )
        else:
            raise ValueError(f"Unknown node type {node_type} for node {i_node + 1}")
    _dump("NODE COORDS", node_data_list)

    #   Dump elements
    element_list = []
    for i_element, (
        connectivity,
        element_type_id,
        element_material_id,
    ) in enumerate(
        zip(
            mesh_representation.connectivity_iterator(),
            mesh_representation.data_iterator("cell_data", "element_type_id"),
            mesh_representation.data_iterator("cell_data", "material_id"),
        )
    ):
        element_type_data = element_type_id_to_data[element_type_id]

        if len(element_type_data) == 1:
            four_c_type, four_c_cell_dict = next(iter(element_type_data.items()))
        else:
            raise ValueError("Expected exactly one entry in type dictionary")
        if len(four_c_cell_dict) == 1:
            four_c_cell, four_c_data = next(iter(four_c_cell_dict.items()))
        else:
            raise ValueError("Expected exactly one entry in cell dictionary")

        node_ordering = _INPUT_FILE_MAPPINGS[
            "four_c_cell_to_connectivity_mapping_from_vtk"
        ].get(four_c_cell, None)
        if node_ordering is not None:
            connectivity = connectivity[node_ordering]

        if element_material_id == -1:
            material_dict = {}
        else:
            material_dict = {"MAT": element_material_id + 1}

        if _INPUT_FILE_MAPPINGS["four_c_type_to_requires_triads"].get(
            four_c_type, False
        ):
            # We modify the dict to add the rotation vectors, thus we create a
            # deep copy here to avoid modifying the original data structure.
            four_c_data = _copy.deepcopy(four_c_data)
            # The numpy quaternion package can return rotation vectors outside
            # the range of -pi to pi, which can cause issues in testing comparisons.
            # To avoid this, we convert the rotations to BeamMe internal rotations
            # and extract the rotation vectors again, ensuring they are in the
            # correct range.
            # This is super slow, but for now keep it, as a mesh based output is to
            # be preferred when performance is of importance.
            if "rotation_vector" not in mesh_representation.point_data:
                raise KeyError(
                    f"Rotation vectors are required for type {four_c_type}, but "
                    "no rotation vectors found in the mesh representation!"
                )
            rotations = [
                _Rotation.from_rotation_vector(rotation_vector)
                for rotation_vector in mesh_representation.point_data[
                    "rotation_vector"
                ][connectivity]
            ]
            four_c_data["TRIADS"] = _np.array(
                [rotation.get_rotation_vector() for rotation in rotations]
            ).ravel()

        element_list.append(
            {
                "id": start_index_elements + i_element + 1,
                "cell": {
                    "type": four_c_cell,
                    "connectivity": start_index_nodes + connectivity + 1,
                },
                "data": {"type": four_c_type, **material_dict, **four_c_data},
            }
        )
    _dump("STRUCTURE ELEMENTS", element_list)

    # Geometry sets
    #   We first create a mapping from the geometry type to a dictionary which maps
    #   the global geometry set ID to the name of the corresponding data array in the
    #   mesh representation. This is required for the sorting later on.
    geometry_type_to_geometry_sets: dict[_Geometry, dict[int, str]] = _defaultdict(dict)
    for name in mesh_representation.point_data.keys():
        info = _string_to_geometry_set_info(name)
        if info is not None:
            geometry_type_to_geometry_sets[info.geometry_type][info.i_global] = name

    for geometry_type, id_to_name_map in geometry_type_to_geometry_sets.items():
        geometry_set_list = []
        # We sort the keys here to ensure that the geometry sets are dumped in the
        # correct order.
        sorted_ids = sorted(id_to_name_map.keys())
        for i_global in sorted_ids:
            name = id_to_name_map[i_global]
            node_indices = _np.nonzero(mesh_representation.point_data[name])[0]
            geometry_set_list.extend(
                [
                    {
                        "type": "NODE",
                        "node_id": start_index_nodes + node_index + 1,
                        "d_type": _INPUT_FILE_MAPPINGS[
                            "geometry_sets_geometry_to_entry_name"
                        ][geometry_type],
                        "d_id": i_global + 1,
                    }
                    for node_index in node_indices
                ]
            )
        _dump(
            _INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"][
                geometry_type
            ],
            geometry_set_list,
        )
