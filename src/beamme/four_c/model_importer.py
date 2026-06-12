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
"""This module contains functions to load and parse existing 4C input files."""

import tempfile as _tempfile
from collections import defaultdict as _defaultdict
from pathlib import Path as _Path
from typing import Tuple as _Tuple

import numpy as _np

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from beamme.core.conf import Geometry as _Geometry
from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_representation import (
    MESH_REPRESENTATION_MAPPINGS as _MESH_REPRESENTATION_MAPPINGS,
)
from beamme.core.mesh_representation import GeometrySetInfo as _GeometrySetInfo
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.core.mesh_representation import (
    string_to_geometry_set_info as _string_to_geometry_set_info,
)
from beamme.core.node import Node as _Node
from beamme.four_c.element_data import FourCElementData as _FourCElementData
from beamme.four_c.element_data import (
    four_c_element_data_from_exo_dict as _four_c_element_data_from_exo_dict,
)
from beamme.four_c.element_data import (
    four_c_element_data_from_yaml_dict as _four_c_element_data_from_yaml_dict,
)
from beamme.four_c.element_solid import get_four_c_solid as _get_four_c_solid
from beamme.four_c.input_file import InputFile as _InputFile
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.four_c.material import MaterialSolid as _MaterialSolid
from beamme.utils.data_structures import (
    create_inverse_mapping as _create_inverse_mapping,
)
from beamme.utils.environment import cubitpy_is_available as _cubitpy_is_available

if _cubitpy_is_available():
    import netCDF4 as _netCDF4
    from cubitpy.conf import cupy as _cupy
    from cubitpy.cubit_to_fourc_input import get_exo_info as _get_exo_info
    from cubitpy.cubit_utility import (
        string_to_node_set_info as _string_to_node_set_info,
    )


class UniqueDataTracker:
    """Helper class to track unique data dictionaries and assign IDs to them.

    When importing input files, we need to identify elements of the same
    type. The type information is given in dictionaries. This class
    provides a tracker that can be queried with a given element data and
    return an already matching element type ID or create a new one.
    """

    def __init__(self) -> None:
        self.unique_id_to_data: dict[int, _FourCElementData] = {}

    def get_unique_id(self, data: _FourCElementData) -> int:
        """Get the unique ID for the given data. If the data has not been seen
        before, a new ID will be assigned to it.

        Args:
            data: The data dictionary to get the ID for.

        Returns:
            The unique ID for the given data.
        """
        for unique_id, seen_data in self.unique_id_to_data.items():
            if data == seen_data:
                return unique_id

        # If we reach this point, the data has not been seen before. We assign a new ID to it.
        new_unique_id = len(self.unique_id_to_data)
        self.unique_id_to_data[new_unique_id] = data
        return new_unique_id


def import_cubitpy_model(
    cubit, convert_input_to_mesh: bool = False
) -> _Tuple[_InputFile, _Mesh]:
    """Convert a CubitPy instance to a BeamMe InputFile.

    Args:
        cubit (CubitPy): An instance of a cubit model.
        convert_input_to_mesh: If this is false, the cubit model will be
            converted to plain FourCIPP input data. If this is true, an input
            file with all the parameters will be returned and a mesh which
            contains the mesh information from cubit converted to BeamMe
            objects.

    Returns:
        A tuple with the input file and the mesh. If convert_input_to_mesh is
        False, the mesh will be empty. Note that the input sections which are
        converted to a BeamMe mesh are removed from the input file object.
    """
    temp_dir: str | _Path
    with _tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = _Path(temp_dir)
        input_file_path = temp_dir / "temp_cubit_input_file.4C.yaml"
        cubit.dump(
            input_file_path, mesh_in_exo=True, mesh_in_exo_add_node_set_info=True
        )
        return import_four_c_model(
            input_file_path, convert_input_to_mesh=convert_input_to_mesh
        )


def import_four_c_model(
    input_file_path: _Path, convert_input_to_mesh: bool = False
) -> _Tuple[_InputFile, _Mesh]:
    """Import an existing 4C input file and optionally convert it into a BeamMe
    mesh.

    Args:
        input_file_path: A file path to an existing 4C input file that will be
            imported.
        convert_input_to_mesh: If True, the input file will be converted to a
            BeamMe mesh.

    Returns:
        A tuple with the input file and the mesh. If convert_input_to_mesh is
        False, the mesh will be empty. Note that the input sections which are
        converted to a BeamMe mesh are removed from the input file object.
    """

    input_file = _InputFile().from_4C_yaml(input_file_path=input_file_path)
    base_path = input_file_path.parent

    if input_file.contains_mesh_based_geometry_exodus():
        (
            mesh_representation,
            element_type_id_to_data,
            node_set_id_mesh_representation_to_input_file,
        ) = _extract_mesh_representation_from_exo(input_file, base_path)

    elif convert_input_to_mesh:
        (
            mesh_representation,
            element_type_id_to_data,
            node_set_id_mesh_representation_to_input_file,
        ) = _extract_mesh_representation(input_file)

    else:
        mesh_representation = None
        element_type_id_to_data = None

    if convert_input_to_mesh:
        return _create_mesh_from_mesh_representation(
            input_file,
            mesh_representation,
            element_type_id_to_data,
            node_set_id_mesh_representation_to_input_file,
        )
    else:
        if mesh_representation is not None:
            input_file.mesh_representation = mesh_representation
        if element_type_id_to_data is not None:
            input_file.element_type_id_to_data = element_type_id_to_data
        return input_file, _Mesh()


def _extract_mesh_representation(
    input_file: _InputFile,
) -> tuple[_MeshRepresentation, dict[int, _FourCElementData], dict[int, int]]:
    """Extract the mesh representation from mesh data directly contained in the
    input file.

    This will do an inplace removal of the mesh data from the provided input file.

    Args:
        input_file: The input file containing the mesh data, will be modified in place.

    Returns:
        A tuple containing:
        - `mesh_representation`: Contains the mesh data extracted from the input file.
        - `element_type_id_to_data`: A mapping between the element type ID and the
            element data.
        - `node_set_id_mesh_representation_to_input_file`: A mapping that can be used
            to map the IDs in the mesh representation to the IDs in the input file.
    """

    # extract nodes
    nodes = input_file.pop("NODE COORDS", [])
    n_nodes = len(nodes)
    points = _np.zeros((n_nodes, 3))
    point_types = _np.full(n_nodes, -1)
    control_point_weights = _np.full(n_nodes, -1.0)
    for i, node in enumerate(nodes):
        four_c_node_type = node["data"]["type"]
        node_id = node["id"]
        try:
            node_type = _INPUT_FILE_MAPPINGS["four_c_node_type_to_beamme_node_type"][
                four_c_node_type
            ]
        except KeyError:
            raise ValueError(
                f"Unknown node type `{four_c_node_type}` for node {node_id}."
            )
        points[i] = node["COORD"]
        point_types[i] = node_type.value
        if node_type == _bme.node_type.control_point:
            control_point_weights[i] = node["data"]["weight"]

    # extract elements
    element_type_tracker = UniqueDataTracker()
    elements = input_file.pop("STRUCTURE ELEMENTS", [])
    n_elements = len(elements)
    cell_connectivity = []
    cell_types = _np.full(n_elements, -1)
    cell_element_type_ids = _np.full(n_elements, -1)
    cell_material_ids = _np.full(n_elements, -1)
    for i_element, input_element in enumerate(elements):
        four_c_element_data, element_id, connectivity, material_id = (
            _four_c_element_data_from_yaml_dict(input_element)
        )
        element_type_id = element_type_tracker.get_unique_id(four_c_element_data)

        # Check if connectivity has to be reordered
        reorder_indices = _INPUT_FILE_MAPPINGS[
            "four_c_cell_to_vtk_connectivity_mapping"
        ].get(four_c_element_data.four_c_cell, None)
        if reorder_indices is not None:
            connectivity = connectivity[reorder_indices]

        cell_connectivity.extend([len(connectivity), *connectivity.tolist()])

        try:
            vtk_cell_type = _INPUT_FILE_MAPPINGS["four_c_cell_to_vtk_cell_type"][
                four_c_element_data.four_c_cell
            ]
        except KeyError:
            raise ValueError(
                f"Unknown cell type `{four_c_element_data.four_c_cell}` for element {element_id}."
            )

        cell_types[i_element] = vtk_cell_type
        cell_element_type_ids[i_element] = element_type_id
        cell_material_ids[i_element] = material_id

    # extract geometry sets
    node_sets: list[_GeometrySetInfo] = []
    node_set_id_mesh_representation_to_input_file: dict[int, int] = {}
    for section_name in input_file.sections:
        if not section_name.endswith("TOPOLOGY"):
            continue

        items = input_file.pop(section_name, [])
        if not items:
            continue

        # Find geometry type for this section
        try:
            geometry_type = _INPUT_FILE_MAPPINGS[
                "geometry_sets_condition_to_geometry_name"
            ][section_name]
        except KeyError as e:
            raise ValueError(f"Unknown geometry section: {section_name}") from e

        # Extract geometry set indices
        geom_dict: dict[int, set[int]] = _defaultdict(set)
        for entry in items:
            geom_dict[entry["d_id"]].add(entry["node_id"] - 1)

        for input_file_node_set_id, node_ids in geom_dict.items():
            node_set_id = len(node_sets)

            node_set_flag = _np.zeros(n_nodes, dtype=int)
            node_set_flag[list(node_ids)] = 1

            node_sets.append(
                _GeometrySetInfo(
                    geometry_type=geometry_type,
                    i_global=node_set_id,
                    point_flag_vector=node_set_flag,
                )
            )

            node_set_id_mesh_representation_to_input_file[node_set_id] = (
                input_file_node_set_id
            )

    # Create the mesh representation and add the extracted data to it.
    mesh_representation = _MeshRepresentation(
        cell_connectivity=cell_connectivity,
        cell_types=cell_types,
        points=points,
        geometry_sets=node_sets,
        point_data={
            "point_type": point_types,
            "control_point_weight": control_point_weights,
        },
        cell_data={
            "element_type_id": cell_element_type_ids,
            "material_id": cell_material_ids,
        },
    )

    return (
        mesh_representation,
        element_type_tracker.unique_id_to_data,
        node_set_id_mesh_representation_to_input_file,
    )


def _get_exodus_path_from_input_file(input_file: _InputFile, base_path: _Path) -> _Path:
    """Returns the path to the exodus file linked in the input file.

    Args:
        input_file: The input file to extract the exodus file path from.
        base_path: The base path for loading the exodus file.

    Returns:
        The path to the exodus file linked in the input file.
    """

    if "STRUCTURE GEOMETRY" in input_file.fourc_input:
        structure_geometry_section = input_file.fourc_input["STRUCTURE GEOMETRY"]
        if "FILE" in structure_geometry_section:
            exodus_file_name = _Path(structure_geometry_section["FILE"])
            if exodus_file_name.suffix.lower() in [".exo", ".e"]:
                exodus_file_path = base_path / exodus_file_name
                if not exodus_file_path.is_file():
                    raise FileNotFoundError(
                        "The input file contains a link to an external mesh file "
                        f"{exodus_file_name}, but this file does not exist at the expected "
                        f"location {exodus_file_path}."
                    )
                return exodus_file_path
            else:
                raise ValueError(
                    "The input file contains a link to an external file "
                    f"{exodus_file_name}, with the extension {exodus_file_name.suffix}, but only "
                    ".exo and .e files are supported."
                )
        else:
            raise ValueError(
                "The input file contains a STRUCTURE GEOMETRY section, but no "
                "FILE entry."
            )
    else:
        raise ValueError("The input file does not contain a STRUCTURE GEOMETRY section")


def _extract_mesh_representation_from_exo(
    input_file: _InputFile, base_path: _Path
) -> tuple[_MeshRepresentation, dict[int, _FourCElementData], dict[int, int]]:
    """Extract the mesh representation from mesh data in exodus format.

    This will do an inplace removal of the extracted sections in the provided input file.

    Args:
        input_file: The input file containing the mesh data, will be modified in place.

    Returns:
        A tuple (mesh_representation, element_type_id_to_data, node_set_id_mesh_representation_to_input_file).
        - `mesh_representation`: Contains the mesh data extracted from the input file.
        - `element_type_id_to_data`: A mapping between the element type ID and the element data.
        - `node_set_id_mesh_representation_to_input_file`: A mapping that can be used to map the
           geometry set IDs in the mesh representation to the ones in the input file.
    """

    # Load the exodus file.
    with _netCDF4.Dataset(
        _get_exodus_path_from_input_file(input_file, base_path)
    ) as exo:
        # Read the coordinate array.
        if "coordz" not in exo.variables:
            raise ValueError(
                "The exodus file provides only 2D coordinates, this is not supported."
            )
        coordinates = _np.array(
            [exo.variables["coord" + dim][:] for dim in ["x", "y", "z"]],
        ).transpose()
        n_points = coordinates.shape[0]
        point_types = _np.full(n_points, _bme.node_type.node.value)

        # Remove the structure geometry section from the input file and extract the
        # element blocks.
        element_blocks = input_file.pop("STRUCTURE GEOMETRY")["ELEMENT_BLOCKS"]
        cubit_id_to_element_blocks = {block["ID"]: block for block in element_blocks}

        # Add the element connectivity
        element_type_tracker = UniqueDataTracker()
        block_connectivity_list = []
        cell_types = []
        cell_element_type_ids = []
        cell_material_ids = []
        _, exo_block_id_to_info = _get_exo_info(exo, "block")
        for exo_id in sorted(exo_block_id_to_info.keys()):
            # First, get the element block ID in beamme and the corresponding element data.
            info = exo_block_id_to_info[exo_id]
            element_data = cubit_id_to_element_blocks[info["cubit_id"]]
            four_c_element_data, material_id = _four_c_element_data_from_exo_dict(
                element_data
            )
            element_type_id = element_type_tracker.get_unique_id(four_c_element_data)

            # Get the connectivity information for this block and if necessary
            # reorder it to match the VTK node ordering.
            connectivity = exo.variables[f"connect{exo_id + 1}"][:] - 1
            n_elements_in_block = connectivity.shape[0]
            n_nodes_per_element = connectivity.shape[1]
            if (
                not n_nodes_per_element
                == _INPUT_FILE_MAPPINGS["four_c_cell_to_element_type_and_n_nodes"][
                    four_c_element_data.four_c_cell
                ][1]
            ):
                raise ValueError(
                    f"Number of nodes per element {n_nodes_per_element} in block with "
                    f"ID {info['cubit_id']} does not match expected number of nodes for "
                    f"cell type {four_c_element_data.four_c_cell}."
                )
            reordering = _MESH_REPRESENTATION_MAPPINGS[
                "connectivity_mapping_exodus_to_vtk"
            ].get(n_nodes_per_element, None)
            if reordering is not None:
                connectivity = connectivity[:, reordering]

            # Add the data for this element block to the cell data lists.
            cell_material_ids.extend([material_id] * n_elements_in_block)
            cell_element_type_ids.extend([element_type_id] * n_elements_in_block)
            cell_types.extend(
                [
                    _INPUT_FILE_MAPPINGS["four_c_cell_to_vtk_cell_type"][
                        four_c_element_data.four_c_cell
                    ]
                ]
                * n_elements_in_block
            )
            block_connectivity = _np.empty(
                (n_elements_in_block, n_nodes_per_element + 1), dtype=int
            )
            block_connectivity[:, 0] = n_nodes_per_element
            block_connectivity[:, 1:] = connectivity
            block_connectivity_list.append(block_connectivity.ravel())
        cell_connectivity = _np.concatenate(block_connectivity_list)

        # Extract the node sets.
        cubitpy_geometry_type_to_beamme = {
            _cupy.geometry.vertex: _bme.geo.point,
            _cupy.geometry.curve: _bme.geo.line,
            _cupy.geometry.surface: _bme.geo.surface,
            _cupy.geometry.volume: _bme.geo.volume,
        }
        node_set_id_mesh_representation_to_input_file = {}
        geometry_sets: list[_GeometrySetInfo] = []
        _, exo_node_set_id_to_info = _get_exo_info(exo, "nodeset")
        for exo_id in sorted(exo_node_set_id_to_info.keys()):
            exo_name = exo_node_set_id_to_info[exo_id]["name"]
            cubit_id, geometry_type_cubitpy, name = _string_to_node_set_info(exo_name)

            node_set_flag = _np.zeros(n_points, dtype=int)
            node_set_flag[exo.variables[f"node_ns{exo_id + 1}"][:] - 1] = 1

            mesh_representation_id = len(geometry_sets)
            node_set_id_mesh_representation_to_input_file[mesh_representation_id] = (
                cubit_id
            )
            geometry_sets.append(
                _GeometrySetInfo(
                    geometry_type=cubitpy_geometry_type_to_beamme[
                        geometry_type_cubitpy
                    ],
                    i_global=mesh_representation_id,
                    point_flag_vector=node_set_flag,
                    name=name,
                )
            )

    # Remove the entries in the boundary condition definitions in the input file that
    # are exodus specific. Also, set the geometry set IDs in the boundary conditions to
    # the ones in the mesh representation.
    node_set_id_input_file_to_mesh_representation = _create_inverse_mapping(
        node_set_id_mesh_representation_to_input_file
    )
    for section_name in input_file.sections:
        if section_name in _INPUT_FILE_MAPPINGS["boundary_conditions"].values():
            items = input_file.pop(section_name)
            for bc in items:
                bc.pop("ENTITY_TYPE")
                bc["E"] = node_set_id_input_file_to_mesh_representation[bc["E"]] + 1
            input_file.add({section_name: items})

    # Create the mesh representation
    mesh_representation = _MeshRepresentation(
        cell_connectivity=cell_connectivity,
        cell_types=cell_types,
        points=coordinates,
        geometry_sets=geometry_sets,
        point_data={"point_type": point_types},
        cell_data={
            "element_type_id": cell_element_type_ids,
            "material_id": cell_material_ids,
        },
    )

    return (
        mesh_representation,
        element_type_tracker.unique_id_to_data,
        {i: i + 1 for i in range(len(node_set_id_mesh_representation_to_input_file))},
    )


def _create_mesh_from_mesh_representation(
    input_file,
    mesh_representation,
    element_type_id_to_data,
    node_set_id_mesh_representation_to_input_file,
) -> tuple[_InputFile, _Mesh]:
    """Extract a BeamMe mesh from a mesh representation.

    Args:
        input_file: The input file containing general data.
        mesh_representation: The mesh representation to convert.
        node_set_id_mesh_representation_to_input_file: A mapping of the mesh
            representation node set IDs to input file IDs, which can be used to link
            the geometry sets in the input file to the node sets in the mesh
            representation.

    Returns:
        A tuple (input_file, mesh). The input_file is modified in place to remove
        sections converted into the BeamMe mesh.
    """

    # convert all sections to native objects and add to a new mesh
    mesh = _Mesh()

    # extract materials
    material_id_map = _extract_materials_from_input_file(input_file)
    mesh.materials.extend(material_id_map.values())

    # extract nodes
    for node_coordinates, node_type in zip(
        mesh_representation.points, mesh_representation.point_data["point_type"]
    ):
        if node_type == _bme.node_type.node.value:
            mesh.nodes.append(_Node(node_coordinates))
        else:
            raise ValueError(
                f"Mesh conversion for node type {_bme.node_type(node_type).name} is not implemented!"
            )

    # extract element types
    element_type_id_to_element_type: dict[int, type] = {}
    for (
        element_type_id,
        element_data,
    ) in element_type_id_to_data.items():
        element_type, n_nodes = _INPUT_FILE_MAPPINGS[
            "four_c_cell_to_element_type_and_n_nodes"
        ][element_data.four_c_cell]
        if not element_type == _bme.element_type.solid:
            raise ValueError(
                f"Mesh conversion for element type {element_type} is not implemented!"
            )
        element_type_id_to_element_type[element_type_id] = _get_four_c_solid(
            element_type,
            element_data.four_c_type,
            n_nodes=n_nodes,
            element_technology=element_data.element_technology,
        )

    # loop over the elements and create the mesh elements with the correct type, connectivity and material.
    for connectivity, cell_element_type_id_, material_id in zip(
        mesh_representation.connectivity_iterator(),
        mesh_representation.cell_data["element_type_id"],
        mesh_representation.cell_data["material_id"],
    ):
        element_type = element_type_id_to_element_type[cell_element_type_id_]

        reorder_indices = _MESH_REPRESENTATION_MAPPINGS[
            "element_type_and_n_nodes_to_connectivity_mapping_vtk_to_beamme"
        ].get((element_type.element_type, len(connectivity)), None)
        if reorder_indices is not None:
            nodes = [mesh.nodes[connectivity[i]] for i in reorder_indices]
        else:
            nodes = [mesh.nodes[i] for i in connectivity]

        element = element_type(nodes=nodes)

        if not material_id == -1:
            element.material = material_id_map[material_id]
        mesh.elements.append(element)

    # extract geometry sets
    geometry_sets_in_sections: dict[_Geometry, dict[int, _GeometrySetNodes]] = (
        _defaultdict(dict)
    )
    for name in mesh_representation.point_data.keys():
        info = _string_to_geometry_set_info(name)
        if info is not None:
            node_indices = _np.nonzero(mesh_representation.point_data[name])[0]
            geometry_type = info.geometry_type
            geometry_set = _GeometrySetNodes(
                geometry_type,
                nodes=[mesh.nodes[i] for i in node_indices],
                name=info.name,
            )
            input_file_id = node_set_id_mesh_representation_to_input_file[info.i_global]
            geometry_sets_in_sections[geometry_type][input_file_id] = geometry_set
            mesh.add(geometry_set)

    # extract boundary conditions
    _standard_bc_types = (
        _bme.bc.dirichlet,
        _bme.bc.neumann,
        _bme.bc.locsys,
        _bme.bc.beam_to_solid_surface_meshtying,
        _bme.bc.beam_to_solid_surface_contact,
        _bme.bc.beam_to_solid_volume_meshtying,
    )

    for (bc_key, geometry_type), section_name in _INPUT_FILE_MAPPINGS[
        "boundary_conditions"
    ].items():
        for bc_data in input_file.pop(section_name, []):
            geometry_set = geometry_sets_in_sections[geometry_type][bc_data.pop("E")]

            bc_obj: _BoundaryConditionBase

            if bc_key in _standard_bc_types or isinstance(bc_key, str):
                bc_obj = _BoundaryCondition(geometry_set, bc_data, bc_type=bc_key)
            elif bc_key is _bme.bc.point_coupling:
                bc_obj = _Coupling(
                    geometry_set, bc_key, bc_data, check_overlapping_nodes=False
                )
            else:
                raise ValueError(f"Unexpected boundary condition: {bc_key}")

            mesh.boundary_conditions.append((bc_key, geometry_type), bc_obj)

    return input_file, mesh


def _extract_materials_from_input_file(
    input_file: _InputFile,
) -> dict[int, _MaterialSolid]:
    """Extract all materials from the input file and convert them to BeamMe
    materials.

    Args:
        input_file: The input file containing the material sections.

    Returns:
        A mapping of material IDs to BeamMe material objects.
    """

    material_id_map_all = {}

    for mat in input_file.pop("MATERIALS", []):
        mat_id = mat.pop("MAT") - 1
        if len(mat) != 1:
            raise ValueError(
                f"Could not convert the material data `{mat}` to a BeamMe material!"
            )
        mat_name, mat_data = list(mat.items())[0]
        material = _MaterialSolid(material_string=mat_name, data=mat_data)
        material_id_map_all[mat_id] = material

    nested_materials = set()
    for material in material_id_map_all.values():
        # Replace the integer IDs in the "MATIDS" list of the material with the actual
        # material objects.
        sub_materials = material.data.get("MATIDS", [])
        sub_material_ids = _np.array(sub_materials) - 1
        for i_sub_material, sub_material_id in enumerate(sub_material_ids):
            try:
                sub_materials[i_sub_material] = material_id_map_all[sub_material_id]
            except KeyError as key_exception:
                raise KeyError(
                    f"Material ID {sub_material_id} not in material_id_map_all (available "
                    f"IDs: {list(material_id_map_all.keys())})."
                ) from key_exception
            nested_materials.add(sub_material_id)

    # Get a map of all non-nested materials. We assume that only those are used as
    # materials for elements. Also, add the non-nested materials to the mesh.
    material_id_map = {
        key: val
        for key, val in material_id_map_all.items()
        if key not in nested_materials
    }

    return material_id_map
