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

from collections import defaultdict as _defaultdict
from pathlib import Path as _Path
from typing import Tuple as _Tuple

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.node import Node as _Node
from beamme.four_c.element_solid import get_four_c_solid as _get_four_c_solid
from beamme.four_c.input_file import InputFile as _InputFile
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.four_c.material import MaterialSolid as _MaterialSolid
from beamme.utils.data_structures import (
    compare_nested_dicts_or_lists as _compare_nested_dicts_or_lists,
)
from beamme.utils.environment import cubitpy_is_available as _cubitpy_is_available

if _cubitpy_is_available():
    from cubitpy.cubit_to_fourc_input import (
        get_input_file_with_mesh as _get_input_file_with_mesh,
    )


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

    input_file = _InputFile()
    input_file.add(_get_input_file_with_mesh(cubit).sections)

    if convert_input_to_mesh:
        return _extract_mesh_sections(input_file)
    else:
        return input_file, _Mesh()


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

    if convert_input_to_mesh:
        return _extract_mesh_sections(input_file)
    else:
        return input_file, _Mesh()


def _extract_mesh_sections(input_file: _InputFile) -> _Tuple[_InputFile, _Mesh]:
    """Convert an InputFile into a native mesh by translating sections like
    materials, nodes, elements, geometry sets, and boundary conditions.

    Args:
        input_file: The input file containing 4C sections.
    Returns:
        A tuple (input_file, mesh). The input_file is modified in place to remove
        sections converted into BeamMe objects.
    """

    # function to pop sections from the input file
    _pop_section = lambda name: input_file.pop(name, [])

    # convert all sections to native objects and add to a new mesh
    mesh = _Mesh()

    # extract materials
    material_id_map_all = {}

    for mat in _pop_section("MATERIALS"):
        mat_id = mat.pop("MAT")
        if len(mat) != 1:
            raise ValueError(
                f"Could not convert the material data `{mat}` to a BeamMe material!"
            )
        mat_name, mat_data = list(mat.items())[0]
        material = _MaterialSolid(material_string=mat_name, data=mat_data)
        material_id_map_all[mat_id] = material

    nested_materials = set()
    for material in material_id_map_all.values():
        # Loop over each material and link nested materials. Also, mark nested materials
        # as they will not be added to the mesh.
        material_ids = material.data.get("MATIDS", [])
        for i_sub_material, material_id in enumerate(material_ids):
            try:
                material_ids[i_sub_material] = material_id_map_all[material_id]
            except KeyError as key_exception:
                raise KeyError(
                    f"Material ID {material_id} not in material_id_map_all (available "
                    f"IDs: {list(material_id_map_all.keys())})."
                ) from key_exception
            nested_materials.add(material_id)

    # Get a map of all non-nested materials. We assume that only those are used as
    # materials for elements. Also, add the non-nested materials to the mesh.
    material_id_map = {
        key: val
        for key, val in material_id_map_all.items()
        if key not in nested_materials
    }
    mesh.materials.extend(material_id_map.values())

    # extract nodes
    mesh.nodes = [_Node(node["COORD"]) for node in _pop_section("NODE COORDS")]

    # extract elements
    element_types: dict[type, dict] = {}
    for input_element in _pop_section("STRUCTURE ELEMENTS"):
        # At this point `input_element` contains the full element data. We can not
        # compare this directly with the other block data dictionaries, we first
        # need to extract connectivity, cell ID and material information from
        # `input_element`.
        input_element.pop("id")
        connectivity = input_element["cell"].pop("connectivity")
        four_c_type = input_element["data"].pop("type")
        material_id = input_element["data"].pop("MAT", None)

        # Loop over already found block element types and check if the current
        # element matches one of those. If not, create a new block element type
        # for this element.
        for element_type, type_data in element_types.items():
            if _compare_nested_dicts_or_lists(type_data, input_element):
                break
        else:
            element_type = _get_four_c_solid(
                _INPUT_FILE_MAPPINGS["four_c_type_to_solid_type"][four_c_type],
                n_nodes=len(connectivity),
                element_technology=input_element["data"],
            )
            element_types[element_type] = input_element

        nodes = [mesh.nodes[i - 1] for i in connectivity]
        element = element_type(nodes=nodes)
        if material_id is not None:
            element.material = material_id_map[material_id]
        mesh.elements.append(element)

    # extract geometry sets
    geometry_sets_in_sections: dict[str, dict[int, _GeometrySetNodes]] = _defaultdict(
        dict
    )

    for section_name in input_file.sections:
        if not section_name.endswith("TOPOLOGY"):
            continue

        items = _pop_section(section_name)
        if not items:
            continue

        # Find geometry type for this section
        try:
            geometry_type = _INPUT_FILE_MAPPINGS[
                "geometry_sets_condition_to_geometry_name"
            ][section_name]
        except ValueError as e:
            raise ValueError(f"Unknown geometry section: {section_name}") from e

        # Extract geometry set indices
        geom_dict: dict[int, list[int]] = _defaultdict(list)
        for entry in items:
            geom_dict[entry["d_id"]].append(entry["node_id"] - 1)

        geometry_sets_in_sections[geometry_type] = {
            gid: _GeometrySetNodes(geometry_type, nodes=[mesh.nodes[i] for i in ids])
            for gid, ids in geom_dict.items()
        }

        mesh.geometry_sets[geometry_type] = list(
            geometry_sets_in_sections[geometry_type].values()
        )

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
        for bc_data in _pop_section(section_name):
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
