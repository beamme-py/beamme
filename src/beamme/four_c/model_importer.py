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
from beamme.four_c.input_file import InputFile as _InputFile
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.four_c.material import MaterialSolid as _MaterialSolid
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

    input_file = _InputFile(sections=_get_input_file_with_mesh(cubit).sections)

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
    material_id_map = {}

    for mat in _pop_section("MATERIALS"):
        mat_id = mat.pop("MAT")
        if len(mat) != 1:
            raise ValueError(
                f"Could not convert the material data `{mat}` to a BeamMe material!"
            )
        mat_name, mat_data = list(mat.items())[0]
        material = _MaterialSolid(material_string=mat_name, data=mat_data)
        mesh.add(material)
        material_id_map[mat_id] = material

    # extract nodes
    mesh.nodes = [_Node(node["COORD"]) for node in _pop_section("NODE COORDS")]

    # extract elements
    for input_element in _pop_section("STRUCTURE ELEMENTS"):
        if (
            input_element["cell"]["type"]
            not in _INPUT_FILE_MAPPINGS["element_four_c_string_to_type"]
        ):
            raise TypeError(
                f"Could not create a BeamMe element for `{input_element['data']['type']}` `{input_element['cell']['type']}`!"
            )
        nodes = [mesh.nodes[i - 1] for i in input_element["cell"]["connectivity"]]
        element_class = _INPUT_FILE_MAPPINGS["element_four_c_string_to_type"][
            input_element["cell"]["type"]
        ]
        element = element_class(nodes=nodes, data=input_element["data"])
        if "MAT" in element.data:
            element.data["MAT"] = material_id_map[element.data["MAT"]]
        mesh.elements.append(element)

    # extract geometry sets
    geometry_sets_in_sections: dict[str, dict[int, _GeometrySetNodes]] = {
        key: {} for key in _bme.geo
    }

    for section_name in input_file.sections:
        if not section_name.endswith("TOPOLOGY"):
            continue

        items = _pop_section(section_name)
        if not items:
            continue

        # Find geometry key for this section
        geom_section_keys = list(
            _INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"].keys()
        )
        geom_section_values = list(
            _INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"].values()
        )

        try:
            geometry_key = geom_section_keys[geom_section_values.index(section_name)]
        except ValueError as e:
            raise ValueError(f"Unknown geometry section: {section_name}") from e

        # Extract geometry set indices
        geom_dict: dict[int, list[int]] = _defaultdict(list)
        for entry in items:
            geom_dict[entry["d_id"]].append(entry["node_id"] - 1)

        geometry_sets_in_sections[geometry_key] = {
            gid: _GeometrySetNodes(geometry_key, nodes=[mesh.nodes[i] for i in ids])
            for gid, ids in geom_dict.items()
        }

        mesh.geometry_sets[geometry_key] = list(
            geometry_sets_in_sections[geometry_key].values()
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

    for (bc_key, geometry_key), section_name in _INPUT_FILE_MAPPINGS[
        "boundary_conditions"
    ].items():
        for bc_data in _pop_section(section_name):
            geometry_set = geometry_sets_in_sections[geometry_key][bc_data.pop("E")]

            bc_obj: _BoundaryConditionBase

            if bc_key in _standard_bc_types or isinstance(bc_key, str):
                bc_obj = _BoundaryCondition(geometry_set, bc_data, bc_type=bc_key)
            elif bc_key is _bme.bc.point_coupling:
                bc_obj = _Coupling(
                    geometry_set, bc_key, bc_data, check_overlapping_nodes=False
                )
            else:
                raise ValueError(f"Unexpected boundary condition: {bc_key}")

            mesh.boundary_conditions.append((bc_key, geometry_key), bc_obj)

    return input_file, mesh
