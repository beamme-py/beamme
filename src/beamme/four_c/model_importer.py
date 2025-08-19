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

from pathlib import Path as _Path
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Union as _Union

import beamme.core.conf as _conf
from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.element_volume import VolumeHEX8 as _VolumeHEX8
from beamme.core.element_volume import VolumeHEX20 as _VolumeHEX20
from beamme.core.element_volume import VolumeHEX27 as _VolumeHEX27
from beamme.core.element_volume import VolumeTET4 as _VolumeTET4
from beamme.core.element_volume import VolumeTET10 as _VolumeTET10
from beamme.core.element_volume import VolumeWEDGE6 as _VolumeWEDGE6
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.node import Node as _Node
from beamme.four_c.element_volume import SolidRigidSphere as _SolidRigidSphere
from beamme.four_c.input_file import InputFile as _InputFile
from beamme.four_c.input_file import (
    get_geometry_set_indices_from_section as _get_geometry_set_indices_from_section,
)
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


def _element_from_dict(nodes: _List[_Node], element: dict, material_id_map: dict):
    """Create a solid element from a dictionary from a 4C input file.

    Args:
        nodes: A list of nodes that are part of the element.
        element: A dictionary with the element data.
        material_id_map: A map between the global material ID and the BeamMe material object.
    Returns:
        A solid element object.
    """

    # Depending on the number of nodes chose which solid element to return.
    # TODO reuse element_type_to_four_c_string from beamme.core.element_volume
    element_type = {
        "HEX8": _VolumeHEX8,
        "HEX20": _VolumeHEX20,
        "HEX27": _VolumeHEX27,
        "TET4": _VolumeTET4,
        "TET10": _VolumeTET10,
        "WEDGE6": _VolumeWEDGE6,
        "POINT1": _SolidRigidSphere,
    }

    if element["cell"]["type"] not in element_type:
        raise TypeError(
            f"Could not create a BeamMe element for {element['data']['type']} {element['cell']['type']}!"
        )

    created_element = element_type[element["cell"]["type"]](
        nodes=nodes, data=element["data"]
    )
    # Check if we have to link this element to a material object (rigid spheres do not
    # have a material).
    if "MAT" in created_element.data:
        created_element.data["MAT"] = material_id_map[created_element.data["MAT"]]
    return created_element


def _boundary_condition_from_dict(
    geometry_set: _GeometrySetNodes,
    bc_key: _Union[_conf.BoundaryCondition, str],
    data: _Dict,
) -> _BoundaryConditionBase:
    """This function acts as a factory and creates the correct boundary
    condition object from a dictionary parsed from an input file."""

    del data["E"]

    if bc_key in (
        _bme.bc.dirichlet,
        _bme.bc.neumann,
        _bme.bc.locsys,
        _bme.bc.beam_to_solid_surface_meshtying,
        _bme.bc.beam_to_solid_surface_contact,
        _bme.bc.beam_to_solid_volume_meshtying,
    ) or isinstance(bc_key, str):
        return _BoundaryCondition(geometry_set, data, bc_type=bc_key)
    elif bc_key is _bme.bc.point_coupling:
        return _Coupling(geometry_set, bc_key, data, check_overlapping_nodes=False)
    else:
        raise ValueError("Got unexpected boundary condition!")


def _get_yaml_geometry_sets(
    nodes: _List[_Node], geometry_key: _conf.Geometry, section_list: _List
) -> _Dict[int, _GeometrySetNodes]:
    """Add sets of points, lines, surfaces or volumes to the object."""

    # Create the individual geometry sets
    geometry_set_dict = _get_geometry_set_indices_from_section(section_list)
    geometry_sets_in_this_section = {}
    for geometry_set_id, node_ids in geometry_set_dict.items():
        geometry_sets_in_this_section[geometry_set_id] = _GeometrySetNodes(
            geometry_key, nodes=[nodes[node_id] for node_id in node_ids]
        )
    return geometry_sets_in_this_section


def _extract_mesh_sections(input_file: _InputFile) -> _Tuple[_InputFile, _Mesh]:
    """Convert an existing input file to a BeamMe mesh with mesh items, e.g.,
    nodes, elements, element sets, node sets, boundary conditions, materials.

    Args:
        input_file: The input file object that contains the sections to be
            converted to a BeamMe mesh.
    Returns:
        A tuple with the input file and the mesh. The input file will be
            modified to remove the sections that have been converted to a
            BeamMe mesh.
    """

    def _get_section_items(section_name):
        """Return the items in a given section.

        Since we will add the created BeamMe objects to the mesh, we
        delete them from the plain data storage to avoid having
        duplicate entries.
        """

        if section_name in input_file:
            return input_file.pop(section_name)
        else:
            return []

    # Go through all sections that have to be converted to full BeamMe objects
    mesh = _Mesh()

    # Add materials
    material_id_map = {}
    if "MATERIALS" in input_file:
        for material_in_input_file in input_file.pop("MATERIALS"):
            material_id = material_in_input_file.pop("MAT")
            if not len(material_in_input_file) == 1:
                raise ValueError(
                    f"Could not convert the material data {material_in_input_file} to a BeamMe material."
                )
            material_string = list(material_in_input_file.keys())[0]
            material_data = list(material_in_input_file.values())[0]
            material = _MaterialSolid(
                material_string=material_string, data=material_data
            )
            material_id_map[material_id] = material
            mesh.add(material)

    # Add nodes
    if "NODE COORDS" in input_file:
        mesh.nodes = [_Node(node["COORD"]) for node in input_file.pop("NODE COORDS")]

    # Add elements
    if "STRUCTURE ELEMENTS" in input_file:
        for element_in_input_file in input_file.pop("STRUCTURE ELEMENTS"):
            nodes = [
                mesh.nodes[node_id - 1]
                for node_id in element_in_input_file["cell"]["connectivity"]
            ]
            mesh.elements.append(
                _element_from_dict(
                    nodes=nodes,
                    element=element_in_input_file,
                    material_id_map=material_id_map,
                )
            )

    # Add geometry sets
    geometry_sets_in_sections: dict[str, dict[int, _GeometrySetNodes]] = {
        key: {} for key in _bme.geo
    }
    for section_name in input_file.sections.keys():
        if section_name.endswith("TOPOLOGY"):
            section_items = _get_section_items(section_name)
            if len(section_items) > 0:
                # Get the geometry key for this set
                for key, value in _INPUT_FILE_MAPPINGS["geometry_sets"].items():
                    if value == section_name:
                        geometry_key = key
                        break
                else:
                    raise ValueError(f"Could not find the set {section_name}")
                geometry_sets_in_section = _get_yaml_geometry_sets(
                    mesh.nodes, geometry_key, section_items
                )
                geometry_sets_in_sections[geometry_key] = geometry_sets_in_section
                mesh.geometry_sets[geometry_key] = list(
                    geometry_sets_in_section.values()
                )

    # Add boundary conditions
    for (
        bc_key,
        geometry_key,
    ), section_name in _INPUT_FILE_MAPPINGS["boundary_conditions"].items():
        for item in _get_section_items(section_name):
            geometry_set_id = item["E"]
            geometry_set = geometry_sets_in_sections[geometry_key][geometry_set_id]
            mesh.boundary_conditions.append(
                (bc_key, geometry_key),
                _boundary_condition_from_dict(geometry_set, bc_key, item),
            )

    return input_file, mesh
