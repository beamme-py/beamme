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
"""This module defines the classes that are used to create an input file for
4C."""

import os as _os
import sys as _sys
from datetime import datetime as _datetime
from pathlib import Path as _Path
from typing import Dict as _Dict
from typing import List as _List

from fourcipp.fourc_input import FourCInput as _FourCInput

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.conf import INPUT_FILE_HEADER as _INPUT_FILE_HEADER
from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.function import Function as _Function
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.material import Material as _Material
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.node import Node as _Node
from beamme.core.nurbs_patch import NURBSPatch as _NURBSPatch
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.utils.environment import cubitpy_is_available as _cubitpy_is_available
from beamme.utils.environment import get_git_data as _get_git_data

if _cubitpy_is_available():
    import cubitpy as _cubitpy


def get_geometry_set_indices_from_section(
    section_list: _List, *, append_node_ids: bool = True
) -> _Dict:
    """Return a dictionary with the geometry set ID as keys and the node IDs as
    values.

    Args:
        section_list: A list with the legacy strings for the geometry pair
        append_node_ids: If the node IDs shall be appended, or only the
            dict with the keys should be returned.
    """

    geometry_set_dict: _Dict[int, _List[int]] = {}
    for entry in section_list:
        id_geometry_set = entry["d_id"]
        index_node = entry["node_id"] - 1
        if id_geometry_set not in geometry_set_dict:
            geometry_set_dict[id_geometry_set] = []
        if append_node_ids:
            geometry_set_dict[id_geometry_set].append(index_node)

    return geometry_set_dict


def _dump_coupling(coupling):
    """Return the input file representation of the coupling condition."""

    # TODO: Move this to a better place / gather all dump functions for general
    # BeamMe items in a file or so.

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
            rotvec = {element.rotvec for element in connected_elements}
            if len(rotvec) > 1 or not rotvec.pop():
                raise TypeError(
                    "Couplings for Kirchhoff beams and rotvec==False not yet implemented."
                )

        data = element_type.get_coupling_dict(coupling.data)

    return {"E": coupling.geometry_set.i_global + 1, **data}


class InputFile(_FourCInput):
    """An item that represents a complete 4C input file."""

    def __init__(self, sections=None):
        """Initialize the input file."""

        super().__init__(sections=sections)

        # Contents of NOX xml file.
        self.nox_xml_contents = ""

        # Register converters to directly convert non-primitive types
        # to native Python types via the FourCIPP type converter.
        self.type_converter.register_numpy_types()
        self.type_converter.register_type(
            (_Function, _Material, _Node), lambda converter, obj: obj.i_global + 1
        )

    def add(self, object_to_add, **kwargs):
        """Add a mesh or a dictionary to the input file.

        Args:
            object: The object to be added. This can be a mesh or a dictionary.
            **kwargs: Additional arguments to be passed to the add method.
        """

        if isinstance(object_to_add, _Mesh):
            self.add_mesh_to_input_file(mesh=object_to_add, **kwargs)

        else:
            super().combine_sections(object_to_add)

    def dump(
        self,
        input_file_path: str | _Path,
        *,
        nox_xml_file: str | None = None,
        add_header_default: bool = True,
        add_header_information: bool = True,
        add_footer_application_script: bool = True,
        sort_sections=False,
        validate=True,
        validate_sections_only: bool = False,
    ):
        """Write the input file to disk.

        Args:
            input_file_path:
                Path to the input file that should be created.
            nox_xml_file:
                If this is a string, the NOX xml file will be created with this
                name. If this is None, the NOX xml file will be created with the
                name of the input file with the extension ".nox.xml".
            add_header_default:
                Prepend the default header comment to the input file.
            add_header_information:
                If the information header should be exported to the input file
                Contains creation date, git details of BeamMe, CubitPy and
                original application which created the input file if available.
            add_footer_application_script:
                Append the application script which creates the input files as a
                comment at the end of the input file.
            sort_sections:
                Sort sections alphabetically with FourCIPP.
            validate:
                Validate if the created input file is compatible with 4C with FourCIPP.
            validate_sections_only:
                Validate each section independently. Required sections are no longer
                required, but the sections must be valid.
        """

        # Make sure the given input file is a Path instance.
        input_file_path = _Path(input_file_path)

        if self.nox_xml_contents:
            if nox_xml_file is None:
                nox_xml_file = input_file_path.name.split(".")[0] + ".nox.xml"

            self["STRUCT NOX/Status Test"] = {"XML File": nox_xml_file}

            # Write the xml file to the disc.
            with open(input_file_path.parent / nox_xml_file, "w") as xml_file:
                xml_file.write(self.nox_xml_contents)

        # Add information header to the input file
        if add_header_information:
            self.add({"TITLE": self._get_header()})

        super().dump(
            input_file_path=input_file_path,
            sort_sections=sort_sections,
            validate=validate,
            validate_sections_only=validate_sections_only,
            convert_to_native_types=False,  # conversion already happens during add()
        )

        if add_header_default or add_footer_application_script:
            with open(input_file_path, "r") as input_file:
                lines = input_file.readlines()

                if add_header_default:
                    lines = ["# " + line + "\n" for line in _INPUT_FILE_HEADER] + lines

                if add_footer_application_script:
                    application_path = _Path(_sys.argv[0]).resolve()
                    lines += self._get_application_script(application_path)

                with open(input_file_path, "w") as input_file:
                    input_file.writelines(lines)

    def add_mesh_to_input_file(self, mesh: _Mesh):
        """Add a mesh to the input file.

        Args:
            mesh: The mesh to be added to the input file.
        """

        # Perform some checks on the mesh.
        if _bme.check_overlapping_elements:
            mesh.check_overlapping_elements()

        def _get_global_start_geometry_set(dictionary):
            """Get the indices for the first "real" BeamMe geometry sets."""

            start_indices_geometry_set = {}
            for geometry_type, section_name in _INPUT_FILE_MAPPINGS[
                "geometry_sets"
            ].items():
                max_geometry_set_id = 0
                if section_name in dictionary:
                    section_list = dictionary[section_name]
                    if len(section_list) > 0:
                        geometry_set_dict = get_geometry_set_indices_from_section(
                            section_list, append_node_ids=False
                        )
                        max_geometry_set_id = max(geometry_set_dict.keys())
                start_indices_geometry_set[geometry_type] = max_geometry_set_id
            return start_indices_geometry_set

        def _get_global_start_node():
            """Get the index for the first "real" BeamMe node."""

            return len(self.sections.get("NODE COORDS", []))

        def _get_global_start_element():
            """Get the index for the first "real" BeamMe element."""

            return sum(
                len(self.sections.get(section, []))
                for section in ["FLUID ELEMENTS", "STRUCTURE ELEMENTS"]
            )

        def _get_global_start_material():
            """Get the index for the first "real" BeamMe material.

            We have to account for materials imported from yaml files
            that have arbitrary numbering.
            """

            # Get the maximum material index in materials imported from a yaml file
            max_material_id = 0
            section_name = "MATERIALS"
            if section_name in self.sections:
                for material in self.sections[section_name]:
                    max_material_id = max(max_material_id, material["MAT"])
            return max_material_id

        def _get_global_start_function():
            """Get the index for the first "real" BeamMe function."""

            max_function_id = 0
            for section_name in self.sections.keys():
                if section_name.startswith("FUNCT"):
                    max_function_id = max(
                        max_function_id, int(section_name.split("FUNCT")[-1])
                    )
            return max_function_id

        def _set_i_global(data_list, *, start_index=0):
            """Set i_global in every item of data_list."""

            # A check is performed that every entry in data_list is unique.
            if len(data_list) != len(set(data_list)):
                raise ValueError("Elements in data_list are not unique!")

            # Set the values for i_global.
            for i, item in enumerate(data_list):
                item.i_global = i + start_index

        def _set_i_global_elements(element_list, *, start_index=0):
            """Set i_global in every item of element_list."""

            # A check is performed that every entry in element_list is unique.
            if len(element_list) != len(set(element_list)):
                raise ValueError("Elements in element_list are not unique!")

            # Set the values for i_global.
            i = start_index
            i_nurbs_patch = 0
            for item in element_list:
                # As a NURBS patch can be defined with more elements, an offset is applied to the
                # rest of the items
                item.i_global = i
                if isinstance(item, _NURBSPatch):
                    item.n_nurbs_patch = i_nurbs_patch + 1
                    offset = item.get_number_elements()
                    i += offset
                    i_nurbs_patch += 1
                else:
                    i += 1

        def _dump_mesh_items(section_name, data_list):
            """Output a section name and apply either the default dump or the
            specialized the dump_to_list for each list item."""

            # Do not write section if no content is available
            if len(data_list) == 0:
                return

            list = []

            for item in data_list:
                if (
                    isinstance(item, _GeometrySet)
                    or isinstance(item, _GeometrySetNodes)
                    or isinstance(item, _NURBSPatch)
                ):
                    list.extend(item.dump_to_list())
                elif hasattr(item, "dump_to_list"):
                    list.append(item.dump_to_list())
                elif isinstance(item, _BoundaryCondition):
                    list.append(
                        {
                            "E": item.geometry_set.i_global + 1,
                            **item.data,
                        }
                    )

                elif isinstance(item, _Coupling):
                    list.append(_dump_coupling(item))
                else:
                    raise TypeError(f"Could not dump {item}")

            # If section already exists, retrieve from input file and
            # add newly. We always need to go through fourcipp to convert
            # the data types correctly.
            if section_name in self.sections:
                existing_entries = self.pop(section_name)
                existing_entries.extend(list)
                list = existing_entries

            self.add({section_name: list})

        # Add sets from couplings and boundary conditions to a temp container.
        mesh.unlink_nodes()
        start_indices_geometry_set = _get_global_start_geometry_set(self.sections)
        mesh_sets = mesh.get_unique_geometry_sets(
            geometry_set_start_indices=start_indices_geometry_set
        )

        # Assign global indices to all entries.
        start_index_nodes = _get_global_start_node()
        _set_i_global(mesh.nodes, start_index=start_index_nodes)

        start_index_elements = _get_global_start_element()
        _set_i_global_elements(mesh.elements, start_index=start_index_elements)

        start_index_materials = _get_global_start_material()
        _set_i_global(mesh.materials, start_index=start_index_materials)

        start_index_functions = _get_global_start_function()
        _set_i_global(mesh.functions, start_index=start_index_functions)

        # Add material data to the input file.
        _dump_mesh_items("MATERIALS", mesh.materials)

        # Add the functions.
        for function in mesh.functions:
            self.add({f"FUNCT{function.i_global + 1}": function.data})

        # If there are couplings in the mesh, set the link between the nodes
        # and elements, so the couplings can decide which DOFs they couple,
        # depending on the type of the connected beam element.
        def get_number_of_coupling_conditions(key):
            """Return the number of coupling conditions in the mesh."""
            if (key, _bme.geo.point) in mesh.boundary_conditions.keys():
                return len(mesh.boundary_conditions[key, _bme.geo.point])
            else:
                return 0

        if (
            get_number_of_coupling_conditions(_bme.bc.point_coupling)
            + get_number_of_coupling_conditions(_bme.bc.point_coupling_penalty)
            > 0
        ):
            mesh.set_node_links()

        # Add the boundary conditions.
        for (bc_key, geom_key), bc_list in mesh.boundary_conditions.items():
            if len(bc_list) > 0:
                section_name = (
                    bc_key
                    if isinstance(bc_key, str)
                    else _INPUT_FILE_MAPPINGS["boundary_conditions"][(bc_key, geom_key)]
                )
                _dump_mesh_items(section_name, bc_list)

        # Add additional element sections, e.g., for NURBS knot vectors.
        for element in mesh.elements:
            element.dump_element_specific_section(self)

        # Add the geometry sets.
        for geom_key, item in mesh_sets.items():
            if len(item) > 0:
                _dump_mesh_items(_INPUT_FILE_MAPPINGS["geometry_sets"][geom_key], item)

        # Add the nodes and elements.
        _dump_mesh_items("NODE COORDS", mesh.nodes)
        _dump_mesh_items("STRUCTURE ELEMENTS", mesh.elements)
        # TODO: reset all links and counters set in this method.

    def _get_header(self) -> dict:
        """Return the information header for the current BeamMe run.

        Returns:
            A dictionary with the header information.
        """

        header: dict = {"BeamMe": {}}

        header["BeamMe"]["creation_date"] = _datetime.now().isoformat(
            sep=" ", timespec="seconds"
        )

        # application which created the input file
        application_path = _Path(_sys.argv[0]).resolve()
        header["BeamMe"]["Application"] = {"path": str(application_path)}

        application_git_sha, application_git_date = _get_git_data(
            application_path.parent
        )
        if application_git_sha is not None and application_git_date is not None:
            header["BeamMe"]["Application"].update(
                {
                    "git_sha": application_git_sha,
                    "git_date": application_git_date,
                }
            )

        # BeamMe information
        beamme_git_sha, beamme_git_date = _get_git_data(
            _Path(__file__).resolve().parent
        )
        if beamme_git_sha is not None and beamme_git_date is not None:
            header["BeamMe"]["BeamMe"] = {
                "git_SHA": beamme_git_sha,
                "git_date": beamme_git_date,
            }

        # CubitPy information
        if _cubitpy_is_available():
            cubitpy_git_sha, cubitpy_git_date = _get_git_data(
                _os.path.dirname(_cubitpy.__file__)
            )

            if cubitpy_git_sha is not None and cubitpy_git_date is not None:
                header["BeamMe"]["CubitPy"] = {
                    "git_SHA": cubitpy_git_sha,
                    "git_date": cubitpy_git_date,
                }

        return header

    def _get_application_script(self, application_path: _Path) -> list[str]:
        """Get the script that created this input file.

        Args:
            application_path: Path to the script that created this input file.
        Returns:
            A list of strings with the script that created this input file.
        """

        application_script_lines = [
            "# Application script which created this input file:\n"
        ]

        with open(application_path) as script_file:
            application_script_lines.extend("# " + line for line in script_file)

        return application_script_lines
