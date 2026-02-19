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
"""This module defines the classes that are used to create an input file for
4C."""

from __future__ import annotations as _annotations

import os as _os
from datetime import datetime as _datetime
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Callable as _Callable
from typing import List as _List

from fourcipp.fourc_input import FourCInput as _FourCInput
from fourcipp.fourc_input import sort_by_section_names as _sort_by_section_names
from fourcipp.utils.not_set import NOT_SET as _NOT_SET

from beamme.core.conf import INPUT_FILE_HEADER as _INPUT_FILE_HEADER
from beamme.core.conf import bme as _bme
from beamme.core.function import Function as _Function
from beamme.core.material import Material as _Material
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.node import Node as _Node
from beamme.four_c.input_file_dump_item import dump_elements as _dump_elements
from beamme.four_c.input_file_dump_item import dump_geometry_sets as _dump_geometry_sets
from beamme.four_c.input_file_dump_item import dump_item_to_list as _dump_item_to_list
from beamme.four_c.input_file_dump_item import (
    dump_item_to_section as _dump_item_to_section,
)
from beamme.four_c.input_file_dump_item import dump_nodes as _dump_nodes
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.utils.environment import cubitpy_is_available as _cubitpy_is_available
from beamme.utils.environment import get_application_path as _get_application_path
from beamme.utils.environment import get_git_data as _get_git_data

if _cubitpy_is_available():
    import cubitpy as _cubitpy


class InputFile:
    """An item that represents a complete 4C input file."""

    def __init__(self):
        """Initialize the input file."""

        self.fourc_input = _FourCInput()

        # Contents of NOX xml file.
        self.nox_xml_contents = ""

        # Register converters to directly convert non-primitive types
        # to native Python types via the FourCIPP type converter.
        self.fourc_input.type_converter.register_numpy_types()
        self.fourc_input.type_converter.register_type(
            (_Function, _Material, _Node), lambda converter, obj: obj.i_global + 1
        )

    def __contains__(self, key: str) -> bool:
        """Contains function.

        Allows to use the `in` operator.

        Args:
            key: Section name to check if it is set

        Returns:
            True if section is set
        """

        return key in self.fourc_input

    def __setitem__(self, key: str, value: _Any) -> None:
        """Set section.

        Args:
            key: Section name
            value: Section entry
        """

        self.fourc_input[key] = value

    def __getitem__(self, key: str) -> _Any:
        """Get section of input file.

        Allows to use the indexing operator.

        Args:
            key: Section name to get

        Returns:
            The section content
        """

        return self.fourc_input[key]

    @classmethod
    def from_4C_yaml(
        cls, input_file_path: str | _Path, header_only: bool = False
    ) -> InputFile:
        """Load 4C yaml file.

        Args:
            input_file_path: Path to yaml file
            header_only: Only extract header, i.e., all sections except the legacy ones

        Returns:
            Initialised object
        """

        obj = cls()
        obj.fourc_input = _FourCInput.from_4C_yaml(input_file_path, header_only)
        return obj

    @property
    def sections(self) -> dict:
        """All the set sections.

        Returns:
            dict: Set sections
        """

        return self.fourc_input.sections

    def pop(self, key: str, default_value: _Any = _NOT_SET) -> _Any:
        """Pop section of input file.

        Args:
            key: Section name to pop

        Returns:
            The section content
        """

        return self.fourc_input.pop(key, default_value)

    def add(self, object_to_add, **kwargs):
        """Add a mesh or a dictionary to the input file.

        Args:
            object: The object to be added. This can be a mesh or a dictionary.
            **kwargs: Additional arguments to be passed to the add method.
        """

        if isinstance(object_to_add, _Mesh):
            self.add_mesh_to_input_file(mesh=object_to_add, **kwargs)

        else:
            self.fourc_input.combine_sections(object_to_add)

    def dump(
        self,
        input_file_path: str | _Path,
        *,
        nox_xml_file: str | None = None,
        add_header_default: bool = True,
        add_header_information: bool = True,
        add_footer_application_script: bool = True,
        validate=True,
        validate_sections_only: bool = False,
        sort_function: _Callable[[dict], dict] | None = _sort_by_section_names,
        fourcipp_yaml_style: bool = True,
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
            validate:
                Validate if the created input file is compatible with 4C with FourCIPP.
            validate_sections_only:
                Validate each section independently. Required sections are no longer
                required, but the sections must be valid.
            sort_function:
                A function which sorts the sections of the input file.
            fourcipp_yaml_style:
                If True, the input file is written in the fourcipp yaml style.
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

        self.fourc_input.dump(
            input_file_path=input_file_path,
            validate=validate,
            validate_sections_only=validate_sections_only,
            convert_to_native_types=False,  # conversion already happens during add()
            sort_function=sort_function,
            use_fourcipp_yaml_style=fourcipp_yaml_style,
        )

        if add_header_default or add_footer_application_script:
            with open(input_file_path, "r") as input_file:
                lines = input_file.readlines()

                if add_header_default:
                    lines = ["# " + line + "\n" for line in _INPUT_FILE_HEADER] + lines

                if add_footer_application_script:
                    application_path = _get_application_path()
                    if application_path is not None:
                        lines += self._get_application_script(application_path)

                with open(input_file_path, "w") as input_file:
                    input_file.writelines(lines)

    def add_mesh_to_input_file(self, mesh: _Mesh) -> None:
        """Add a mesh to the input file.

        Args:
            mesh: The mesh to be added to the input file.
        """

        if _bme.check_overlapping_elements:
            mesh.check_overlapping_elements()

        # Compute geometry-set starting indices
        start_indices_geometry_set = {
            geometry_type: max(
                (entry["d_id"] for entry in self.sections.get(section_name, [])),
                default=0,
            )
            for geometry_type, section_name in _INPUT_FILE_MAPPINGS[
                "geometry_sets_geometry_to_condition_name"
            ].items()
        }

        # Determine global start indices
        start_index_nodes = len(self.sections.get("NODE COORDS", []))

        start_index_elements = sum(
            len(self.sections.get(section, []))
            for section in ("FLUID ELEMENTS", "STRUCTURE ELEMENTS")
        )

        start_index_functions = max(
            (
                int(section.split("FUNCT")[-1])
                for section in self.sections
                if section.startswith("FUNCT")
            ),
            default=0,
        )

        start_index_materials = max(
            (material["MAT"] for material in self.sections.get("MATERIALS", [])),
            default=0,
        )  # materials imported from YAML may have arbitrary numbering

        # Get the BeamMe mesh representation of the mesh
        grid, mesh_sets, all_materials, cell_block_id_to_element_type_and_material = (
            mesh.get_mesh_representation()
        )

        # Dump mesh to input file
        def _dump(section_name: str, items: _List) -> None:
            """Dump list of items to a section in the input file.

            Args:
                section_name: Name of the section
                items: List of items to be dumped
            """
            if not items:  # do not write empty sections
                return
            dumped: list[_Any] = []
            for item in items:
                _dump_item_to_list(dumped, item)

            # Go through FourCIPP to convert to native types
            # TODO this can be simplified/removed by using an internal type converter
            if section_name in self.sections:
                existing = self.pop(section_name)
                existing.extend(dumped)
                dumped = existing

            self.add({section_name: dumped})

        #   Materials
        _dump("MATERIALS", all_materials)

        #   Functions
        for function in mesh.functions:
            self.add({f"FUNCT{function.i_global + 1}": function.data})

        #   Couplings
        #     If there are couplings in the mesh, set the link between the nodes
        #     and elements, so the couplings can decide which DOFs they couple,
        #     depending on the type of the connected beam element.
        if any(
            mesh.boundary_conditions.get((key, _bme.geo.point), [])
            for key in (_bme.bc.point_coupling, _bme.bc.point_coupling_penalty)
        ):
            mesh.set_node_links()

        #   Boundary conditions
        for (bc_key, geometry_key), bc_list in mesh.boundary_conditions.items():
            if bc_list:
                section = (
                    bc_key
                    if isinstance(bc_key, str)
                    else _INPUT_FILE_MAPPINGS["boundary_conditions"][
                        (bc_key, geometry_key)
                    ]
                )
                _dump(section, bc_list)

        #   Additional element sections (NURBS etc.)
        for element in mesh.elements:
            _dump_item_to_section(self, element)

        def dump_no_type_converter(
            section_name,
            items,
        ):
            """TODO."""
            if section_name not in self.fourc_input.sections:
                self.add({section_name: []})
            self.fourc_input.sections[section_name].extend(items)

        #   Geometry sets
        for geometry_key, geometry_sets in mesh_sets.items():
            dumped_geometry_sets = _dump_geometry_sets(
                geometry_sets,
                grid,
                geometry_key,
                start_indices_geometry_set[geometry_key],
                start_index_nodes,
            )
            if len(dumped_geometry_sets) > 0:
                dump_no_type_converter(
                    _INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"][
                        geometry_key
                    ],
                    dumped_geometry_sets,
                )

        #   Nodes
        dump_no_type_converter("NODE COORDS", _dump_nodes(grid, start_index_nodes))
        #   Elements
        dump_no_type_converter(
            "STRUCTURE ELEMENTS",
            _dump_elements(
                mesh,
                grid,
                cell_block_id_to_element_type_and_material,
                start_index_elements,
                start_index_nodes,
            ),
        )

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
        application_path = _get_application_path()
        if application_path is not None:
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
