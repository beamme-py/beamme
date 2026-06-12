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

import os as _os
from datetime import datetime as _datetime
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Callable as _Callable

from fourcipp.fourc_input import FourCInput as _FourCInput
from fourcipp.fourc_input import sort_by_section_names as _sort_by_section_names
from fourcipp.utils.not_set import NOT_SET as _NOT_SET

from beamme.core.conf import INPUT_FILE_HEADER as _INPUT_FILE_HEADER
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.four_c.element_data import FourCElementData as _FourCElementData
from beamme.four_c.input_file_dump_functions import (
    dump_mesh_representation_to_input_file_vtu as _dump_mesh_representation_to_input_file_vtu,
)
from beamme.four_c.input_file_dump_functions import (
    dump_mesh_representation_to_input_file_yaml as _dump_mesh_representation_to_input_file_yaml,
)
from beamme.four_c.input_file_dump_functions import (
    dump_mesh_to_input_file as _dump_mesh_to_input_file,
)
from beamme.utils.environment import cubitpy_is_available as _cubitpy_is_available
from beamme.utils.environment import get_application_path as _get_application_path
from beamme.utils.environment import get_git_data as _get_git_data

if _cubitpy_is_available():
    import cubitpy as _cubitpy


class InputFile:
    """An item that represents a complete 4C input file."""

    def __init__(self) -> None:
        """Initialize the input file."""

        self.fourc_input = _FourCInput()

        # Register converters to directly convert non-primitive types
        # to native Python types via the FourCIPP type converter.
        self.fourc_input.type_converter.register_numpy_types()

        # Contents of NOX xml file.
        self.nox_xml_contents = ""

        # Mesh representation for this input file.
        self.mesh_representation = _MeshRepresentation()
        self.element_type_id_to_data: dict[_Any, _FourCElementData] = {}

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
    ) -> "InputFile":
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
            _dump_mesh_to_input_file(self, mesh=object_to_add, **kwargs)

        else:
            self.fourc_input.combine_sections(object_to_add)

    def get_fourcipp_input_with_mesh(self) -> _FourCInput:
        """Return a copy of the FourCIPP input file with the contents of the
        mesh representation dumped to the yaml sections."""
        fourc_input = self.fourc_input.copy()
        _dump_mesh_representation_to_input_file_yaml(
            fourc_input,
            self.mesh_representation,
            self.element_type_id_to_data,
        )
        return fourc_input

    def dump(
        self,
        input_file_path: str | _Path,
        *,
        mesh_format: str = "yaml",
        vtu_binary: bool = False,
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
            mesh_format:
                The format in which the mesh information should be written.
                Currently, "vtu" and "yaml" are supported.
            vtu_binary:
                Only relevant if mesh_format is "vtu". If True, the vtu file will
                be written in binary format. Otherwise, it will be written in ascii format.
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

        # Base name of the input file without extension.
        if not input_file_path.name.endswith(".4C.yaml"):
            raise ValueError(
                "Input file must have a .4C.yaml extension, but got the "
                f"path {input_file_path}"
            )
        input_file_base_name = input_file_path.name.removesuffix(".4C.yaml")

        # Create a deep copy of the existing input sections - this function should not alter
        # the present instance of InputFile
        fourc_input = self.fourc_input.copy()

        # Add the mesh representation, either directly to the yaml input file or via an
        # external mesh format.
        if mesh_format == "vtu":
            vtu_file_path = input_file_path.parent / (
                input_file_base_name + ".mesh.vtu"
            )
            vtu_grid = _dump_mesh_representation_to_input_file_vtu(
                fourc_input, self.mesh_representation, self.element_type_id_to_data
            )
            # Save the grid and add the file name to the input file
            vtu_grid.save(vtu_file_path, binary=vtu_binary)
            fourc_input["STRUCTURE GEOMETRY"]["FILE"] = vtu_file_path.name
        elif mesh_format == "yaml":
            _dump_mesh_representation_to_input_file_yaml(
                fourc_input,
                self.mesh_representation,
                self.element_type_id_to_data,
            )
        else:
            raise ValueError(f"Unsupported mesh format: {mesh_format}.")

        if self.nox_xml_contents:
            if nox_xml_file is None:
                nox_xml_file = input_file_base_name + ".nox.xml"

            fourc_input["STRUCT NOX/Status Test"] = {"XML File": nox_xml_file}

            # Write the xml file to the disc.
            with open(input_file_path.parent / nox_xml_file, "w") as xml_file:
                xml_file.write(self.nox_xml_contents)

        # Add information header to the input file
        if add_header_information:
            fourc_input.combine_sections({"TITLE": self._get_header()})

        fourc_input.dump(
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

    def contains_mesh_based_geometry_exodus(self) -> bool:
        """Check if the input file contains exodus mesh-based geometry.

        Returns:
            True if the input file contains exodus mesh-based geometry, False otherwise.
        """
        structure_geometry_section = self.fourc_input.sections.get(
            "STRUCTURE GEOMETRY", None
        )
        if structure_geometry_section is not None:
            file_name = structure_geometry_section.get("FILE", None)
            if file_name is not None:
                exodus_file_name = _Path(file_name)
                if exodus_file_name.suffix.lower() in [".exo", ".e"]:
                    return True
        return False
