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
"""Testing framework infrastructure for result comparison."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import vtk
import xmltodict
from vistools.vtk.compare_grids import compare_grids

from beamme.core.mesh import Mesh
from beamme.four_c.input_file import InputFile
from beamme.four_c.input_file_dump_item import (
    dump_mesh_representation_to_input_file_legacy,
)
from beamme.utils.data_structures import compare_nested_dicts_or_lists

# GLOBAL DEFAULT TEST TOLERANCES
RELATIVE_TOLERANCE = 1e-13
ABSOLUTE_TOLERANCE = 1e-13


@pytest.fixture(scope="function")
def assert_results_close(tmp_path, current_test_name) -> Callable:
    """Return function to compare either string or files.

    Necessary to enable the function call through pytest fixtures.

    Args:
        tmp_path: Temporary path to write file if assertion fails.
        current_test_name: Name of the current test to create file
            if assertion fails.

    Returns:
        Function to compare results.
    """

    def _assert_results_close(
        reference: (
            Path | str | int | float | dict | list | np.ndarray | InputFile | Mesh
        ),
        result: Path | str | int | float | dict | list | np.ndarray | InputFile | Mesh,
        rtol: float = RELATIVE_TOLERANCE,
        atol: float = ABSOLUTE_TOLERANCE,
    ) -> None:
        """Comparison between reference and result with relative or absolute
        tolerance.

        If the comparison fails, an assertion is raised.

        Args:
            reference: The reference data.
            result: The result data.
            rtol: The relative tolerance.
            atol: The absolute tolerance.
        """

        # special case to compare vtk files
        if (
            isinstance(reference, Path)
            and reference.suffix in {".vtk", ".vtu"}
            and isinstance(result, Path)
            and result.suffix in {".vtk", ".vtu"}
        ):
            compare_vtk_files(reference, result, rtol, atol)
            return

        # convert all other types into dicts/lists
        converted_reference = convert_to_primitive_type(reference)
        converted_result = convert_to_primitive_type(result)

        try:
            compare_nested_dicts_or_lists(
                converted_reference,
                converted_result,
                rtol=rtol,
                atol=atol,
                allow_int_vs_float_comparison=True,
                custom_compare=lambda obj, ref_obj: custom_fourcipp_comparison(
                    obj, ref_obj, rtol=rtol, atol=atol
                ),
                raise_on_mismatch=True,
            )
        except AssertionError as error:
            handle_failed_assertion(tmp_path, current_test_name, reference, result)
            raise error

    return _assert_results_close


def compare_vtk_files(reference: Path, result: Path, rtol: float, atol: float) -> None:
    """Compare two VTK files for equality within a given tolerance.

    Args:
        reference: The path to the reference VTK file.
        result: The path to the result VTK file to be compared.
        rtol: The relative tolerance.
        atol: The absolute tolerance.
    """

    def get_vtk(path: Path) -> vtk.vtkDataObject:
        """Return vtk data object for given vtk file.

        Args:
            path: Path to .vtu/.vtk file.

        Returns:
            VTK data object.
        """

        reader = vtk.vtkXMLGenericDataObjectReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()

    compare = compare_grids(
        get_vtk(reference), get_vtk(result), output=True, rtol=rtol, atol=atol
    )

    if not compare[0]:
        raise AssertionError("\n".join(compare[1]))


def convert_to_primitive_type(
    obj: str | int | float | dict | list | np.ndarray | Path | Mesh | InputFile,
) -> int | float | dict | list | np.ndarray:
    """Convert the given object to a primitive type (dict, list, numpy array).

    Args:
        obj: The object to convert.

    Returns:
        The raw data (either a dictionary, list, numpy array).
    """

    if isinstance(obj, (int, float, dict, list, np.ndarray)):
        return obj

    if isinstance(obj, Path):
        if obj.suffix == ".xml" or obj.suffix == ".pvd":
            return xmltodict.parse(obj.read_text(encoding="utf-8"))

        elif obj.suffix == ".json":
            return json.loads(obj.read_text(encoding="utf-8"))

        elif obj.name.endswith(".4C.yaml"):
            # return sections in next step
            obj = InputFile().from_4C_yaml(input_file_path=obj)

        elif obj.suffix == ".inp":
            # Abaqus input files are read as strings
            # split into list of fragments in next step
            obj = obj.read_text(encoding="utf-8")

    if isinstance(obj, Mesh):
        # Internally convert Mesh to InputFile to allow for simple comparison via dictionary
        # TODO this should be improved in the future to not fall back to use the 4C specific InputFile
        input_file = InputFile()
        input_file.add(obj)
        # return sections in next step
        obj = input_file

    if isinstance(obj, InputFile):
        # Create a copy of the 4C input file via FourCIPP and dump the mesh representation to it.
        fourc_input_file = obj.fourc_input.copy()
        dump_mesh_representation_to_input_file_legacy(
            fourc_input_file, obj.mesh_representation, obj.element_type_id_to_data
        )
        return fourc_input_file.sections

    if isinstance(obj, str):
        # Comparison for string based Abaqus input files
        # Split the string into individual fragments which can then be compared with tolerance

        def str_to_float(string: str) -> str | float:
            """Convert string to float if possible, otherwise return the
            string.

            Args:
                string: The string to convert.
            Returns:
                The converted string or float.
            """
            try:
                return float(string)
            except ValueError:
                return string

        return [
            str_to_float(fragment)
            for line in obj.splitlines()
            for fragment in line.split(",")
        ]

    raise TypeError(f"The comparison for {type(obj)} is not yet implemented!")


def custom_fourcipp_comparison(
    obj: Any, reference_obj: Any, rtol: float, atol: float
) -> bool | None:
    """Custom comparison function for the FourCIPP
    compare_nested_dicts_or_lists function.

    Comparison between two objects, either lists or numpy arrays.

    Args:
        obj: The object to compare.
        reference_obj: The reference object to compare against.

    Returns:
        True if the objects are equal, otherwise raises an AssertionError.
        If no comparison took place, None is returned.
    """

    if isinstance(obj, (np.ndarray, np.generic)) or isinstance(
        reference_obj, (np.ndarray, np.generic)
    ):
        if not np.allclose(obj, reference_obj, rtol=rtol, atol=atol):
            raise AssertionError(
                f"Custom BeamMe comparison failed!\n\nThe objects are not equal:\n\nobj: {obj}\n\nreference_obj: {reference_obj}"
            )
        return True

    return None


def handle_failed_assertion(
    tmp_path: Path,
    current_test_name: str,
    reference: Path | str | dict | list | np.ndarray | InputFile | Mesh,
    result: Path | str | dict | list | np.ndarray | InputFile | Mesh,
) -> None:
    """Handle failed assertions by opening a diff tool.

    For failed assertions the new result file is written to the temporary
    pytest directory and the VSCode diff tool is opened if available.

    Args:
        tmp_path: Temporary pytest directory.
        current_test_name: Name of the current test.
        reference: The reference data.
        result: The result data.
    """

    # if reference is not a file or if result is not a Mesh or InputFile we do not open the diff
    if not isinstance(reference, Path) or not isinstance(result, (Mesh, InputFile)):
        return

    # save result string to file
    result_path = tmp_path / (
        current_test_name + "_result" + "".join(reference.suffixes)
    )

    if isinstance(result, Mesh):
        input_file = InputFile()
        input_file.add(result)
        result = input_file

    result.dump(
        result_path,
        add_header_default=False,
        add_header_information=False,
        add_footer_application_script=False,
        validate=False,
    )

    print(f"Result string saved to: '{result_path}'.")

    # open VSCode diff tool if available
    if shutil.which("code") is not None:
        child = subprocess.Popen(
            ["code", "--diff", result_path, reference],
            stderr=subprocess.PIPE,
        )
        child.communicate()
