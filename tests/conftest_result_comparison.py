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
import pyvista as pv
import xmltodict
from vistools.vtk.compare_grids import compare_grids

from beamme.core.mesh import Mesh
from beamme.four_c.input_file import InputFile
from beamme.four_c.input_file_dump_functions import (
    dump_mesh_representation_to_input_file_vtu,
)
from beamme.utils.data_structures import compare_nested_dicts_or_lists
from tests.conftest import USED_REFERENCE_FILES

# GLOBAL DEFAULT TEST TOLERANCES
RELATIVE_TOLERANCE = 1e-13
ABSOLUTE_TOLERANCE = 1e-13


@pytest.fixture(scope="function")
def assert_results_close(
    tmp_path, current_test_name, get_corresponding_reference_file_path
) -> Callable:
    """Return function to compare either string or files.

    Necessary to enable the function call through pytest fixtures.

    Args:
        tmp_path: Temporary path to write file if assertion fails.
        current_test_name: Name of the current test to create file
            if assertion fails.
        get_corresponding_reference_file_path: Fixture to get the path of the
            corresponding reference file.

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
        four_c_input_file_data_format: str = "yaml",
    ) -> None:
        """Comparison between reference and result with relative or absolute
        tolerance.

        If the comparison fails, an assertion is raised.

        Args:
            reference: The reference data.
            result: The result data.
            rtol: The relative tolerance.
            atol: The absolute tolerance.
            four_c_input_file_data_format: Mesh format for the FourC input file.
        """

        # convert all other types into dicts/lists
        converted_reference = convert_to_primitive_type(
            reference, four_c_input_file_data_format=four_c_input_file_data_format
        )
        converted_result = convert_to_primitive_type(
            result, four_c_input_file_data_format=four_c_input_file_data_format
        )

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


def convert_to_primitive_type(
    obj: str | int | float | dict | list | np.ndarray | Path | Mesh | InputFile,
    four_c_input_file_data_format: str | None = None,
) -> int | float | dict | list | np.ndarray | pv.UnstructuredGrid:
    """Convert the given object to a primitive type, e.g., dict, list, numpy
    array, or pyvista grid.

    Args:
        obj: The object to convert.
        four_c_input_file_data_format: Mesh format for the FourC input file.

    Returns:
        The raw data (either a dictionary, list, numpy array, or pyvista grid).
    """

    if isinstance(obj, (int, float, dict, list, np.ndarray)):
        return obj

    if isinstance(obj, Path):
        if obj.suffix == ".xml" or obj.suffix == ".pvd":
            return xmltodict.parse(obj.read_text(encoding="utf-8"))

        elif obj.suffix == ".json":
            return json.loads(obj.read_text(encoding="utf-8"))

        elif obj.suffix == ".vtu":
            return pv.read(obj)

        elif obj.name.endswith(".4C.yaml"):
            # Return input file sections as dictionary
            sections = (
                InputFile().from_4C_yaml(input_file_path=obj).fourc_input.sections
            )
            if "STRUCTURE GEOMETRY" in sections:
                # If external geometry is given, add the vtu file to the data structure
                # for comparison.
                mesh_file_path = obj.parent / sections["STRUCTURE GEOMETRY"]["FILE"]
                sections["STRUCTURE GEOMETRY"]["FILE"] = pv.read(mesh_file_path)
                # Add the file to the set of used reference files.
                USED_REFERENCE_FILES.add(mesh_file_path.resolve())
            return sections

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
        if four_c_input_file_data_format == "yaml":
            return obj.get_fourcipp_input_with_mesh().sections
        elif four_c_input_file_data_format == "vtu":
            fourc_input = obj.fourc_input.copy()
            vtu_grid = dump_mesh_representation_to_input_file_vtu(
                fourc_input, obj.mesh_representation, obj.element_type_id_to_data
            )
            # Add the grid to the dictionary containing the input file information
            fourc_input["STRUCTURE GEOMETRY"]["FILE"] = vtu_grid
            return fourc_input.sections
        else:
            raise ValueError(
                f"Got unexpected argument {four_c_input_file_data_format} for "
                "`four_c_input_file_data_format`."
            )

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

    if isinstance(obj, pv.UnstructuredGrid):
        return obj

    raise TypeError(f"The comparison for {type(obj)} is not yet implemented!")


def custom_fourcipp_comparison(
    obj: Any, reference_obj: Any, rtol: float, atol: float
) -> bool | None:
    """Custom comparison function for the FourCIPP
    compare_nested_dicts_or_lists function.

    Comparison between two special objects like numpy arrays or pyvista grids.

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

    if isinstance(obj, pv.UnstructuredGrid) and isinstance(
        reference_obj, pv.UnstructuredGrid
    ):
        compare = compare_grids(obj, reference_obj, output=True, rtol=rtol, atol=atol)
        if not compare[0]:
            raise AssertionError("\n".join(compare[1]))
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
