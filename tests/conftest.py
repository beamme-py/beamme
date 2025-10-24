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
"""Base testing framework infrastructure."""

import os
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser

from beamme.core.conf import bme
from beamme.core.material import MaterialBeamBase
from beamme.four_c.material import (
    MaterialEulerBernoulli,
    MaterialKirchhoff,
    MaterialReissner,
    MaterialReissnerElastoplastic,
    MaterialStVenantKirchhoff,
)

# Import additional confest files (split for better overview)
pytest_plugins = [
    "tests.conftest_performance_tests",
    "tests.conftest_result_comparison",
]


def pytest_addoption(parser: Parser) -> None:
    """Add custom command line options to pytest.

    Args:
        parser: Pytest parser
    """

    parser.addoption(
        "--4C",
        action="store_true",
        default=False,
        help="Execute standard and 4C based tests.",
    )

    parser.addoption(
        "--ArborX",
        action="store_true",
        default=False,
        help="Execute standard and ArborX based tests.",
    )

    parser.addoption(
        "--CubitPy",
        action="store_true",
        default=False,
        help="Execute standard and CubitPy based tests.",
    )

    parser.addoption(
        "--performance-tests",
        action="store_true",
        default=False,
        help="Execute standard and performance tests.",
    )

    parser.addoption(
        "--exclude-standard-tests",
        action="store_true",
        default=False,
        help="Exclude standard tests.",
    )


def pytest_collection_modifyitems(config: Config, items: list) -> None:
    """Filter tests based on their markers and provided command line options.

    Currently configured options:
        `pytest`: Execute standard tests with no markers
        `pytest --4C`: Execute standard tests and tests with the `fourc` marker
        `pytest --ArborX`: Execute standard tests and tests with the `arborx` marker
        `pytest --CubitPy`: Execute standard tests and tests with the `cubitpy` marker
        `pytest --performance-tests`: Execute standard tests and tests with the `performance` marker
        `pytest --exclude-standard-tests`: Execute tests with any other marker and exclude the standard unmarked tests

    Args:
        config: Pytest config
        items: Pytest list of tests
    """

    selected_tests = []

    # loop over all collected tests
    for item in items:
        # Get all set markers for current test (e.g. `fourc_arborx`, `cubitpy`, `performance`, ...)
        # We don't care about the "parametrize" marker here
        markers = [
            marker.name
            for marker in item.iter_markers()
            if not marker.name == "parametrize"
        ]

        for flag, marker in zip(
            ["--4C", "--ArborX", "--CubitPy", "--performance-tests"],
            ["fourc", "arborx", "cubitpy", "performance"],
        ):
            if config.getoption(flag) and marker in markers:
                selected_tests.append(item)

        if not markers and not config.getoption("--exclude-standard-tests"):
            selected_tests.append(item)

    deselected_tests = list(set(items) - set(selected_tests))

    items[:] = selected_tests
    config.hook.pytest_deselected(items=deselected_tests)


@pytest.fixture(autouse=True)
def run_before_each_test():
    """Reset the global bme object before each test."""
    bme.set_default_values()


@pytest.fixture(scope="session")
def reference_file_directory() -> Path:
    """Provide the path to the reference file directory.

    Returns:
        Path: A Path object representing the full path to the reference file directory.
    """

    testing_path = Path(__file__).resolve().parent
    return testing_path / "reference-files"


@pytest.fixture(scope="function")
def current_test_name(request: pytest.FixtureRequest) -> str:
    """Return the name of the current pytest test.

    Args:
        request: The pytest request object.

    Returns:
        str: The name of the current pytest test.
    """

    return request.node.originalname


@pytest.fixture(scope="function")
def get_corresponding_reference_file_path(
    reference_file_directory, current_test_name
) -> Callable:
    """Return function to get path to corresponding reference file for each
    test.

    Necessary to enable the function call through pytest fixtures.
    """

    def _get_corresponding_reference_file_path(
        reference_file_base_name: Optional[str] = None,
        additional_identifier: Optional[str] = None,
        extension: str = "4C.yaml",
    ) -> Path:
        """Get path to corresponding reference file for each test. Also check
        if this file exists. Basename, additional identifier and extension can
        be adjusted.

        Args:
            reference_file_base_name: Basename of reference file, if none is
                provided the current test name is utilized
            additional_identifier: Additional identifier for reference file, by default none
            extension: Extension of reference file, by default ".4C.yaml"

        Returns:
            Path to reference file.
        """

        corresponding_reference_file = reference_file_base_name or current_test_name

        if additional_identifier:
            corresponding_reference_file += f"_{additional_identifier}"

        corresponding_reference_file += "." + extension

        corresponding_reference_file_path = (
            reference_file_directory / corresponding_reference_file
        )

        if not os.path.isfile(corresponding_reference_file_path):
            raise AssertionError(
                f"File path: {corresponding_reference_file_path} does not exist"
            )

        return corresponding_reference_file_path

    return _get_corresponding_reference_file_path


@pytest.fixture(scope="function")
def get_bc_data() -> Callable:
    """Return a function to create a dummy definition for a boundary condition
    in 4C.

    Returns:
        A function to create a dummy boundary condition definition.
    """

    def _get_bc_data(*, identifier=None, num_dof: int = 3) -> Dict:
        """Return a dummy definition for a boundary condition in 4C that can be
        used for testing purposes.

        Args:
            identifier: Any value, will be written to the value for the first DOF. This can be used to create multiple boundary conditions and distinguish them in the input file.
            num_dof: Number of degrees of freedom constrained by this boundary condition.
        """

        val = [0] * num_dof
        if identifier is not None:
            val[0] = identifier

        return {
            "NUMDOF": num_dof,
            "ONOFF": [1] * num_dof,
            "VAL": val,
            "FUNCT": [0] * num_dof,
        }

    return _get_bc_data


@pytest.fixture(scope="function")
def get_default_test_beam_material() -> Callable:
    """Return a function to create a default beam material for testing
    purposes.

    Returns:
        A function that creates a default beam material.
    """

    def _get_default_test_beam_material(material_type: str = "reissner", **kwargs):
        """Return a default material for testing purposes.

        Args:
            material_type: The type of beam material to return.

        Returns:
            A material object corresponding to the specified beam type.
        """

        if material_type == "base":
            return MaterialBeamBase(
                radius=1.0, youngs_modulus=1.0, nu=0.3, density=1.0, **kwargs
            )

        elif material_type == "reissner":
            return MaterialReissner(
                radius=1.0, youngs_modulus=1.0, nu=0.3, density=1.0, **kwargs
            )

        elif material_type == "reissner_elastoplastic":
            return MaterialReissnerElastoplastic(
                radius=1.0,
                youngs_modulus=1.0,
                nu=0.3,
                density=1.0,
                yield_moment=2.0,
                isohardening_modulus_moment=3.0,
                **kwargs,
            )

        elif material_type == "kirchhoff":
            return MaterialKirchhoff(
                radius=1.0, youngs_modulus=1.0, nu=1.0, density=1.0, **kwargs
            )

        elif material_type == "euler_bernoulli":
            return MaterialEulerBernoulli(
                radius=1.0, youngs_modulus=1.0, nu=0.3, density=1.0, **kwargs
            )

        else:
            raise ValueError(f"Unknown beam type: {material_type}")

    return _get_default_test_beam_material


@pytest.fixture(scope="function")
def get_default_test_solid_material() -> Callable:
    """Return a function to create a default solid material for testing
    purposes.

    Args:
        material_type: The type of solid material to return.

    Returns:
        A function that creates a default solid material.
    """

    def _get_default_test_solid_material(
        material_type: str = "st_venant_kirchhoff",
    ):
        """Return a default solid material for testing purposes.

        Args:
            material_type: The type of solid material to return.

        Returns:
            A material object corresponding to the specified solid material type.
        """

        if material_type == "st_venant_kirchhoff":
            return MaterialStVenantKirchhoff(youngs_modulus=1.0, nu=0.3, density=1.0)

        else:
            raise ValueError(f"Unknown solid material type: {material_type}")

    return _get_default_test_solid_material
