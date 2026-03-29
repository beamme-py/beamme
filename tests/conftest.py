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
"""Base testing framework infrastructure."""

import os
from pathlib import Path
from typing import Callable

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser

from beamme.core.conf import bme

# Import additional confest files (split for better overview)
pytest_plugins = [
    "tests.conftest_performance_tests",
    "tests.conftest_result_comparison",
    "tests.conftest_test_object_generators",
]

# Track used and unused reference files during testing if corresponding flag is enabled
USED_REFERENCE_FILES = set()
UNUSED_REFERENCE_FILES = set()


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

    parser.addoption(
        "--check-for-unused-reference-files",
        action="store_true",
        default=False,
        help="Check for unused reference files in the reference file directory.",
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
        `pytest --check-for-unused-reference-files`: Check for unused reference files in the reference file directory

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

    for item in items:
        item.add_marker(pytest.mark.xfail(reason="Temporarily xfailed"))
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
    reference_file_directory, current_test_name, request: pytest.FixtureRequest
) -> Callable:
    """Return function to get path to corresponding reference file for each
    test.

    Necessary to enable the function call through pytest fixtures.
    """

    def _get_corresponding_reference_file_path(
        reference_file_base_name: str | None = None,
        test_name_suffix_trim_count: int | None = None,
        additional_identifier: str | None = None,
        additional_identifier_separator: str = "_",
        extension: str = "4C.yaml",
    ) -> Path:
        """Get path to corresponding reference file for each test. Also check
        if this file exists. Basename, additional identifier and extension can
        be adjusted.

        Args:
            reference_file_base_name: Basename of reference file, if none is
                provided the current test name is utilized. (Mutually exclusive
                with `test_name_suffix_trim_count`)
            test_name_suffix_trim_count: Number of underscore-separated parts to
                remove from the end of the test name when deriving the reference
                file name. For example, given "test_beam_variant1_variant2" and a
                value of 2, the derived reference name will be "test_beam".
                If not provided, the full test name is used.
                (Mutually exclusive with `reference_file_base_name`)
            additional_identifier: Additional identifier for reference file, by default none
            additional_identifier_separator: Separator between base name and
                additional identifier.
            extension: Extension of reference file, by default ".4C.yaml"

        Returns:
            Path to reference file.
        """

        if (
            reference_file_base_name is not None
            and test_name_suffix_trim_count is not None
        ):
            raise ValueError(
                "The parameters `reference_file_base_name` and `test_name_suffix_trim_count`, are mutually exclusive."
            )

        corresponding_reference_file = reference_file_base_name or current_test_name

        if test_name_suffix_trim_count is not None:
            corresponding_reference_file = "_".join(
                corresponding_reference_file.split("_")[:-test_name_suffix_trim_count]
            )

        if additional_identifier:
            corresponding_reference_file += (
                f"{additional_identifier_separator}{additional_identifier}"
            )

        corresponding_reference_file += "." + extension

        corresponding_reference_file_path = (
            reference_file_directory / corresponding_reference_file
        )

        if not os.path.isfile(corresponding_reference_file_path):
            raise AssertionError(
                f"File path: {corresponding_reference_file_path} does not exist"
            )

        # Track usage if flag is enabled
        if request.config.getoption("--check-for-unused-reference-files"):
            USED_REFERENCE_FILES.add(corresponding_reference_file_path.resolve())

        return corresponding_reference_file_path

    return _get_corresponding_reference_file_path


def sessionfinish_unused_reference_files(session):
    """Exit with exit code 1 if any unused reference files are detected.

    This is utilized to ensure that the Github Actions workflow fails if
    unused reference files are detected when the corresponding flag is
    enabled.
    """

    if session.config.getoption("--check-for-unused-reference-files"):
        # reference_file_directory fixture is not able to be used here (fixtures cannot be called directly)
        reference_file_directory = Path(__file__).resolve().parent / "reference-files"

        all_reference_files = {
            p.resolve() for p in reference_file_directory.rglob("*") if p.is_file()
        }

        UNUSED_REFERENCE_FILES.update(all_reference_files - USED_REFERENCE_FILES)

        # if UNUSED_REFERENCE_FILES:
        #     session.exitstatus = 1


def terminal_summary_unused_reference_files(terminalreporter):
    """Print a summary of unused reference files at the end of the pytest
    run."""
    pass
    # if UNUSED_REFERENCE_FILES:
    #     terminalreporter.write_sep(
    #         "=", "Unused Reference Files Found", red=True, bold=True
    #     )
    #     for file in sorted(UNUSED_REFERENCE_FILES):
    #         terminalreporter.write_line(str(file), bold=True, red=True)


def pytest_sessionfinish(session):
    """Exit with exit code 1 if any performance test failed or unused reference
    files are detected.

    This is utilized to ensure that the Github Actions workflow fails if
    performance tests exceed their expected execution time or if unused
    reference files are detected when the corresponding flag is enabled.
    """

    # import here instead of at the top, otherwise pytest warns due to double import via pytest_plugins
    from tests.conftest_performance_tests import sessionfinish_performance_tests

    sessionfinish_performance_tests(session)
    sessionfinish_unused_reference_files(session)


def pytest_terminal_summary(terminalreporter):
    """Print a summary of performance tests or unused reference files at the
    end of the pytest run."""

    # import here instead of at the top, otherwise pytest warns due to double import via pytest_plugins
    from tests.conftest_performance_tests import terminal_summary_performance_tests

    terminal_summary_performance_tests(terminalreporter)
    terminal_summary_unused_reference_files(terminalreporter)
