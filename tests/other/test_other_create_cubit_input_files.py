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
"""Test that the input files created with CubitPy are up to date."""

import pytest

from tests.create_test_models import (
    create_block,
    create_multiple_solid_bricks,
    create_single_solid_element_brick,
    create_solid_brick,
    create_solid_shell_meshes,
    create_tube,
)


@pytest.mark.cubitpy
def test_other_create_cubit_input_files_tube(
    tmp_path,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that the solid tube reference file is up to date."""

    result_path = tmp_path / get_corresponding_reference_file_path().name
    create_tube(result_path)
    assert_results_close(result_path, get_corresponding_reference_file_path())


@pytest.mark.cubitpy
def test_other_create_cubit_input_files_block(
    tmp_path,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that the solid block reference file is up to date."""

    result_path = tmp_path / get_corresponding_reference_file_path().name
    create_block(result_path)
    assert_results_close(result_path, get_corresponding_reference_file_path())


@pytest.mark.cubitpy
def test_other_create_cubit_input_files_solid_shell(
    tmp_path,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that the solid shell reference files are up to date."""

    reference_path_blocks = get_corresponding_reference_file_path(
        additional_identifier="blocks"
    )
    reference_path_dome = get_corresponding_reference_file_path(
        additional_identifier="dome"
    )
    result_path_blocks = tmp_path / reference_path_blocks.name
    result_path_dome = tmp_path / reference_path_dome.name

    create_solid_shell_meshes(result_path_blocks, result_path_dome)
    assert_results_close(result_path_blocks, reference_path_blocks)
    assert_results_close(result_path_dome, reference_path_dome)


@pytest.mark.cubitpy
def test_other_create_cubit_input_files_single_solid_element_brick(
    tmp_path,
    get_corresponding_reference_file_path,
    get_default_test_solid_material,
    assert_results_close,
):
    """Test that the single solid element brick reference files are up to
    date."""

    reference_file = get_corresponding_reference_file_path()
    result_path = tmp_path / reference_file.name

    create_single_solid_element_brick(result_path, get_default_test_solid_material)

    assert_results_close(result_path, reference_file)


@pytest.mark.cubitpy
def test_other_create_cubit_input_files_solid_brick(
    tmp_path,
    get_corresponding_reference_file_path,
    get_default_test_solid_material,
    assert_results_close,
):
    """Test that the solid brick reference files are up to date."""

    reference_file = get_corresponding_reference_file_path()
    result_path = tmp_path / reference_file.name

    create_solid_brick(result_path, get_default_test_solid_material)
    assert_results_close(result_path, reference_file)


@pytest.mark.cubitpy
def test_other_create_cubit_input_files_multiple_solid_bricks(
    tmp_path, get_corresponding_reference_file_path, assert_results_close
):
    """Test that the solid brick reference files are up to date."""

    reference_file = get_corresponding_reference_file_path()
    result_path = tmp_path / reference_file.name

    cubit = create_multiple_solid_bricks()
    cubit.dump(str(result_path), mesh_in_exo=True)
    assert False
    # assert_results_close(result_path, reference_file)
