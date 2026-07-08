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
"""Unit tests for the run 4C module."""

from unittest import mock

import pytest

from beamme.four_c.run_four_c import clean_simulation_directory


@pytest.fixture
def create_simulation_dir(tmp_path):
    """Fixture to create a dummy simulation directory with a file and subdirectory."""
    simulation_dir = tmp_path / "existing_dir"
    simulation_dir.mkdir()
    simulation_sub_dir = simulation_dir / "sub"
    simulation_sub_dir.mkdir()
    test_file = simulation_dir / "test.txt"
    test_file.write_text("content")
    return simulation_dir, simulation_sub_dir, test_file


def test_beamme_four_c_run_four_c_clean_simulation_directory_create_directory(tmp_path):
    """Test that clean_simulation_directory creates a directory."""
    simulation_dir = tmp_path / "new_dir"
    assert not simulation_dir.exists()

    clean_simulation_directory(simulation_dir)
    assert simulation_dir.exists()


@pytest.mark.parametrize("prompt", (False, True))
def test_beamme_four_c_run_four_c_clean_simulation_directory_clear_existing_directory(
    prompt, create_simulation_dir
):
    """Test that clean_simulation_directory cleans an existing directory."""
    simulation_dir, _, _ = create_simulation_dir

    if prompt:
        with mock.patch("builtins.input", return_value="y"):
            clean_simulation_directory(simulation_dir, ask_before_clean=True)
    else:
        clean_simulation_directory(simulation_dir, ask_before_clean=False)

    assert simulation_dir.exists()
    assert not any(simulation_dir.iterdir())


def test_beamme_four_c_run_four_c_clean_simulation_directory_prompt_no(
    create_simulation_dir,
):
    """Test that clean_simulation_directory NO prompt works."""
    simulation_dir, simulation_sub_dir, test_file = create_simulation_dir

    with mock.patch("builtins.input", return_value="n"):
        with pytest.raises(ValueError, match="Directory is not deleted!"):
            clean_simulation_directory(simulation_dir, ask_before_clean=True)

    assert simulation_dir.exists()
    assert simulation_sub_dir.exists()
    assert test_file.exists()
