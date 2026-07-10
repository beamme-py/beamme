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
"""Test environment utils."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beamme.utils.environment import (
    cubitpy_is_available,
    get_env_variable,
    get_git_data,
    is_mybinder,
    is_nbsphinx,
    is_testing,
)


def test_beamme_utils_environment_is_cubitpy_available() -> None:
    """Test is_cubitpy_available function."""
    with patch("beamme.utils.environment._find_spec", return_value=True):
        assert cubitpy_is_available() is True

    with patch("beamme.utils.environment._find_spec", return_value=None):
        assert cubitpy_is_available() is False


def test_beamme_utils_environment_is_mybinder() -> None:
    """Test is_mybinder function."""
    with patch.dict(os.environ, {"BINDER_LAUNCH_HOST": "some_value"}):
        assert is_mybinder() is True

    with patch.dict(os.environ, {}, clear=True):
        assert is_mybinder() is False


def test_beamme_utils_environment_is_nbsphinx() -> None:
    """Test is_nbsphinx function."""
    with patch.dict(os.environ, {"IS_NBSPHINX": "some_value"}):
        assert is_nbsphinx() is True

    with patch.dict(os.environ, {}, clear=True):
        assert is_nbsphinx() is False


def test_beamme_utils_environment_is_testing() -> None:
    """Test is_testing function."""
    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "some_value"}):
        assert is_testing() is True

    with patch.dict(os.environ, {}, clear=True):
        assert is_testing() is False


def test_beamme_utils_environment_get_env_variable() -> None:
    """Test get_env_variable function."""
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert get_env_variable("TEST_VAR") == "test_value"

    with patch.dict(os.environ, {}, clear=True):
        assert get_env_variable("TEST_VAR", default="default_value") == "default_value"

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError, match="Environment variable TEST_VAR is not set"
        ):
            get_env_variable("TEST_VAR")


@patch("beamme.utils.environment._shutil.which")
@patch("beamme.utils.environment._subprocess.run")
def test_beamme_utils_environment_get_git_data_success(mock_run, mock_which):
    """Test get_git_data function with successful git command execution."""
    mock_which.return_value = "/usr/bin/git"

    mock_sha_process = MagicMock()
    mock_sha_process.returncode = 0
    mock_sha_process.stdout = b"abcdef1234567890\n"

    mock_date_process = MagicMock()
    mock_date_process.returncode = 0
    mock_date_process.stdout = b"2024-04-26 12:34:56 +0000\n"

    mock_run.side_effect = [mock_sha_process, mock_date_process]

    sha, date = get_git_data(Path("/path/to/repo"))

    assert sha == "abcdef1234567890"
    assert date == "2024-04-26 12:34:56 +0000"
    assert mock_run.call_count == 2


@patch("beamme.utils.environment._shutil.which")
def test_beamme_utils_environment_get_git_data_git_not_found(mock_which):
    """Test get_git_data function when git executable is not found."""
    mock_which.return_value = None

    with pytest.raises(RuntimeError, match="Git executable not found"):
        get_git_data(Path("/path/to/repo"))


@patch("beamme.utils.environment._shutil.which")
@patch("beamme.utils.environment._subprocess.run")
def test_beamme_utils_environment_get_git_data_subprocess_failure(mock_run, mock_which):
    """Test get_git_data function with subprocess command failure."""
    mock_which.return_value = "/usr/bin/git"

    mock_failed_process = MagicMock()
    mock_failed_process.returncode = 1
    mock_failed_process.stdout = b""

    mock_run.side_effect = [mock_failed_process, mock_failed_process]

    sha, date = get_git_data(Path("/path/to/repo"))

    assert sha is None
    assert date is None
    assert mock_run.call_count == 2
