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
"""Helper functions to interact with the BeamMe environment."""

import os as _os
import shutil as _shutil
import subprocess as _subprocess  # nosec B404
import sys as _sys
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path
from typing import Optional as _Optional
from typing import Tuple as _Tuple


def cubitpy_is_available() -> bool:
    """Check if CubitPy is installed.

    Returns:
        True if CubitPy is installed, False otherwise
    """

    if _find_spec("cubitpy") is None:
        return False
    return True


def is_mybinder():
    """Check if the current environment is running on mybinder."""
    return "BINDER_LAUNCH_HOST" in _os.environ.keys()


def is_nbsphinx():
    """Check if the current environment is running in nbsphinx."""
    return "IS_NBSPHINX" in _os.environ


def is_testing():
    """Check if the current environment is a pytest testing run."""
    return "PYTEST_CURRENT_TEST" in _os.environ


def get_env_variable(name, *, default="default_not_set"):
    """Return the value of an environment variable.

    Args
    ----
    name: str
        Name of the environment variable
    default:
        Value to be returned if the given named environment variable does
        not exist. If this is not set and the name is not in the env
        variables, then an error will be thrown.
    """
    if name in _os.environ.keys():
        return _os.environ[name]
    elif default == "default_not_set":
        raise ValueError(f"Environment variable {name} is not set")
    return default


def get_git_data(repo_path: _Path) -> _Tuple[_Optional[str], _Optional[str]]:
    """Return the hash and date of the current git commit of a git repo for a
    given repo path.

    Args:
        repo_path: Path to the git repository.
    Returns:
        A tuple with the hash and date of the current git commit
        if available, otherwise None.
    """

    git = _shutil.which("git")
    if git is None:
        raise RuntimeError("Git executable not found")

    out_sha = _subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=repo_path,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.DEVNULL,
    )

    out_date = _subprocess.run(  # nosec B603
        [git, "show", "-s", "--format=%ci"],
        cwd=repo_path,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.DEVNULL,
    )

    if not out_sha.returncode + out_date.returncode == 0:
        return None, None

    git_sha = out_sha.stdout.decode("ascii").strip()
    git_date = out_date.stdout.decode("ascii").strip()

    return git_sha, git_date


def get_application_path() -> _Path | None:
    """Returns the application path which created this input file and ensures
    that the file exists.

    Returns:
        A path to the file, which created this input file or None.
    """

    # return valid application path if it exists.
    if _Path(_sys.argv[0]).resolve().exists():
        return _Path(_sys.argv[0]).resolve()

    return None
