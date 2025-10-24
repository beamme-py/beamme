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
"""This script validates all 4C input files with FourCIPP in this
repository."""

import sys as _sys
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

from fourcipp.fourc_input import FourCInput


@_dataclass
class Error:
    """Class to store import errors."""

    filename: str
    message: str


def validate_file(filename: str) -> list[Error]:
    """Validate a 4C input file with FourCIPP.

    Args:
        filename: The filename of the 4C input file to validate.

    Returns:
        List of errors found in the 4C input file or None.
    """

    try:
        inputfile = FourCInput.from_4C_yaml(_Path(filename))
        inputfile.validate(sections_only=True)
    except Exception as e:
        return [Error(filename, str(e))]

    return []


def main() -> None:
    """Validate all 4C input files with FourCIPP."""

    errors: list[Error] = []

    for filename in _sys.argv[1:]:
        if not _Path(filename).exists():
            errors.append(Error(filename, "File not found!"))

        errors.extend(validate_file(filename))

    if errors:
        print("Found 4C input files with errors:\n")

        for error in errors:
            print(f"{error.filename}: {error.message}\n")

        print("Please ensure that all 4C input files are correctly validated.")
        _sys.exit(1)
    else:
        _sys.exit(0)


if __name__ == "__main__":
    """Run the main function if the script is executed."""
    main()
