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
"""This script checks if all Python imports within provided files align with our coding
style, as described in the section "Coding guidelines" in the repository README.md."""

import ast as _ast
import sys as _sys
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path


@_dataclass
class Error:
    """Class to store import errors."""

    filename: str
    message: str
    line_nr: int | None = None
    column_nr: int | None = None


class ImportChecker(_ast.NodeVisitor):
    """Class to check if Python imports align with our coding style."""

    def __init__(self, filename: str) -> None:
        """Initialize the ImportChecker.

        Args:
            filename: The filename of the current file (to store for errors)
        """
        self.errors: list[Error] = []
        self.filename = filename

    def visit_Import(self, node: _ast.Import) -> None:
        """Visit an import statement and check if the import aligns with our coding
        style.

        Args:
            node: The import node to check.
        """
        for alias in node.names:
            if alias.asname is None:
                self.errors.append(
                    Error(
                        self.filename,
                        f"Import '{alias.name}' has no alias!",
                        node.lineno,
                        node.col_offset,
                    )
                )
            elif not alias.asname.startswith("_"):
                self.errors.append(
                    Error(
                        self.filename,
                        f"Import '{alias.name}' has alias '{alias.asname}' that doesn't start with '_'!",
                        node.lineno,
                        node.col_offset,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: _ast.ImportFrom) -> None:
        """Visit an import from statement and check if the import aligns with our coding
        style.

        Args:
            node: The import from node to check.
        """
        for alias in node.names:
            # Check for wildcard imports
            if alias.name == "*":
                self.errors.append(
                    Error(
                        self.filename,
                        "Wildcard imports are not allowed!",
                        node.lineno,
                        node.col_offset,
                    )
                )
                continue

            # Check for private imports from third party libraries
            if not (
                node.module is not None and node.module.startswith("beamme.")
            ) and alias.name.startswith("_"):
                from_module = node.module or f"relative import (level {node.level})"
                self.errors.append(
                    Error(
                        self.filename,
                        f"Import of private functionality '{alias.name}' from '{from_module}' is not allowed!",
                        node.lineno,
                        node.col_offset,
                    )
                )
                continue

            # Check for missing aliases
            if alias.asname is None:
                from_module = node.module or f"relative import (level {node.level})"
                self.errors.append(
                    Error(
                        self.filename,
                        f"Import '{alias.name}' from '{from_module}' has no alias!",
                        node.lineno,
                        node.col_offset,
                    )
                )
            else:
                # Check for correct aliases (internal imports cannot be renamed)
                if node.module is not None and node.module.startswith("beamme."):
                    expected_asname = "_" + alias.name
                    if alias.asname != expected_asname:
                        from_module = (
                            node.module or f"relative import (level {node.level})"
                        )
                        self.errors.append(
                            Error(
                                self.filename,
                                f"Internal import '{alias.name}' from '{from_module}' has incorrect alias '{alias.asname}'! It should be '{expected_asname}'!",
                                node.lineno,
                                node.col_offset,
                            )
                        )
                # Check for correct aliases (external imports can be renamed)
                elif not alias.asname.startswith("_"):
                    from_module = node.module or f"relative import (level {node.level})"
                    self.errors.append(
                        Error(
                            self.filename,
                            f"External import '{alias.name}' from '{from_module}' has incorrect alias '{alias.asname}'! It should be '_{alias.asname}'!",
                            node.lineno,
                            node.col_offset,
                        )
                    )
        self.generic_visit(node)


def check_file(filename: str) -> list[Error]:
    """Check a Python file for import errors.

    Args:
        filename: The filename of the Python file to check.

    Returns:
        List of errors found in the Python file.
    """
    with open(filename, "r") as file:
        content = file.read()
    tree = _ast.parse(content, filename=filename)

    checker = ImportChecker(filename)
    checker.visit(tree)
    return checker.errors


def main() -> None:
    """Check all provided Python files for import errors."""
    errors: list[Error] = []

    for filename in _sys.argv[1:]:
        if not _Path(filename).exists():
            errors.append(Error(filename, "File not found!"))
            continue

        errors.extend(check_file(filename))

    if errors:
        print("Found imports which do not align with our coding style:\n")

        for error in errors:
            if error.line_nr is not None and error.column_nr is not None:
                print(
                    f"    * {error.filename}:{error.line_nr}:{error.column_nr}: {error.message}\n"
                )
            else:
                print(f"{error.filename}: {error.message}\n")

        print(
            "For more information on how to fix these issues, please refer to the contributing guidelines in our README.md."
        )
        _sys.exit(1)
    else:
        _sys.exit(0)


if __name__ == "__main__":
    """Run the main function if the script is executed."""
    main()
