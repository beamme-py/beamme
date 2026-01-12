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
"""Prepare documentation for the website."""

import os
import shutil
from pathlib import Path


def prepare_docs():
    """Prepare documentation for the website.

    This function copies all relevant files for the website.

    Note: We remove the target directories before copying the files to ensure
    that no stale files remain.
    """

    # create directory which contains all the markdown files
    markdown_dir = Path("website/docs/source/md")
    if markdown_dir.exists():
        shutil.rmtree(markdown_dir)
    os.makedirs(markdown_dir, exist_ok=True)

    # copy readme
    shutil.copy("README.md", markdown_dir / "README.md")

    # create directory which contains all the example files
    examples_source_dir = Path("examples")
    examples_target_dir = Path("website/docs/source/examples")
    if examples_target_dir.exists():
        shutil.rmtree(examples_target_dir)
    file_extensions_to_copy = {".ipynb", ".py"}
    for file in examples_source_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() in file_extensions_to_copy:
            rel_path = file.relative_to(examples_source_dir)
            dest_path = examples_target_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, dest_path)


if __name__ == "__main__":
    """Prepare documentation for the website."""
    prepare_docs()
