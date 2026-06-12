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
"""Test README.md."""

import os
import re

import pytest

from beamme.core.mesh import Mesh


@pytest.fixture(scope="session")
def extract_code_snippets(pytestconfig) -> tuple[dict[str, str], list[str]]:
    """Parse README.md for fenced code blocks marked as tests.

    Code fences may be written in two forms:
        ```python test
        # unnamed test block
        ...
        ```
    or
        ```python test:my_name
        # named test block
        ...
        ```

    Returns:
        A pair consisting of:
            - **snippets_named**: dict mapping snippet name (the part after
              `test:`) to the code string.
            - **snippets_unnamed**: list of code strings from unnamed test blocks.
    """
    readme_content = (pytestconfig.rootpath / "README.md").read_text(encoding="utf-8")

    # Match ```python test or ```python test:name
    pattern = re.compile(r"```python\s+test(?::([^\n]+))?\s*\r?\n(.*?)```", re.DOTALL)
    snippets_named = {}
    snippets_unnamed = []
    for i, (name, code) in enumerate(pattern.findall(readme_content)):
        code = code.strip()
        if name:
            snippets_named[name.strip()] = code
        else:
            snippets_unnamed.append(code)
    return snippets_named, snippets_unnamed


def test_other_readme_auto(extract_code_snippets):
    """Run all unnamed code snippets from README automatically."""
    _, snippets_unnamed = extract_code_snippets
    for code in snippets_unnamed:
        exec(code, {})


def test_other_readme_getting_started(
    extract_code_snippets,
    get_corresponding_reference_file_path,
    assert_results_close,
    tmp_path,
):
    """Test the getting started example in the README.md."""
    snippets_named, _ = extract_code_snippets
    os.chdir(tmp_path)
    globals = {}
    exec(snippets_named["getting_started"], globals)

    # The example creates an object `mesh` - check if it exists and is of type `Mesh`.
    assert "mesh" in globals
    assert isinstance(globals["mesh"], Mesh)

    # What we can do, is to check the created vtu output.
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / "getting_started.vtu"
    assert_results_close(ref_file, vtk_file)
