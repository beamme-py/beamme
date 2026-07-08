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
"""Test that the input files created for testing are up to date."""

import pytest

from tests.create_test_models import (
    create_beam_to_solid_conditions_model,
)


@pytest.mark.parametrize("full_import", (False, True))
def test_other_create_input_files_beam_to_solid(
    full_import,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create the input file for some beam-to-solid tests."""
    # Create the input file
    input_file, mesh = create_beam_to_solid_conditions_model(
        get_default_test_beam_material,
        get_corresponding_reference_file_path,
        full_import=full_import,
    )
    input_file.add(mesh)

    # Check results
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="full_import" if full_import else None
        ),
        input_file,
    )
