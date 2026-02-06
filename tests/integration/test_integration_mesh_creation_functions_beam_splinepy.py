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
"""Integration tests for splinepy beam mesh creation functions."""

import pytest

from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_splinepy import (
    create_beam_mesh_from_splinepy,
)


@pytest.mark.parametrize(
    ("splinepy_type", "ref_length"),
    [
        ("curve_bezier", 5.064502347301837),
        ("curve_nurbs", 3.140204403683438),
        ("curve_nurbs_distorted", 12.056886977957966),
    ],
)
def test_integration_mesh_creation_functions_beam_splinepy(
    splinepy_type,
    ref_length,
    get_splinepy_object,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_beam_mesh_from_splinepy function with different splinepy
    curves."""

    curve = get_splinepy_object(splinepy_type)
    mat = get_default_test_beam_material(material_type="reissner")
    mesh = Mesh()
    _, length = create_beam_mesh_from_splinepy(
        mesh, Beam3rHerm2Line3, mat, curve, n_el=3, output_length=True
    )
    assert_results_close(ref_length, length)

    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=splinepy_type), mesh
    )
