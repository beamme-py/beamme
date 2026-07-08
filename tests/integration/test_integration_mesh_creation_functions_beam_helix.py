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
"""This script is used to test the helix mesh creation functions."""

import numpy as np
import pytest

from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


@pytest.mark.parametrize(
    "helix_kwargs",
    (
        {"helix_angle": np.pi / 4, "height_helix": 10.0},
        {"helix_angle": np.pi / 4, "turns": 2.5 / np.pi},
        {"height_helix": 10.0, "turns": 2.5 / np.pi},
    ),
)
def test_integration_mesh_creation_functions_beam_helix_no_rotation(
    helix_kwargs,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix and compare it with the reference file."""
    ## Helix angle and height helix combination
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Add simple line to verify that the helix creation does not alter additional meshes
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
    )

    # Create helix.
    helix_set = create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        **helix_kwargs,
        l_el=5.0,
    )
    mesh.add(helix_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize(
    "helix_kwargs",
    (
        {"helix_angle": np.pi / 6, "height_helix": 10.0},
        {"helix_angle": np.pi / 6, "turns": 2.5 / np.pi * np.sqrt(2)},
        {"height_helix": 10.0, "turns": 2.5 / np.pi * np.sqrt(2)},
    ),
)
def test_integration_mesh_creation_functions_beam_helix_rotation_offset(
    helix_kwargs,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix and compare it with the reference file."""
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Add simple line to verify that the helix creation does not alter additional meshes
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
    )

    # Create helix.
    helix_set = create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [3.0, 0.0, 0.0],
        **helix_kwargs,
        l_el=5.0,
    )
    mesh.add(helix_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_helix_radius_zero(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix and compare it with the reference file."""
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Add simple line to verify that the helix creation does not alter additional meshes
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
    )

    # Create helix.
    helix_set = create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        helix_angle=np.pi / 6,
        height_helix=80.0,
        n_el=4,
        warning_straight_line=False,
    )
    mesh.add(helix_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_helix_right_angle(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix and compare it with the reference file."""
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Add simple line to verify that the helix creation does not alter additional meshes
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]
    )

    # Create helix.
    helix_set = create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [2.0, 2.0, 1.0],
        helix_angle=np.pi / 2,
        height_helix=10.0,
        l_el=5.0,
        warning_straight_line=False,
    )
    mesh.add(helix_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)
