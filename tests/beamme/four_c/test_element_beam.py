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
"""Unit tests beam elements for 4C."""

import warnings

import numpy as np
import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import (
    Beam3eb,
    Beam3rHerm2Line3,
    Beam3rLine2Line2,
    get_four_c_kirchhoff_beam,
)
from beamme.four_c.four_c_types import (
    BeamKirchhoffConstraintType,
    BeamKirchhoffParametrizationType,
)
from beamme.four_c.input_file import InputFile
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.nodes import get_single_node


def test_reissner_beam(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that the input file for all types of Reissner beams is generated
    correctly."""

    # Create mesh
    mesh = Mesh()

    # Create material
    material = get_default_test_beam_material("reissner", interaction_radius=2.0)

    # Create a beam arc with the different Reissner beam types.
    for i, beam_type in enumerate([Beam3rHerm2Line3, Beam3rLine2Line2]):
        create_beam_mesh_arc_segment_via_rotation(
            mesh,
            beam_type,
            material,
            [0.0, 0.0, i],
            Rotation([0.0, 0.0, 1.0], np.pi / 2.0),
            2.0,
            np.pi / 2.0,
            n_el=2,
        )

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_kirchhoff_beam(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that the input file for all types of Kirchhoff beams is generated
    correctly."""

    # Create mesh
    mesh = Mesh()

    with warnings.catch_warnings():
        # Ignore the warnings for the rotvec beams.
        warnings.simplefilter("ignore")

        # Loop over options.
        for is_fad in (True, False):
            material = get_default_test_beam_material("kirchhoff", is_fad=is_fad)
            for constraint in BeamKirchhoffConstraintType:
                for parametrization in BeamKirchhoffParametrizationType:
                    # Define the beam object factory function for the
                    # creation functions.
                    BeamObject = get_four_c_kirchhoff_beam(
                        constraint=constraint,
                        parametrization=parametrization,
                        is_fad=is_fad,
                    )

                    # Create a beam.
                    set_1 = create_beam_mesh_line(
                        mesh,
                        BeamObject,
                        material,
                        [0, 0, 0],
                        [1, 0, 0],
                        n_el=2,
                    )
                    set_2 = create_beam_mesh_line(
                        mesh,
                        BeamObject,
                        material,
                        [1, 0, 0],
                        [2, 0, 0],
                        n_el=2,
                    )

                    # Couple the nodes.
                    if parametrization == BeamKirchhoffParametrizationType.rot:
                        mesh.couple_nodes(
                            nodes=[
                                get_single_node(set_1["end"]),
                                get_single_node(set_2["start"]),
                            ]
                        )

                    # Move the mesh away from the next created beam.
                    mesh.translate([0, 0.5, 0])

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_euler_bernoulli(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Recreate the 4C test case beam3eb_static_endmoment_quartercircle.4C.yaml
    This tests the implementation for Euler Bernoulli beams."""

    # Create the mesh and add function and material.
    mesh = Mesh()
    fun = Function([{"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])
    mesh.add(fun)
    mat = get_default_test_beam_material("euler_bernoulli")

    # Create the beam.
    beam_set = create_beam_mesh_line(mesh, Beam3eb, mat, [-1, 0, 0], [1, 0, 0], n_el=16)

    # Add boundary conditions.
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            {
                "NUMDOF": 6,
                "ONOFF": [1, 1, 1, 0, 1, 1],
                "VAL": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "FUNCT": [0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["end"],
            {
                "NUMDOF": 6,
                "ONOFF": [0, 0, 0, 0, 0, 1],
                "VAL": [0.0, 0.0, 0.0, 0.0, 0.0, 7.8539816339744e-05],
                "FUNCT": [0, 0, 0, 0, 0, fun],
            },
            bc_type=bme.bc.moment_euler_bernoulli,
        )
    )

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # Test consistency checks.
    rot = Rotation([1, 2, 3], 2.3434)
    mesh.nodes[-1].rotation = rot
    with pytest.raises(
        ValueError,
        match="The two nodal rotations in Euler Bernoulli beams must be the same",
    ):
        # This raises an error because not all rotation in the beams are
        # the same.
        input_file = InputFile()
        input_file.add(mesh)

    for node in mesh.nodes:
        node.rotation = rot
    with pytest.raises(
        ValueError,
        match="The rotations do not match the direction of the Euler Bernoulli beam",
    ):
        # This raises an error because the rotations do not match the
        # director between the nodes.
        input_file = InputFile()
        input_file.add(mesh)
