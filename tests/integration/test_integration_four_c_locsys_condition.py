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
"""Integration tests for 4C locsys conditions."""

import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.locsys_condition import LocSysCondition
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_integration_four_c_locsys_condition_locsys(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test case for point locsys condition for beams.

    The testcase is adapted from to beam3r_herm2line3_static_locsys.4C.yaml. However it
    has a simpler material, and an additional line locsys condition.
    """
    locsys_rotation = Rotation([0, 0, 1], 0.1)

    # Create the mesh.
    mesh = Mesh()

    fun = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])
    mesh.add(fun)

    # Create the beam.
    beam_set = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        [2.5, 2.5, 2.5],
        [4.5, 2.5, 2.5],
        n_el=1,
    )

    # Add dirichlet boundary conditions.
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    # Add additional dirichlet boundary condition to check if combination with locsys condition works.
    mesh.add(
        BoundaryCondition(
            beam_set["end"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                "VAL": [1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [fun, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )

    # Add locsys condition with rotation
    mesh.add(LocSysCondition(beam_set["end"], rotation=locsys_rotation))

    # Add line Function with function array

    fun_2 = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME": "2.0*t"}])
    mesh.add(fun_2)

    # Check if the LocSys condition is added correctly for a line with additional options.
    mesh.add(
        LocSysCondition(
            GeometrySet(beam_set["line"]),
            rotation=locsys_rotation,
            function_array=[fun_2],
            update_node_position=True,
            use_consistent_node_normal=True,
        )
    )

    # Check that the Locsys condition works with 3 given functions
    mesh.add(
        LocSysCondition(
            GeometrySet(beam_set["start"]),
            function_array=[fun, fun_2, fun_2],
            update_node_position=True,
            use_consistent_node_normal=False,
        )
    )

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # Check that the combination of 3 functions and a rotation raises an error
    with pytest.raises(ValueError, match="If more than a single"):
        mesh.add(
            LocSysCondition(
                GeometrySet(beam_set["start"]),
                rotation=locsys_rotation,
                function_array=[fun, fun_2, fun_2],
                update_node_position=True,
                use_consistent_node_normal=False,
            )
        )
