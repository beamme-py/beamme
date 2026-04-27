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
"""Integration tests for the beam potential functionality of BeamMe."""

import numpy as np

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.four_c.beam_potential import BeamPotential
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.input_file import InputFile
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix
from beamme.utils.nodes import is_node_on_plane


def test_integration_four_c_beam_potential_helix(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the correct creation of input files for simulations including beam
    to beam potential interactions."""

    mesh = Mesh()
    input_file = InputFile()

    # define function for line charge density
    fun = Function([{"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])
    mesh.add(fun)

    # define the beam potential
    beampotential = BeamPotential(
        pot_law_prefactor=[-1.0e-3, 12.45e-8],
        pot_law_exponent=[6.0, 12.0],
        pot_law_line_charge_density=[1.0, 2.0],
        pot_law_line_charge_density_funcs=[fun, None],
    )

    # set headers for static case and beam potential
    input_file.add(
        beampotential.create_header(
            potential_type="volume",
            cutoff_radius=10.0,
            evaluation_strategy="single_length_specific_small_separations_simple",
            regularization_type="linear",
            regularization_separation=0.1,
            integration_segments=2,
            gauss_points=50,
            potential_reduction_length=15.0,
            automatic_differentiation=False,
            choice_source_target="smaller_eleGID_is_source",
            two_half_pass=True,
            runtime_output_interval_steps=1,
            runtime_output_force=True,
            runtime_output_moment=True,
            runtime_output_uids=True,
            runtime_output_per_ele_pair=True,
            runtime_output_every_iteration=True,
        )
    )

    # create helix
    helix_set = create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        helix_angle=np.pi / 4,
        height_helix=10,
        n_el=4,
    )

    # add potential charge conditions to input file
    mesh.add(
        beampotential.create_potential_charge_conditions(geometry_set=helix_set["line"])
    )

    # Add boundary condition to bottom node
    mesh.add(
        BoundaryCondition(
            GeometrySet(
                mesh.get_nodes_by_function(
                    is_node_on_plane,
                    normal=[0, 0, 1],
                    origin_distance=0.0,
                    tol=0.1,
                )
            ),
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )

    input_file.add(mesh)

    assert_results_close(get_corresponding_reference_file_path(), input_file)
