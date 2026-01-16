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
"""This script is used to test the functionality to create 4C input files."""

import numpy as np
import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.beam_potential import BeamPotential
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.input_file import InputFile
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.nodes import is_node_on_plane


def test_four_c_beam_potential_helix(
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
            choice_master_slave="smaller_eleGID_is_slave",
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


def test_four_c_add_beam_interaction_condition(
    get_default_test_beam_material,
):
    """Ensure that the contact-boundary conditions ids are estimated
    correctly."""

    # Create the mesh.
    mesh = Mesh()

    # Create Material.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create a beam in x-axis.
    beam_x = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0],
        [2, 0, 0],
        n_el=3,
    )

    # Create a second beam in y-axis.
    beam_y = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0],
        [0, 2, 0],
        n_el=3,
    )

    # Add two contact node sets.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_y["line"], bme.bc.beam_to_beam_contact
    )
    assert id == 0

    # Check if we can add the same set twice.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_x["line"], bme.bc.beam_to_beam_contact
    )
    assert id == 1

    # Add some more functions to ensure that everything works as expected:
    for node in mesh.nodes:
        mesh.add(
            BoundaryCondition(
                GeometrySet(node),
                "",
                bc_type=bme.bc.dirichlet,
            )
        )

    # Add condition with higher id.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_x["line"], bme.bc.beam_to_beam_contact, id=3
    )
    assert id == 3

    # Check if the id gap is filled automatically.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_y["line"], bme.bc.beam_to_beam_contact
    )
    assert id == 2


def test_four_c_beam_to_beam_contact(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the beam-to-beam contact boundary conditions."""

    # Create the mesh.
    mesh = Mesh()

    # Create Material.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create a beam in x-axis.
    beam_x = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0],
        [1, 0, 0],
        n_el=2,
    )

    # Create a second beam in y-axis.
    beam_y = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0.5],
        [1, 0, 0.5],
        n_el=2,
    )

    # Add the beam-to-beam contact condition.
    add_beam_interaction_condition(
        mesh, beam_x["line"], beam_y["line"], bme.bc.beam_to_beam_contact
    )

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_four_c_beam_to_solid(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that the automatic ID creation for beam-to-solid conditions
    works."""

    # Load a solid
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_block"
        ),
        convert_input_to_mesh=True,
    )

    # The yaml file already contains the beam-to-solid boundary conditions
    # for the solid. We don't need them in this test case, as we want to
    # create them again. Thus, we have to delete them here.
    mesh.boundary_conditions[
        (bme.bc.beam_to_solid_volume_meshtying, bme.geo.volume)
    ].clear()
    mesh.boundary_conditions[
        (bme.bc.beam_to_solid_surface_meshtying, bme.geo.surface)
    ].clear()

    # Get the geometry set objects representing the geometry from the cubit
    # file.
    surface_set = mesh.geometry_sets[bme.geo.surface][0]
    volume_set = mesh.geometry_sets[bme.geo.volume][0]

    # Add the beam
    material = get_default_test_beam_material(material_type="reissner")
    beam_set_1 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, 0, 0], [0, 0, 1], n_el=1
    )
    beam_set_2 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, 1, 0], [0, 1, 1], n_el=2
    )
    add_beam_interaction_condition(
        mesh,
        volume_set,
        beam_set_1["line"],
        bme.bc.beam_to_solid_volume_meshtying,
    )
    add_beam_interaction_condition(
        mesh,
        volume_set,
        beam_set_2["line"],
        bme.bc.beam_to_solid_volume_meshtying,
    )
    add_beam_interaction_condition(
        mesh,
        surface_set,
        beam_set_2["line"],
        bme.bc.beam_to_solid_surface_meshtying,
    )
    add_beam_interaction_condition(
        mesh,
        surface_set,
        beam_set_1["line"],
        bme.bc.beam_to_solid_surface_meshtying,
    )

    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # If we try to add this the IDs won't match, because the next volume ID for
    # beam-to-surface coupling should be 0 (this one does not make sense, but
    # this is checked in a later test) and the next line ID for beam-to-surface
    # coupling is 2 (there are already two of these conditions).
    with pytest.raises(ValueError):
        add_beam_interaction_condition(
            mesh,
            volume_set,
            beam_set_1["line"],
            bme.bc.beam_to_solid_surface_meshtying,
        )

    # If we add a wrong geometries to the mesh, the creation of the input file
    # should fail, because there is no beam-to-surface contact section that
    # contains a volume set.
    with pytest.raises(KeyError):
        add_beam_interaction_condition(
            mesh,
            volume_set,
            beam_set_1["line"],
            bme.bc.beam_to_solid_surface_contact,
        )
        assert_results_close(get_corresponding_reference_file_path(), mesh)
