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
"""This script is used to test the functionality to create 4C input files."""

import numpy as np
import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.beam_potential import BeamPotential
from beamme.four_c.dbc_monitor import linear_time_transformation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.input_file import InputFile
from beamme.four_c.locsys_condition import LocSysCondition
from beamme.four_c.material import MaterialReissner
from beamme.four_c.model_importer import import_four_c_model
from beamme.four_c.solid_shell_thickness_direction import (
    get_visualization_third_parameter_direction_hex8,
    set_solid_shell_thickness_direction,
)
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.nodes import is_node_on_plane


def test_four_c_material_numbering(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test that materials can be added as strings to an input file (as is done
    when importing input files) and that the numbering with other added
    materials does not lead to materials with double IDs."""

    input_file = InputFile()
    input_file.add(
        {
            "MATERIALS": [
                {
                    "MAT": 1,
                    "MAT_ViscoElastHyper": {
                        "NUMMAT": 4,
                        "MATIDS": [10, 11, 12, 13],
                        "DENS": 1.3e-06,
                    },
                },
                {
                    "MAT": 10,
                    "ELAST_CoupNeoHooke": {
                        "YOUNG": 0.16,
                        "NUE": 0.45,
                    },
                },
                {
                    "MAT": 11,
                    "VISCO_GenMax": {
                        "TAU": 0.1,
                        "BETA": 0.4,
                        "SOLVE": "OST",
                    },
                },
                {
                    "MAT": 12,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0024,
                        "K2": 0.14,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 100,
                        "INIT": 3,
                    },
                },
                {
                    "MAT": 13,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0054,
                        "K2": 1.24,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 100,
                        "INIT": 3,
                        "FIBER_ID": 2,
                    },
                },
                {
                    "MAT": 100,
                    "ELAST_StructuralTensor": {
                        "STRATEGY": "Standard",
                    },
                },
                {
                    "MAT": 2,
                    "MAT_ElastHyper": {
                        "NUMMAT": 3,
                        "MATIDS": [20, 21, 22],
                        "DENS": 1.3e-06,
                    },
                },
                {
                    "MAT": 20,
                    "ELAST_CoupNeoHooke": {
                        "YOUNG": 1.23,
                        "NUE": 0.45,
                    },
                },
                {
                    "MAT": 21,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0004,
                        "K2": 12,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 200,
                        "INIT": 3,
                    },
                },
                {
                    "MAT": 22,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0502,
                        "K2": 10,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 200,
                        "INIT": 3,
                        "FIBER_ID": 2,
                    },
                },
                {
                    "MAT": 200,
                    "ELAST_StructuralTensor": {
                        "STRATEGY": "Standard",
                    },
                },
            ]
        }
    )

    mesh = Mesh()
    mesh.add(MaterialReissner(youngs_modulus=1.0, radius=2.0))

    input_file.add(mesh)

    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_four_c_simulation_beam_potential_helix(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test the correct creation of input files for simulations including beam
    to beam potential interactions."""

    mesh = Mesh()
    input_file = InputFile()

    mat = MaterialReissner(youngs_modulus=1000, radius=0.5, shear_correction=1.0)

    # define function for line charge density
    fun = Function([{"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])

    # define the beam potential
    beampotential = BeamPotential(
        input_file,
        mesh,
        pot_law_prefactor=[-1.0e-3, 12.45e-8],
        pot_law_exponent=[6.0, 12.0],
        pot_law_line_charge_density=[1.0, 2.0],
        pot_law_line_charge_density_funcs=[fun, None],
    )

    # set headers for static case and beam potential
    beampotential.add_header(
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
    )
    beampotential.add_runtime_output(every_iteration=True)

    # create helix
    helix_set = create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        helix_angle=np.pi / 4,
        height_helix=10,
        n_el=4,
    )

    # add potential charge conditions to helix
    beampotential.add_potential_charge_condition(geometry_set=helix_set["line"])

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


def test_four_c_solid_shell_direction_detection(
    assert_results_close,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Test the solid shell direction detection functionality."""

    # Test the plates
    _, mesh_block = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_solid_shell_blocks"
        ),
        convert_input_to_mesh=True,
    )

    # Add a beam element to check the function also works with beam elements
    mat = MaterialReissner()
    create_beam_mesh_line(
        mesh_block, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0], n_el=1
    )
    # Set the thickness direction and compare result
    set_solid_shell_thickness_direction(mesh_block.elements, selection_type="thickness")
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="blocks"),
        mesh_block,
    )

    # Test the dome
    _, mesh_dome_original = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_solid_shell_dome"
        ),
        convert_input_to_mesh=True,
    )

    # Test that the thickness version works
    mesh_dome = mesh_dome_original.copy()
    set_solid_shell_thickness_direction(mesh_dome.elements, selection_type="thickness")
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="dome_thickness"),
        mesh_dome,
    )

    # Test that the direction function version works
    def director_function(cell_center):
        """Return director that will be used to determine the solid thickness
        direction."""
        return cell_center / np.linalg.norm(cell_center)

    mesh_dome = mesh_dome_original.copy()
    set_solid_shell_thickness_direction(
        mesh_dome.elements,
        selection_type="projection_director_function",
        director_function=director_function,
    )
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="dome_thickness"),
        mesh_dome,
    )

    # Test that the constant direction version works
    mesh_dome = mesh_dome_original.copy()
    set_solid_shell_thickness_direction(
        mesh_dome.elements,
        selection_type="projection_director",
        director=[0, 0, 1],
        identify_threshold=None,
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="dome_constant_direction"
        ),
        mesh_dome,
    )

    # Also test the visualization function
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="dome_constant_direction", extension="vtu"
    )
    test_file = tmp_path / ref_file.name

    grid = get_visualization_third_parameter_direction_hex8(mesh_dome)
    grid.save(test_file)
    assert_results_close(ref_file, test_file)


def test_four_c_locsys_condition(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test case for point locsys condition for beams.

    The testcase is adapted from to
    beam3r_herm2line3_static_locsys.4C.yaml. However it has a simpler
    material, and an additional line locsys condition.
    """

    # Create the mesh.
    mesh = Mesh()

    fun = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])
    mesh.add(fun)

    mat = MaterialReissner()
    mesh.add(mat)

    # Create the beam.
    beam_set = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [2.5, 2.5, 2.5], [4.5, 2.5, 2.5], n_el=1
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
    mesh.add(LocSysCondition(beam_set["end"], Rotation([0, 0, 1], 0.1)))

    # Add line Function with function array

    fun_2 = Function([{"SYMBOLIC_FUNCTION_OF_SPACE_TIME": "2.0*t"}])
    mesh.add(fun_2)

    # Check if the LocSys condition is added correctly for a line with additional options.
    mesh.add(
        LocSysCondition(
            GeometrySet(beam_set["line"]),
            Rotation(
                [0, 0, 1],
                0.1,
            ),
            function_array=[fun_2],
            update_node_position=True,
            use_consistent_node_normal=True,
        )
    )

    # Check that the Locsys condition works with 3 given functions
    mesh.add(
        LocSysCondition(
            GeometrySet(beam_set["start"]),
            Rotation(
                [0, 0, 1],
                0.1,
            ),
            function_array=[fun, fun_2, fun_2],
            update_node_position=True,
            use_consistent_node_normal=False,
        )
    )

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_four_c_linear_time_transformation_scaling():
    """Test the scaling of the interval for the function.

    Starts with a function within the interval [0,1] and transforms
    them.
    """

    # starting time array
    time = np.array([0, 0.5, 0.75, 1.0])

    # corresponding values 3 values per time step
    force = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # with the result vector
    force_result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # base case no scaling only
    time_trans, force_trans = linear_time_transformation(
        time, force, [0, 1], flip=False
    )

    # first result is simply the attached
    time_result = np.array([0, 0.5, 0.75, 1.0])

    # check solution
    assert time_trans.tolist() == time_result.tolist()
    assert force_trans.tolist() == force_result.tolist()

    # transform to interval [0, 2]
    time_trans, force_trans = linear_time_transformation(
        time, force, [0, 2], flip=False
    )

    # time values should double
    assert time_trans.tolist() == (2 * time_result).tolist()
    assert force_trans.tolist() == force_result.tolist()

    # new result
    force_result = np.array(
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [10, 11, 12]]
    )

    # shift to the interval [1 ,2] and add valid start end point
    time_trans, force_trans = linear_time_transformation(
        time, force, [1, 2, 5], flip=False, valid_start_and_end_point=True
    )
    assert time_trans.tolist() == np.array([0, 1.0, 1.5, 1.75, 2.0, 5.0]).tolist()
    assert force_trans.tolist() == force_result.tolist()


def test_four_c_linear_time_transformation_flip():
    """Test the flip flag option of linear_time_transformation to mirror the
    function."""

    # base case no scaling no end points should be attached
    # starting time array
    time = np.array([0, 0.5, 0.75, 1.0])

    # corresponding values:  3 values per time step
    force = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # first result is simply the attached point at the end
    time_result = np.array([0, 0.25, 0.5, 1.0])

    # with the value vector:
    force_result = np.array([[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]])

    # base case no scaling only end points should be attached
    time_trans, force_trans = linear_time_transformation(time, force, [0, 1], flip=True)

    # check solution
    assert time_result.tolist() == time_trans.tolist()
    assert force_trans.tolist() == force_result.tolist()

    # new force result
    force_result = np.array([[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]])

    time_result = np.array([0, 0.25, 0.5, 1.0]) + 1

    # test now an shift to the interval [1 ,2]
    time_trans, force_trans = linear_time_transformation(time, force, [1, 2], flip=True)
    assert time_result.tolist() == time_trans.tolist()
    assert force_trans.tolist() == force_result.tolist()

    # same trick as above but with 2
    time_result = np.array([0, 2.0, 2.25, 2.5, 3.0, 5.0])
    # new force result
    force_result = np.array(
        [[10, 11, 12], [10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3], [1, 2, 3]]
    )

    # test offset and scaling and add valid start and end point
    time_trans, force_trans = linear_time_transformation(
        time, force, [2, 3, 5], flip=True, valid_start_and_end_point=True
    )
    assert time_result.tolist() == time_trans.tolist()
    assert force_trans.tolist() == force_result.tolist()


def test_four_c_add_beam_interaction_condition():
    """Ensure that the contact-boundary conditions ids are estimated
    correctly."""

    # Create the mesh.
    mesh = Mesh()

    # Create Material.
    mat = MaterialReissner()

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
    assert_results_close, get_corresponding_reference_file_path
):
    """Test the beam-to-beam contact boundary conditions."""

    # Create the mesh.
    mesh = Mesh()

    # Create Material.
    mat = MaterialReissner()

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
    get_corresponding_reference_file_path, assert_results_close
):
    """Test that the automatic ID creation for beam-to-solid conditions
    works."""

    # Load a solid
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_block"
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
    material = MaterialReissner()
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


@pytest.mark.parametrize(
    ("full_import", "additional_identifier"),
    [(False, "dict_import"), (True, "full_import")],
)
def test_four_c_import_non_consecutive_geometry_sets(
    full_import,
    additional_identifier,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that we can import non-consecutively numbered geometry sets."""

    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="input"
        ),
        convert_input_to_mesh=full_import,
    )

    material = MaterialReissner()
    for i in range(3):
        beam_set = create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            material,
            [i + 3, 0, 0],
            [i + 3, 0, 4],
            n_el=2,
        )
        mesh.add(beam_set)

    input_file.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        input_file,
    )
