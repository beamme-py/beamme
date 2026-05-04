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
"""This script is used to test general functionality of the 4C module with end-
to-end integration tests."""

import numpy as np
import pytest
import splinepy

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySetNodes
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import (
    Beam3rHerm2Line3,
    get_four_c_reissner_beam,
)
from beamme.four_c.header_functions import (
    add_result_description,
    set_beam_to_solid_meshtying,
    set_header_static,
    set_runtime_output,
)
from beamme.four_c.input_file import InputFile
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.mesh_creation_functions.nurbs_generic import add_splinepy_nurbs_to_mesh
from beamme.mesh_creation_functions.nurbs_utils import translate_splinepy


@pytest.mark.parametrize(
    ("coupling_type", "coupling_dof_type", "additional_identifier"),
    (
        (bme.bc.point_coupling, bme.coupling_dof.fix, "exact"),
        (
            bme.bc.point_coupling_penalty,
            {"POSITIONAL_PENALTY_PARAMETER": 10000, "ROTATIONAL_PENALTY_PARAMETER": 0},
            "penalty",
        ),
    ),
)
def test_integration_four_c_point_coupling(
    coupling_type,
    coupling_dof_type,
    additional_identifier,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of point couplings for 4C."""

    # Create material and mesh
    material = get_default_test_beam_material(
        material_type="reissner", interaction_radius=2.0
    )
    mesh = Mesh()

    # Create a 2x2 grid of beams.
    for i in range(3):
        for j in range(2):
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, material, [j, i, 0.0], [j + 1, i, 0.0]
            )
            create_beam_mesh_line(
                mesh, Beam3rHerm2Line3, material, [i, j, 0.0], [i, j + 1, 0.0]
            )

    # Couple the beams.
    mesh.couple_nodes(
        reuse_matching_nodes=True,
        coupling_type=coupling_type,
        coupling_dof_type=coupling_dof_type,
    )

    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        mesh,
    )


def test_integration_four_c_point_coupling_indirect(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that indirect point coupling works as expected."""

    material = get_default_test_beam_material(material_type="reissner")
    mesh = Mesh()

    beam_set_1 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [-1, 0, 0], [1, 0, 0]
    )
    beam_set_2 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, -1, 0.1], [0, 1, 0.1], n_el=3
    )
    for i_set, beam_set in enumerate([beam_set_1, beam_set_2]):
        data = {"COUPLING_ID": 3}
        if i_set == 0:
            data["PARAMETERS"] = {
                "POSITIONAL_PENALTY_PARAMETER": 1.1,
                "ROTATIONAL_PENALTY_PARAMETER": 1.2,
                "PROJECTION_VALID_FACTOR": 1.3,
                "CONSTRAINT_ENFORCEMENT": "lagrange_multiplier",
            }
        mesh.add(
            BoundaryCondition(
                beam_set["line"],
                data=data,
                bc_type=bme.bc.point_coupling_indirect,
            )
        )

    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_four_c_fluid_element_section(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Add beam elements to an input file containing fluid elements."""

    input_file, _ = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="import"
        )
    )

    beam_mesh = Mesh()
    material = get_default_test_beam_material(material_type="reissner")
    beam_mesh.add(material)

    create_beam_mesh_line(
        beam_mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        material,
        [0, -0.5, 0],
        [0, 0.2, 0],
        n_el=5,
    )

    input_file.add(beam_mesh)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_surface_to_surface_contact_import(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test that surface-to-surface contact problems can be imported as
    expected."""

    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="solid_mesh"
        ),
        convert_input_to_mesh=True,
    )

    input_file.add(mesh)

    # Compare with the reference file.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_nurbs_import(
    get_default_test_beam_material,
    get_default_test_solid_material,
    get_default_test_solid_element,
    assert_results_close,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Test if the import of a NURBS mesh works as expected.

    We first create the NURBS structure, dump it to disk and then read it again.
    Then we add the beam and beam-to-volume stuff.

    This script generates the 4C test case:
    beam3r_herm2line3_static_beam_to_solid_volume_meshtying_nurbs27_mortar_penalty_line4

    TODO: This test case basically covers that we can import NURBS from
    existing import files. However, this test case also contains a fully
    working 4C test case. So this case should be moved to the 4C tests and
    we should add a very basic unit test case if we can import NURBS.
    """

    # Create a third of the NURBS hollow cylinder
    base = splinepy.helpme.create.disk(
        outer_radius=0.3, inner_radius=0.2, angle=120, n_knot_spans=1
    )
    extruded = base.create.extruded(extrusion_vector=[0, 0, 1])
    extruded.elevate_degrees([0, 2])

    # Add NURBS to mesh (3 times to get the full cylinder)
    mesh = Mesh()
    mat = get_default_test_solid_material(material_type="st_venant_kirchhoff")

    volume_set = GeometrySetNodes(geometry_type=bme.geo.volume)
    fix_set = GeometrySetNodes(geometry_type=bme.geo.surface)
    for i in range(3):
        element_type = get_default_test_solid_element("nurbs_3d")
        patch_set = add_splinepy_nurbs_to_mesh(
            mesh, element_type, extruded, material=mat
        )
        volume_set = volume_set + patch_set["vol"]
        fix_set = fix_set + patch_set["surf_w_min"]
        mesh.add(patch_set)
        mesh.rotate(Rotation([0, 0, 1], np.pi * 2.0 / 3.0))

    mesh.add(
        BoundaryCondition(
            fix_set,
            data={
                "NUMDOF": 3,
                "ONOFF": [1, 1, 1],
                "VAL": [0, 0, 0],
                "FUNCT": [0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            volume_set,
            data={"COUPLING_ID": 1},
            bc_type=bme.bc.beam_to_solid_volume_meshtying,
        )
    )

    # We need this because we get "double" CP from the splinepy object.
    mesh.couple_nodes(reuse_matching_nodes=True)
    # We need this to match the "old" result description
    mesh.rotate(Rotation([0, 0, 1], np.pi * 0.5))
    nurbs_input_file = InputFile()
    nurbs_input_file.add(mesh)
    nurbs_path = tmp_path / "nurbs_mesh.4C.yaml"
    nurbs_input_file.dump(
        nurbs_path,
        validate=False,
        add_header_default=False,
        add_header_information=False,
        add_footer_application_script=False,
    )

    # Create mesh and load solid file.
    input_file, mesh = import_four_c_model(input_file_path=nurbs_path)

    set_header_static(
        input_file,
        time_step=0.5,
        n_steps=2,
        tol_residuum=1e-14,
        tol_increment=1e-8,
    )
    set_beam_to_solid_meshtying(
        input_file,
        bme.bc.beam_to_solid_volume_meshtying,
        contact_discretization="mortar",
        mortar_shape="line4",
        penalty_parameter=1000,
        n_gauss_points=6,
        segmentation=True,
        binning_parameters={
            "binning_bounding_box": [-3, -3, -1, 3, 3, 5],
            "binning_cutoff_radius": 1,
        },
    )
    set_runtime_output(input_file)
    input_file["PROBLEM TYPE"]["SHAPEFCT"] = "Nurbs"
    input_file["IO"]["OUTPUT_BIN"] = True
    input_file["IO"]["STRUCT_DISP"] = True
    input_file["IO"]["VERBOSITY"] = "Standard"

    fun = Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}])
    mesh.add(fun)

    # Create the beam material.
    material = get_default_test_beam_material(material_type="reissner")

    # Create the beams.
    set_1 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, 0, 0.95], [1, 0, 0.95], n_el=2
    )
    set_2 = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        material,
        [-0.25, -0.3, 0.85],
        [-0.25, 0.5, 0.85],
        n_el=2,
    )

    # Add boundary conditions on the beams.
    mesh.add(
        BoundaryCondition(
            set_1["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            set_1["end"],
            {
                "NUMDOF": 9,
                "ONOFF": [0, 1, 0, 0, 0, 0, 0, 0, 0],
                "VAL": [0, 0.02, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, fun, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.neumann,
        )
    )
    mesh.add(
        BoundaryCondition(
            set_2["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            set_2["end"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                "VAL": [-0.06, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [fun, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.neumann,
        )
    )
    mesh.add(
        BoundaryCondition(
            set_1["line"] + set_2["line"],
            {"COUPLING_ID": 1},
            bc_type=bme.bc.beam_to_solid_volume_meshtying,
        )
    )

    # Add result checks.
    displacements = [
        [
            -5.14451531199392575e-01,
            -1.05846397823837826e-01,
            -1.77822866488512921e-01,
        ]
    ]
    nodes = [64]
    add_result_description(input_file, displacements, nodes)

    # Add the mesh to the input file
    input_file.add(mesh)

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_nurbs_multiple_additions_to_input_file(
    get_default_test_solid_element,
    get_default_test_solid_material,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """In this test, we add a NURBS patch to an input file, then a beam and
    then a second NURBS patch.

    This checks that we can add multiple NURBS patches to the input file
    and that the patch IDs are represented correctly.
    """

    def create_nurbs_brick(n_el_dim):
        """Create a NURBS brick with the given number of elements in each
        direction."""
        box_dimensions = [1.5, 3.0, 2.4]
        vol_obj = splinepy.helpme.create.box(*box_dimensions).nurbs
        vol_obj.elevate_degrees([0, 1, 2])
        for i_dim, n_el in enumerate(n_el_dim):
            vol_obj.insert_knots(i_dim, np.linspace(0, 1, n_el + 1))
        return vol_obj

    brick_1 = create_nurbs_brick([1, 2, 3])
    brick_2 = create_nurbs_brick([3, 1, 2])
    translate_splinepy(brick_2, [3, 0, 0])

    def create_mesh(nurbs_obj):
        """Create a mesh with the given NURBS object."""
        mesh = Mesh()
        element_type = get_default_test_solid_element("nurbs_3d")
        patch_set = add_splinepy_nurbs_to_mesh(
            mesh,
            element_type,
            nurbs_obj,
            material=get_default_test_solid_material(
                material_type="st_venant_kirchhoff"
            ),
        )
        mesh.add(patch_set)
        return mesh

    mesh_1 = create_mesh(brick_1)
    mesh_2 = create_mesh(brick_2)

    # Create the input file. Add the first block first, then a beam and then the second block.
    input_file = InputFile()
    input_file.add(mesh_1)

    mesh_beam = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_beam_mesh_line(
        mesh_beam, Beam3rHerm2Line3, mat, [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], n_el=2
    )
    mesh_beam.add(beam_set)
    input_file.add(mesh_beam)

    input_file.add(mesh_2)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_user_defined_boundary_condition(
    get_bc_data,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Check if a user-defined boundary condition can be added."""

    mesh = Mesh()

    mat = get_default_test_beam_material(material_type="reissner")
    sets = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 2, 3])
    mesh.add(
        BoundaryCondition(
            sets["line"], get_bc_data(), bc_type="DESIGN VOL ALE DIRICH CONDITIONS"
        )
    )

    # Compare the output of the mesh.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize("reuse_nodes", [False, True])
def test_integration_four_c_check_multiple_node_penalty_coupling(
    get_default_test_beam_material,
    reuse_nodes,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """For point penalty coupling constraints, we add multiple coupling
    conditions.

    This is checked in this test case. The flag reuse_nodes decides
    whether equal nodes are unified to a single node.
    """

    # Create mesh object
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add three beams that have one common point
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -1, 0])

    mesh.couple_nodes(
        reuse_matching_nodes=reuse_nodes,
        coupling_type=bme.bc.point_coupling_penalty,
        coupling_dof_type={
            "POSITIONAL_PENALTY_PARAMETER": 10000,
            "ROTATIONAL_PENALTY_PARAMETER": 0,
        },
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="reuse" if reuse_nodes else ""
        ),
        mesh,
    )
