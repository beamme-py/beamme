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
"""This script is used to simulate create 4C input files."""

import os
import re

import numpy as np
import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.dbc_monitor import (
    dbc_monitor_to_mesh,
    dbc_monitor_to_mesh_all_values,
)
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.function_utility import create_linear_interpolation_function
from beamme.four_c.header_functions import (
    add_result_description,
    set_beam_contact_runtime_output,
    set_beam_contact_section,
    set_header_static,
    set_runtime_output,
)
from beamme.four_c.input_file import InputFile
from beamme.four_c.locsys_condition import LocSysCondition
from beamme.four_c.material import MaterialReissner
from beamme.four_c.model_importer import import_four_c_model
from beamme.four_c.run_four_c import run_four_c
from beamme.mesh_creation_functions.applications.beam_honeycomb import (
    create_beam_mesh_honeycomb,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.nodes import check_node_by_coordinate

# We test all test cases in this file twice. Once we only run up to the first
# call of 4C and compare the created input files, this allows to run some core
# functionalities of this file even if 4C is not available. To achieve "full"
# test coverage we also run the test, where we enforce 4C to be run.
PYTEST_4C_SIMULATION_PARAMETRIZE = [
    "enforce_four_c",
    [False, pytest.param(True, marks=pytest.mark.fourc)],
]


def create_cantilever_model(n_steps, time_step=0.5):
    """Create a simple cantilever model.

    Args
    ----
    n_steps: int
        Number of simulation steps.
    time_step: float
        Time step size.
    """

    input_file = InputFile()
    set_header_static(input_file, time_step=time_step, n_steps=n_steps)
    input_file["IO"]["OUTPUT_BIN"] = True
    input_file["IO"]["STRUCT_DISP"] = True

    mesh = Mesh()
    ft = Function(
        [
            {
                "COMPONENT": 0,
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t",
            },
        ]
    )
    mesh.add(ft)

    mat = MaterialReissner(youngs_modulus=100.0, radius=0.1)
    beam_set = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=10
    )

    return input_file, mesh, beam_set


def run_four_c_test(
    tmp_path, name, input_file, n_proc=2, restart=[None, None], **kwargs
):
    """Run 4C with a input file and check the output.

    Args
    ----
    tmp_path: Path
        Path to the temporary directory
    name: str
        Name of the test case
    mesh: InputFile
        The InputFile object that contains the simulation
    n_proc: int
        Number of processors to run 4C on
    restart: [restart_step, restart_from]
        If the simulation should be a restart
    """

    # Check if temp directory exists.
    testing_dir = tmp_path / name
    os.makedirs(testing_dir, exist_ok=True)

    # Create input file.
    input_file_name = os.path.join(testing_dir, name + ".4C.yaml")
    input_file.dump(input_file_name, add_header_information=False, **kwargs)

    return_code = run_four_c(
        input_file_name,
        testing_dir,
        output_name=name,
        n_proc=n_proc,
        restart_step=restart[0],
        restart_from=restart[1],
    )
    assert 0 == return_code


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
@pytest.mark.parametrize("full_import", [False, True])
def test_integration_four_c_simulation_honeycomb_sphere(
    enforce_four_c,
    full_import,
    tmp_path,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create the same honeycomb mesh as defined in 4C/tests/input_files/beam3r
    _herm2lin3_static_point_coupling_BTSPH_contact_stent_honeycomb_stretch_r01_
    circ10.4C.yaml The honeycomb beam is in contact with a rigid sphere, the
    sphere is moved compared to the original test file, since there are some
    problems with the contact convergence.

    The sphere is imported as an existing mesh.
    """

    # Read input file with information of the sphere and simulation.
    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="import"
        ),
        convert_input_to_mesh=full_import,
    )
    # add mesh back to the input file to check if import works
    if full_import:
        input_file.add(mesh)

    # Modify the time step options.
    input_file["STRUCTURAL DYNAMIC"]["NUMSTEP"] = 5
    input_file["STRUCTURAL DYNAMIC"]["TIMESTEP"] = 0.2

    # Delete the results given in the input file.
    input_file.pop("RESULT DESCRIPTION")

    # Add result checks.
    displacement = [0.0, -8.09347204109101170, 2.89298005937795688]
    result_descriptions = []
    nodes = [268, 188, 182]
    for node in nodes:
        for i, direction in enumerate(["x", "y", "z"]):
            result_descriptions.append(
                {
                    "STRUCTURE": {
                        "DIS": "structure",
                        "NODE": node,
                        "QUANTITY": f"disp{direction}",
                        "VALUE": displacement[i],
                        "TOLERANCE": 1e-10,
                    },
                }
            )

    input_file.add({"RESULT DESCRIPTION": result_descriptions})

    # Material for the beam.
    material = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)

    # Create the honeycomb mesh.
    mesh_honeycomb = Mesh()
    honeycomb_set = create_beam_mesh_honeycomb(
        mesh_honeycomb,
        Beam3rHerm2Line3,
        material,
        50.0,
        10,
        4,
        n_el=1,
        closed_top=False,
    )
    mesh_honeycomb.add(honeycomb_set)
    mesh_honeycomb.rotate(Rotation([0, 0, 1], 0.5 * np.pi))

    # Functions for the boundary conditions
    ft = Function(
        [
            {
                "COMPONENT": 0,
                "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "a",
            },
            {
                "VARIABLE": 0,
                "NAME": "a",
                "TYPE": "linearinterpolation",
                "NUMPOINTS": 3,
                "TIMES": [0, 0.2, 1],
                "VALUES": [0, 1, 1],
            },
        ]
    )
    mesh_honeycomb.add(ft)

    # Change the sets to lines, only for purpose of matching the test file
    honeycomb_set["bottom"].geo_type = bme.geo.line
    honeycomb_set["top"].geo_type = bme.geo.line
    mesh_honeycomb.add(
        BoundaryCondition(
            honeycomb_set["bottom"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh_honeycomb.add(
        BoundaryCondition(
            honeycomb_set["top"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                "VAL": [0, 0, 5.0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, ft, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )

    # Add the mesh to the imported solid mesh.
    input_file.add(mesh_honeycomb)

    # Check the created input file
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="full_import" if full_import else None
        ),
        input_file,
    )

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    # Run the input file in 4C.
    run_four_c_test(
        tmp_path,
        "honeycomb_sphere" + ("_full" if full_import else ""),
        input_file,
    )


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
@pytest.mark.parametrize("full_import", [False, True])
def test_integration_four_c_simulation_beam_and_solid_tube(
    enforce_four_c,
    full_import,
    tmp_path,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Merge a solid tube with a beam tube and simulate them together."""

    # Create the input file and read solid mesh data.
    input_file, imported_mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_tube"
        ),
        convert_input_to_mesh=full_import,
    )
    # Add mesh back to the input file to check if import works
    if full_import:
        input_file.add(imported_mesh)

    # Add options for beam_output.
    input_file.add(
        {
            "IO/RUNTIME VTK OUTPUT/BEAMS": {
                "OUTPUT_BEAMS": True,
                "DISPLACEMENT": True,
                "USE_ABSOLUTE_POSITIONS": True,
                "TRIAD_VISUALIZATIONPOINT": True,
                "STRAINS_GAUSSPOINT": True,
            }
        }
    )

    # Add functions for boundary conditions and material.
    mesh = Mesh()
    sin = Function([{"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "sin(t*2*pi)"}])
    cos = Function([{"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "cos(t*2*pi)"}])
    material = MaterialReissner(youngs_modulus=1e9, radius=0.25, shear_correction=0.75)
    mesh.add(sin, cos, material)

    # Add a straight beam.
    mesh.add(material)
    cantilever_set = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [2, 0, -5], [2, 0, 5], n_el=3
    )

    # Add boundary conditions.
    mesh.add(
        BoundaryCondition(
            cantilever_set["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            cantilever_set["end"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                "VAL": [3.0, 3.0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [cos, sin, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )

    # Add result checks.
    displacements = [
        [1.50796091342925, 1.31453288915877e-8, 0.0439008100184687],
        [0.921450108160878, 1.41113401669104e-15, 0.0178350143764099],
    ]

    nodes = [32, 69]
    add_result_description(input_file, displacements, nodes)

    # Call get_unique_geometry_sets to check that this does not affect the
    # mesh creation.
    mesh.get_unique_geometry_sets(link_to_nodes="all_nodes")

    # Add mesh to input file
    input_file.add(mesh)

    # Check the created input file
    assert_results_close(
        get_corresponding_reference_file_path(),
        input_file,
    )

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    # Run the input file in 4C.
    run_four_c_test(
        tmp_path,
        "beam_and_solid_tube" + ("_full" if full_import else ""),
        input_file,
    )


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
def test_integration_four_c_simulation_honeycomb_variants(
    enforce_four_c,
    assert_results_close,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Create a few different honeycomb structures."""

    # Create input file.
    input_file = InputFile()

    # Set options with different syntaxes.
    input_file.add(
        {
            "PROBLEM TYPE": {
                "PROBLEMTYPE": "Structure",
            },
            "IO": {
                "OUTPUT_BIN": False,
                "STRUCT_DISP": False,
                "VERBOSITY": "Standard",
            },
        }
    )
    input_file.add(
        {
            "IO/RUNTIME VTK OUTPUT": {
                "OUTPUT_DATA_FORMAT": "binary",
                "INTERVAL_STEPS": 1,
                "EVERY_ITERATION": False,
            }
        }
    )
    input_file.add(
        {
            "STRUCTURAL DYNAMIC": {
                "LINEAR_SOLVER": 1,
                "INT_STRATEGY": "Standard",
                "DYNAMICTYPE": "Statics",
                "PREDICT": "TangDis",
                "TIMESTEP": 1.0,
                "NUMSTEP": 666,
                "MAXTIME": 10.0,
                "TOLRES": 1.0e-4,
                "TOLDISP": 1.0e-11,
                "MAXITER": 20,
            }
        }
    )
    input_file.add(
        {
            "SOLVER 1": {
                "NAME": "Structure_Solver",
                "SOLVER": "UMFPACK",
            },
        }
    )
    input_file.add(
        {
            "IO/RUNTIME VTK OUTPUT/BEAMS": {
                "OUTPUT_BEAMS": True,
                "DISPLACEMENT": True,
                "TRIAD_VISUALIZATIONPOINT": True,
                "STRAINS_GAUSSPOINT": True,
            },
        }
    )

    # Check that we can overwrite keys
    input_file["STRUCTURAL DYNAMIC"]["NUMSTEP"] = 1

    # This does not work, because we would overwrite the entire section.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Section(s) STRUCTURAL DYNAMIC are defined in both InputFile objects. "
            "In order to join the InputFile objects remove the section(s) in one of them."
        ),
    ):
        input_file.add({"STRUCTURAL DYNAMIC": {"NUMSTEP": "something"}})

    # Create four meshes with different types of honeycomb structure.
    mesh = Mesh()
    material = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)
    ft = []
    ft.append(Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}]))
    ft.append(Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}]))
    ft.append(Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}]))
    ft.append(Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}]))
    mesh.add(ft)

    counter = 0
    for vertical in [False, True]:
        for closed_top in [False, True]:
            mesh.translate(17 * np.array([1, 0, 0]))
            honeycomb_set = create_beam_mesh_honeycomb(
                mesh,
                Beam3rHerm2Line3,
                material,
                10,
                6,
                3,
                n_el=2,
                vertical=vertical,
                closed_top=closed_top,
            )
            mesh.add(
                BoundaryCondition(
                    honeycomb_set["bottom"],
                    {
                        "NUMDOF": 9,
                        "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                        "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    },
                    bc_type=bme.bc.dirichlet,
                )
            )
            mesh.add(
                BoundaryCondition(
                    honeycomb_set["top"],
                    {
                        "NUMDOF": 9,
                        "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                        "VAL": [0.0001, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0],
                        "FUNCT": [
                            ft[counter],
                            ft[counter],
                            ft[counter],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                    },
                    bc_type=bme.bc.neumann,
                    double_nodes=bme.double_nodes.remove,
                )
            )
            counter += 1

    # Add mesh to input file.
    input_file.add(mesh)

    # Add result checks.
    displacements = [
        [1.31917210027397980e-01, 1.99334884558314690e-01, 6.92209310957152130e-02],
        [1.32982726482608615e-01, 2.00555145810952351e-01, 6.97003431426771458e-02],
        [7.69274209663553116e-02, 1.24993734710951515e-01, 5.86799180712692867e-02],
        [6.98802675783889299e-02, 1.09892533095288236e-01, 4.83525527530398319e-02],
    ]
    nodes = [190, 470, 711, 1071]
    add_result_description(input_file, displacements, nodes)

    # Check the created input file
    assert_results_close(get_corresponding_reference_file_path(), input_file)

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    # Run the input file in 4C.
    run_four_c_test(tmp_path, "honeycomb_variants", input_file)


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
def test_integration_four_c_simulation_rotated_beam_axis(
    enforce_four_c,
    tmp_path,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """
    Create three beams that consist of two connected lines.
    - The first case uses the same nodes for the connection of the lines,
        and the nodes are equal in this case.
    - The second case uses the same nodes for the connection of the lines,
        but the nodes have a different rotation along the basis vector 1.
    - The third case uses two nodes at the connection between the lines,
        and couples them with a coupling.
    """

    # Create input file
    input_file = InputFile()

    # Create mesh
    mesh = Mesh()

    # Set header
    set_header_static(input_file, time_step=0.05, n_steps=20)

    # Define linear function over time.
    ft = Function([{"SYMBOLIC_FUNCTION_OF_TIME": "t"}])
    mesh.add(ft)

    # Set beam material.
    mat = MaterialReissner(youngs_modulus=2.07e2, radius=0.1, shear_correction=1.1)

    # Direction of the lines and the rotation between the beams.
    direction = np.array([0.5, 1, 2])
    alpha = np.pi / 27 * 7
    force_fac = 0.01

    # Create mesh.
    for i in range(3):
        mesh_line = Mesh()

        # Create the first line.
        set_1 = create_beam_mesh_line(
            mesh_line, Beam3rHerm2Line3, mat, [0, 0, 0], 1.0 * direction, n_el=3
        )

        if not i == 0:
            # In the second case rotate the line, so the triads do not
            # match any more.
            mesh_line.rotate(Rotation(direction, alpha))

        if i == 2:
            # The third line is with couplings.
            start_node = None
        else:
            start_node = set_1["end"]

        # Add the second line.
        set_2 = create_beam_mesh_line(
            mesh_line,
            Beam3rHerm2Line3,
            mat,
            1.0 * direction,
            2.0 * direction,
            n_el=3,
            start_node=start_node,
        )

        # Add boundary conditions.
        mesh_line.add(
            BoundaryCondition(
                set_1["start"],
                {
                    "NUMDOF": 9,
                    "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                    "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                bc_type=bme.bc.dirichlet,
            )
        )
        mesh_line.add(
            BoundaryCondition(
                set_2["end"],
                {
                    "NUMDOF": 9,
                    "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                    "VAL": [force_fac] * 6 + [0] * 3,
                    "FUNCT": [ft] * 6 + [0] * 3,
                },
                bc_type=bme.bc.neumann,
            )
        )

        if i == 2:
            # In the third case add a coupling.
            mesh_line.couple_nodes()

        # Add the mesh to the input file.
        mesh.add(mesh_line)

        # Each time move the whole mesh.
        mesh.translate([1, 0, 0])

    # Add result checks.
    displacements = [[1.5015284845, 0.35139255451, -1.0126517891]] * 3
    nodes = [13, 26, 40]
    add_result_description(input_file, displacements, nodes)

    # Add the mesh to the input file
    input_file.add(mesh)

    # Check the created input file
    assert_results_close(get_corresponding_reference_file_path(), input_file)

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    # Run the input file in 4C.
    run_four_c_test(tmp_path, "rotated_beam_axis", input_file)
    run_four_c_test(
        tmp_path, "rotated_beam_axis", input_file, nox_xml_file="xml_name.xml"
    )


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
@pytest.mark.parametrize("all_values", (True, False))
def test_integration_four_c_simulation_dbc_monitor_to_input(
    enforce_four_c,
    all_values,
    tmp_path,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Common driver to simulate a cantilever beam with Dirichlet boundary
    conditions and then apply those as Neumann boundaries.

    This can be used to test the two different functions
    by selecting one of the appropriate initial run_names:

    test_cantilever_w_dbc_monitor_to_input:
        This function explicitly tests dbc_monitor_to_input.

    test_cantilever_w_dbc_monitor_to_input_all_values:
        For the application of the boundary conditions, the last values for
        the force are used. This function explicitly tests
        dbc_monitor_to_input_all_values.
    """

    # Create and run the initial simulation.
    initial_input_file, initial_mesh, mesh_beam_set = create_cantilever_model(n_steps=2)

    initial_mesh.add(
        BoundaryCondition(
            mesh_beam_set["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    initial_mesh.add(
        BoundaryCondition(
            mesh_beam_set["end"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                "VAL": [-0.2, 1.5, 1.0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                "TAG": "monitor_reaction",
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    initial_input_file.add(
        {
            "IO/MONITOR STRUCTURE DBC": {
                "INTERVAL_STEPS": 1,
                "WRITE_CONDITION_INFORMATION": True,
                "FILE_TYPE": "yaml",
            }
        }
    )
    initial_input_file.add(initial_mesh)

    # Check the input file
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="initial"),
        initial_input_file,
    )

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    # Run the simulation in 4C
    initial_run_name = "test_dbc_monitor"
    run_four_c_test(tmp_path, initial_run_name, initial_input_file)

    # Create and run the second simulation.
    restart_input_file, restart_mesh, mesh_beam_set = create_cantilever_model(
        n_steps=21
    )

    restart_mesh.add(
        BoundaryCondition(
            mesh_beam_set["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    function_nbc = Function(
        [
            {"SYMBOLIC_FUNCTION_OF_TIME": "nbc_value"},
            {
                "VARIABLE": 0,
                "NAME": "nbc_value",
                "TYPE": "linearinterpolation",
                "NUMPOINTS": 2,
                "TIMES": [1, 11],
                "VALUES": [1, 0],
            },
        ]
    )
    restart_mesh.add(function_nbc)

    if all_values:
        dbc_monitor_to_mesh_all_values(
            restart_mesh,
            tmp_path / initial_run_name / f"{initial_run_name}-102_monitor_dbc.yaml",
            n_dof=9,
            time_span=[10 * 0.5, 21 * 0.5],
            functions=[function_nbc, function_nbc, function_nbc],
        )
    else:
        dbc_monitor_to_mesh(
            restart_mesh,
            tmp_path / initial_run_name / f"{initial_run_name}-102_monitor_dbc.yaml",
            n_dof=9,
            function=function_nbc,
        )

    restart_input_file.add(restart_mesh)

    displacements = [
        [-4.09988307566066690e-01, 9.93075098427816383e-01, 6.62050065618549843e-01]
    ]
    nodes = [21]
    add_result_description(restart_input_file, displacements, nodes)

    # Check the input file of the restart simulation
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="restart"),
        restart_input_file,
    )

    # Run the restart simulation
    run_four_c_test(
        tmp_path,
        f"{initial_run_name}_restart",
        restart_input_file,
        restart=[2, f"../{initial_run_name}/{initial_run_name}"],
    )


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
def test_integration_four_c_simulation_dirichlet_boundary_to_neumann_boundary_with_all_values(
    enforce_four_c,
    tmp_path,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """First simulate a cantilever beam with Dirichlet boundary conditions and
    then apply those as Neumann boundaries.

    For the application of the boundary conditions, all values of the
    force are used.
    """

    # Define Parameters.
    n_steps = 5  # number of simulation steps
    dt = 0.1  # time step size from create_cantilever_model

    # Create and run the initial simulation.
    initial_simulation, mesh, beam_set = create_cantilever_model(n_steps, dt)

    # Add simple lienar interpolation function.
    mesh.add(
        Function(
            [
                {
                    "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "a",
                },
                {
                    "VARIABLE": 0,
                    "NAME": "a",
                    "TYPE": "linearinterpolation",
                    "NUMPOINTS": 4,
                    "TIMES": [0, dt * n_steps, 2 * dt * n_steps, 9999999999.0],
                    "VALUES": [0.0, 1.0, 0.0, 0.0],
                },
            ]
        )
    )

    # Apply displacements to all nodes.
    for _, node in enumerate(beam_set["line"].get_all_nodes()):
        # do not constraint middle nodes
        if not node.is_middle_node:
            # Set Dirichlet conditions at one end.
            if check_node_by_coordinate(node, 0, 0):
                mesh.add(
                    BoundaryCondition(
                        GeometrySet(node),
                        {
                            "NUMDOF": 9,
                            "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0],
                            "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        },
                        bc_type=bme.bc.dirichlet,
                    )
                )
            else:
                # Add small displacement at other end.
                mesh.add(
                    BoundaryCondition(
                        GeometrySet(node),
                        {
                            "NUMDOF": 9,
                            "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0],
                            "VAL": [
                                0,
                                0,
                                0.25 * np.sin(node.coordinates[0] * np.pi),
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            ],
                            "FUNCT": [0, 0, 2, 0, 0, 0, 0, 0, 0],
                            "TAG": "monitor_reaction",
                        },
                        bc_type=bme.bc.dirichlet,
                    )
                )

    initial_simulation.add(mesh)

    # Add DB-monitor header.
    initial_simulation.add(
        {
            "IO/MONITOR STRUCTURE DBC": {
                "INTERVAL_STEPS": 1,
                "WRITE_CONDITION_INFORMATION": True,
                "FILE_TYPE": "yaml",
            }
        }
    )

    # Check the input file.
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="dirichlet"),
        initial_simulation,
    )

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    # Run the simulation in 4C.
    initial_run_name = "all_dbc_to_nbc_initial"
    run_four_c_test(tmp_path, initial_run_name, initial_simulation)

    # Create and run the second simulation.
    force_simulation, mesh, beam_set = create_cantilever_model(2 * n_steps, dt)

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

    # Set up path to monitor.
    monitor_db_path = tmp_path / initial_run_name

    # Convert the Dirichlet conditions into Neuman conditions.
    for _, _, file_names in os.walk(monitor_db_path):
        for file_name in sorted(file_names):
            if "_monitor_dbc" in file_name:
                dbc_monitor_to_mesh_all_values(
                    mesh,
                    os.path.join(monitor_db_path, file_name),
                    steps=[0, n_steps + 1],
                    time_span=[0, n_steps * dt, 2 * n_steps * dt],
                    type="hat",
                    n_dof=9,
                )

    force_simulation.add(mesh)

    displacements = [[0.0, 0.0, 0.0]]
    nodes = [21]
    add_result_description(force_simulation, displacements, nodes)

    # Compare the input file of the restart simulation.
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="neumann"),
        force_simulation,
        atol=1e-6,
    )

    # Add runtime output.
    set_runtime_output(force_simulation)

    initial_run_name = "all_dbc_to_nbc_initial_3"
    run_four_c_test(tmp_path, initial_run_name, force_simulation)


@pytest.mark.fourc
def test_integration_four_c_simulation_cantilever_convergence(
    tmp_path, assert_results_close
):
    """Create multiple simulations of a cantilever beam.

    This is a legacy test that used to test the simulation manager.
    """

    def create_and_run_cantilever(n_el, *, n_proc=1):
        """Create a cantilever beam for a convergence analysis."""

        input_file = InputFile()
        set_header_static(input_file, time_step=0.25, n_steps=4)
        set_runtime_output(input_file, output_energy=True)

        mesh = Mesh()
        mat = MaterialReissner(radius=0.1, youngs_modulus=10000.0)
        beam_set = create_beam_mesh_line(
            mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0], n_el=n_el
        )
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
        fun = Function(
            [
                {"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"},
            ]
        )
        mesh.add(
            fun,
            BoundaryCondition(
                beam_set["end"],
                {
                    "NUMDOF": 9,
                    "ONOFF": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    "VAL": [0, 0, -0.5, 0, 0, 0, 0, 0, 0],
                    "FUNCT": [0, 0, fun, 0, 0, 0, 0, 0, 0],
                },
                bc_type=bme.bc.dirichlet,
            ),
        )

        input_file.add(mesh)

        output_name = f"cantilever_convergence_{n_el}"
        run_four_c_test(tmp_path, output_name, input_file, n_proc=n_proc)
        testing_dir = tmp_path / output_name
        my_data = np.genfromtxt(
            testing_dir / f"{output_name}_energy.csv", delimiter=","
        )
        return my_data[-1, 2]

    results = {}
    for n_el in range(1, 7, 2):
        results[str(n_el)] = create_and_run_cantilever(n_el)
    results["ref"] = create_and_run_cantilever(40, n_proc=4)

    results_ref = {
        "5": 0.335081498526998,
        "3": 0.335055487040675,
        "1": 0.33453718896204,
        "ref": 0.335085590674607,
    }
    assert_results_close(results_ref, results)


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
def test_integration_four_c_simulation_beam_to_beam_contact_example(
    enforce_four_c,
    tmp_path,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Small test example to show how a beam contact example with beam penalty
    contact can be set up.

    The test case consists of two beams: one beam is allocated along the x-axis and
    the other beam is located along the y-axis.
    The beam along the y-axis is placed above the other beam by an additional offset in z-Directions.
    Due to prescribed displacements at the tips of beam in y-axis, the two beams get in contact around the origin.
    """

    # Define Parameters for example
    l_beam = 2
    r_beam = 0.1
    n_ele = 11
    h = 0.25

    # Set up mesh and material.
    mesh = Mesh()
    mat = MaterialReissner(radius=r_beam, youngs_modulus=1)

    # Create a beam in x-axis.
    beam_x = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [-l_beam / 2, 0, 0],
        [l_beam / 2, 0, 0],
        n_el=n_ele,
    )

    # Apply Dirichlet condition to start and end nodes of beam 1:
    for set_name in ["start", "end"]:
        mesh.add(
            BoundaryCondition(
                beam_x[set_name],
                {
                    "NUMDOF": 9,
                    "ONOFF": [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "VAL": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "FUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                bc_type=bme.bc.dirichlet,
            )
        )

    # Create a second beam in y-axis.
    beam_y = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, -l_beam / 2, h],
        [0, l_beam / 2, h],
        n_el=n_ele,
    )

    t = [0, 1, 1000.0]
    disp_values = [0.0, -h - r_beam, -h - r_beam]

    # Create a linear interpolation function with the displacement.
    fun = create_linear_interpolation_function(t, disp_values)

    mesh.add(fun)

    # Apply Dirichlet conditions at starting and end node to displace the beam endings.
    for set_name in ["start", "end"]:
        mesh.add(
            BoundaryCondition(
                beam_y[set_name],
                {
                    "NUMDOF": 9,
                    "ONOFF": [1, 1, 1, 0, 1, 0, 0, 0, 0],
                    "VAL": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    "FUNCT": [0, 0, fun, 0, 0, 0, 0, 0, 0],
                },
                bc_type=bme.bc.dirichlet,
            )
        )

    # Create a beam to beam contact boundary condition factory.
    add_beam_interaction_condition(
        mesh,
        beam_x["line"],
        beam_y["line"],
        bme.bc.beam_to_beam_contact,
    )

    # Create the input file
    input_file = InputFile()

    # Add the standard,static header.
    set_header_static(
        input_file,
        time_step=0.05,
        n_steps=24,
        total_time=1.2,
        write_stress="2PK",
        tol_residuum=1e-5,
        tol_increment=1e-5,
    )

    # Set the parameters for beam to beam contact.
    set_beam_contact_section(
        input_file,
        btb_penalty=50,
        penalty_regularization_g0=r_beam * 0.02,
        binning_parameters={
            "binning_cutoff_radius": 5,
            "binning_bounding_box": [-3, -3, -3, 3, 3, 3],
        },
    )

    # Add the mesh to the input file.
    input_file.add(mesh)

    # Add normal runtime output.
    set_runtime_output(input_file)

    # Add special runtime output for beam interaction.
    set_beam_contact_runtime_output(input_file, every_iteration=False)

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), input_file)

    displacements = [[1.11158519615313324e-03, 0, -1.48443346935174636e-01]]
    nodes = [13]
    add_result_description(input_file, displacements, nodes, tol=1e-8)

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    initial_run_name = "beam_to_beam_contact_simulation"
    run_four_c_test(tmp_path, initial_run_name, input_file, n_proc=1)


@pytest.mark.parametrize(*PYTEST_4C_SIMULATION_PARAMETRIZE)
def test_integration_four_c_simulation_locsys(
    tmp_path,
    enforce_four_c,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a star like structure made out of 3 beams to test complex locsys
    conditions.

    We first rotate the star and then apply a prescribed displacemet to
    its center.
    """

    # Define Parameters for example
    l_beam = 2
    r_beam = 0.05
    n_ele_base = 2

    # Set up mesh and material.
    mesh = Mesh()
    mat = MaterialReissner(radius=r_beam, youngs_modulus=1)

    # List of rotations to be applied on the full stystem starting from time t=0.
    final_rotation_vector = (
        0.2 * np.pi * np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3])
    )
    time_values = np.linspace(0, 1, 10)
    rotations = [
        Rotation.from_rotation_vector(final_rotation_vector * time)
        for time in time_values
    ]

    # Create the beams.
    for i in range(3):
        phi = np.pi * 2.0 / 3.0 * i
        ref_position = l_beam * np.array([np.cos(phi), np.sin(phi), 0])
        beam_set = create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 0, 0],
            ref_position,
            n_el=n_ele_base
            + i,  # A variable number of elements helps to take the symmetry out of the system.
        )

        # Get the functions describing the rotation vector components.
        ref_rotation = Rotation([0, 0, 1], phi)
        rotation_vectors = np.array(
            [(rotation * ref_rotation).get_rotation_vector() for rotation in rotations]
        )
        rotation_vector_component_functions = [
            create_linear_interpolation_function(
                time_values, rotation_vectors[:, i_dir]
            )
            for i_dir in range(3)
        ]
        mesh.add(rotation_vector_component_functions)

        # Get the "displacements" for the end node for each time step, also in the locsys coordinate system.
        displacements = np.array(
            [rotation * ref_position - ref_position for rotation in rotations]
        )
        displacements_locsys = np.array(
            [
                (rotation * ref_rotation).inv() * displacement
                for rotation, displacement in zip(rotations, displacements)
            ]
        )
        displacements_locsys_functions = [
            create_linear_interpolation_function(
                time_values, displacements_locsys[:, i_dir]
            )
            for i_dir in range(3)
        ]
        mesh.add(displacements_locsys_functions)

        # Set the boundary condition and locsys on the outer end.
        mesh.add(
            LocSysCondition(
                beam_set["end"], function_array=rotation_vector_component_functions
            )
        )
        mesh.add(
            BoundaryCondition(
                beam_set["end"],
                {
                    "NUMDOF": 9,
                    "ONOFF": [0, 1, 1] + [0] * 6,
                    "VAL": [1, 1, 1] + [0] * 6,
                    "FUNCT": displacements_locsys_functions + [0] * 6,
                },
                bc_type=bme.bc.dirichlet,
            )
        )

    # Couple the nodes at the origin together.
    mesh.couple_nodes()

    # Apply the boundary condition to the last node at the origin. This DBC is applied starting from time t=1
    f_dbc = create_linear_interpolation_function([0, 1, 2], [0, 0, 1])
    mesh.add(f_dbc)
    mesh.add(LocSysCondition(beam_set["start"], rotation=rotations[-1]))
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            {
                "NUMDOF": 9,
                "ONOFF": [0, 0, 1] + [0] * 6,
                "VAL": [0, 0, 1.0] + [0] * 6,
                "FUNCT": [0, 0, f_dbc] + [0] * 6,
            },
            bc_type=bme.bc.dirichlet,
        )
    )

    # Create the input file
    input_file = InputFile()

    # Add the standard,static header.
    set_header_static(
        input_file,
        total_time=2.0,
        n_steps=20,
        tol_residuum=1e-12,
        tol_increment=1e-12,
    )
    set_runtime_output(input_file)

    # Add the mesh to the input file.
    input_file.add(mesh)

    # Add result checks
    displacements = [
        [-6.26693202358605039e-01, 8.32271999424795572e-01, -4.56166283491497349e-01]
    ]
    nodes = [5]
    add_result_description(input_file, displacements, nodes, tol=1e-10)

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), input_file)

    # Check if we still have to actually run 4C.
    if not enforce_four_c:
        return

    run_four_c_test(tmp_path, "test_four_c_simulation_locsys", input_file, n_proc=1)
