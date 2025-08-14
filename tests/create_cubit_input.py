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
"""This script creates solid input files with CubitPy which are then used in
testing."""

from beamme.four_c.input_file import InputFile
from beamme.four_c.model_importer import import_cubitpy_model
from beamme.utils.environment import cubitpy_is_available

if cubitpy_is_available():
    from cubitpy import CubitPy, cupy
    from cubitpy.mesh_creation_functions import (
        create_brick,
        extrude_mesh_normal_to_surface,
    )


def create_tube_cubit_mesh(r, h, n_circumference, n_height):
    """Create a solid tube in cubit.

    Args
    ----
    r: float
        Radius of the cylinder.
    h: float
        Height of the cylinder.
    n_circumference: int
        Number of elements along the circumferential direction.
    n_height: int
        Number of elements along the axial direction.

    Return
    ----
    The created cubit object.
    """

    # Initialize cubit.
    cubit = CubitPy()

    # Create cylinder.
    cylinder = cubit.cylinder(h, r, r, r)

    # Set the mesh size.
    for curve in cylinder.curves():
        cubit.set_line_interval(curve, n_circumference)
    cubit.cmd("surface 1 size {}".format(h / n_height))

    # Set blocks and sets.
    cubit.add_element_type(cylinder.volumes()[0], cupy.element_type.hex8, name="tube")

    # Return the cubit object.
    return cubit, cylinder


def create_tube_cubit():
    """Load the solid tube and add input file parameters."""

    # Initialize cubit.
    cubit, cylinder = create_tube_cubit_mesh(0.25, 10.0, 6, 10)

    # Mesh the geometry.
    cylinder.volumes()[0].mesh()

    # Set boundary conditions.
    cubit.add_node_set(
        cylinder.surfaces()[1],
        name="fix",
        bc_section="DESIGN SURF DIRICH CONDITIONS",
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [1, 1, 1],
            "VAL": [0, 0, 0],
            "FUNCT": [0, 0, 0],
        },
    )
    cubit.add_node_set(
        cylinder.surfaces()[2],
        name="dirichlet_controlled",
        bc_section="DESIGN SURF DIRICH CONDITIONS",
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [1, 1, 1],
            "VAL": [3.0, 3.0, 0],
            "FUNCT": [1, 2, 0],
        },
    )

    # Set header.
    cubit.fourc_input.combine_sections(
        {
            "PROBLEM TYPE": {"PROBLEMTYPE": "Structure"},
            "IO": {
                "VERBOSITY": "Standard",
            },
            "IO/RUNTIME VTK OUTPUT": {
                "OUTPUT_DATA_FORMAT": "binary",
                "INTERVAL_STEPS": 1,
                "EVERY_ITERATION": False,
            },
            "IO/RUNTIME VTK OUTPUT/STRUCTURE": {
                "OUTPUT_STRUCTURE": True,
                "DISPLACEMENT": True,
            },
            "STRUCTURAL DYNAMIC": {
                "LINEAR_SOLVER": 1,
                "INT_STRATEGY": "Standard",
                "DYNAMICTYPE": "Statics",
                "RESTARTEVERY": 5,
                "PREDICT": "TangDis",
                "TIMESTEP": 0.05,
                "NUMSTEP": 20,
                "MAXTIME": 1.0,
                "TOLRES": 1.0e-5,
                "TOLDISP": 1.0e-11,
                "MAXITER": 20,
            },
            "SOLVER 1": {"NAME": "Structure_Solver", "SOLVER": "UMFPACK"},
            "MATERIALS": [
                {
                    "MAT": 1,
                    "MAT_Struct_StVenantKirchhoff": {
                        "YOUNG": 1.0e9,
                        "NUE": 0.0,
                        "DENS": 7.8e-6,
                    },
                }
            ],
            "FUNCT1": [
                {
                    "COMPONENT": 0,
                    "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "cos(2*pi*t)",
                }
            ],
            "FUNCT2": [
                {
                    "COMPONENT": 0,
                    "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "sin(2*pi*t)",
                }
            ],
        }
    )

    # Return the cubit object.
    return cubit


def create_tube(file_path):
    """Write the solid tube to a file."""

    # Export mesh.
    create_tube_cubit().dump(file_path)


def create_tube_tutorial(file_path):
    """Create the solid tube for the tutorial."""

    # Initialize cubit.
    cubit, cylinder = create_tube_cubit_mesh(0.05, 3.0, 6, 10)

    # Put the tube in the correct position.
    cubit.cmd("rotate volume 1 angle -90 about X include_merged")
    cubit.move(cylinder, [0, 1.5, 1.5])

    # Mesh the geometry.
    cylinder.volumes()[0].mesh()

    # Set boundary conditions.
    cubit.add_node_set(
        cylinder.surfaces()[1],
        name="fix",
        bc_type=cupy.bc_type.dirichlet,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [1, 1, 1],
            "VAL": [0, 0, 0],
            "FUNCT": [0, 0, 0],
        },
    )
    cubit.add_node_set(
        cylinder.surfaces()[2],
        name="dirichlet_controlled",
        bc_type=cupy.bc_type.dirichlet,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [1, 0, 0],
            "VAL": [0.5, 0, 0],
            "FUNCT": [1, 0, 0],
        },
    )

    # Set header.
    cubit.fourc_input.combine_sections(
        {
            "MATERIALS": [
                {
                    "MAT": 1,
                    "MAT_Struct_StVenantKirchhoff": {
                        "YOUNG": 1.0,
                        "NUE": 0.0,
                        "DENS": 0.0,
                    },
                }
            ]
        }
    )

    # Export mesh.
    cubit.dump(file_path)


def create_block_cubit():
    """Create a solid block in cubit and add a volume condition."""

    # Initialize cubit.
    cubit = CubitPy()

    # Create the block.
    cube = create_brick(cubit, 1, 1, 1, mesh_factor=9)

    # Set the material.
    cubit.fourc_input["MATERIALS"] = [
        {"MAT": 1, "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2}}
    ]

    # Add the boundary condition.
    cubit.add_node_set(
        cube.volumes()[0],
        bc_type=cupy.bc_type.beam_to_solid_volume_meshtying,
        bc_description={"COUPLING_ID": 1},
    )

    # Add the boundary condition.
    cubit.add_node_set(
        cube.surfaces()[0],
        bc_type=cupy.bc_type.beam_to_solid_surface_meshtying,
        bc_description={"COUPLING_ID": 2},
    )

    # Set point coupling conditions.
    nodes = cubit.group()
    nodes.add([cube.vertices()[0], cube.vertices()[2]])
    cubit.add_node_set(
        nodes,
        bc_type=cupy.bc_type.point_coupling,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [1, 2, 3],
        },
    )

    # Return the cubit object.
    return cubit


def create_block(file_path):
    """Create the solid cube in cubit and write it to a file."""

    # Export mesh.
    create_block_cubit().dump(file_path)


def create_solid_shell_meshes(file_path_blocks, file_path_dome):
    """Create the meshes needed for the solid shell tests."""

    def create_brick_mesh(
        dimensions, n_elements, *, element_type=cupy.element_type.hex8sh
    ):
        """Create a mesh with a solid brick."""
        cubit = CubitPy()
        create_brick(
            cubit,
            *dimensions,
            mesh_interval=n_elements,
            element_type=element_type,
            mesh=True,
        )
        cubit.fourc_input["MATERIALS"] = [
            {
                "MAT": 1,
                "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2},
            }
        ]
        _, mesh = import_cubitpy_model(cubit, convert_input_to_mesh=True)
        return mesh

    # Create the input file with the blocks representing plates in different planes
    input_file = InputFile()
    dimensions = [0.1, 2, 4]
    elements = [1, 2, 2]

    def rotate_list(original_list, n):
        """Rotate the list."""
        return original_list[-n:] + original_list[:-n]

    # Add the plates in all directions (permute the dimension and number of elements
    # in each direction)
    for i in range(3):
        brick = create_brick_mesh(rotate_list(dimensions, i), rotate_list(elements, i))
        brick.translate([i * 4, 0, 0])
        input_file.add(brick)

    # Add a last plate with standard solid elements, to make sure that the algorithm
    # skips those
    brick = create_brick_mesh(
        rotate_list(dimensions, 1),
        rotate_list(elements, 1),
        element_type=cupy.element_type.hex8,
    )
    brick.translate([3 * 4, 0, 0])
    input_file.add(brick)

    input_file.dump(
        file_path_blocks, add_header_information=False, validate_sections_only=True
    )

    # Create the dome input
    cubit = CubitPy()
    cubit.cmd("create sphere radius 1 zpositive")
    cubit.cmd("surface 2 size auto factor 6")
    cubit.cmd("mesh surface 2")
    dome_mesh = extrude_mesh_normal_to_surface(
        cubit, [cubit.surface(2)], 0.1, n_layer=1
    )
    cubit.add_element_type(dome_mesh, cupy.element_type.hex8sh)
    cubit.fourc_input["MATERIALS"] = [
        {
            "MAT": 1,
            "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2},
        }
    ]
    cubit.dump(file_path_dome)
