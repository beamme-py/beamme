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
"""This script contains functionality to create solid input files (or plain
cubit instances) with CubitPy which are then used in testing."""

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.input_file import InputFile
from beamme.four_c.model_importer import import_cubitpy_model, import_four_c_model
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.environment import cubitpy_is_available

if cubitpy_is_available():
    from cubitpy import CubitPy, cupy
    from cubitpy.cubit_utility import formatter as fmt
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


def create_beam_to_solid_conditions_model(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    full_import: bool,
):
    """Create the input file for the beam-to-solid input conditions tests."""

    # Create input file
    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_block"
        ),
        convert_input_to_mesh=full_import,
    )

    # Add beams to the model
    mesh_beams = Mesh()
    material = get_default_test_beam_material(material_type="reissner")
    create_beam_mesh_line(
        mesh_beams, Beam3rHerm2Line3, material, [0, 0, 0], [0, 0, 1], n_el=3
    )
    create_beam_mesh_line(
        mesh_beams, Beam3rHerm2Line3, material, [0, 0.5, 0], [0, 0.5, 1], n_el=3
    )

    # Set beam-to-solid coupling conditions.
    line_set = GeometrySet(mesh_beams.elements)
    mesh_beams.add(
        BoundaryCondition(
            line_set,
            bc_type=bme.bc.beam_to_solid_volume_meshtying,
            data={"COUPLING_ID": 1},
        )
    )
    mesh_beams.add(
        BoundaryCondition(
            line_set,
            bc_type=bme.bc.beam_to_solid_surface_meshtying,
            data={"COUPLING_ID": 2},
        )
    )
    mesh.add(mesh_beams)

    return input_file, mesh


def create_single_solid_element_brick(input_file_path, get_default_test_solid_material):
    """Create an input file with a single solid element brick in CubitPy for
    testing purposes."""

    # Create the brick with a single solid element
    cubit = CubitPy()
    create_brick(
        cubit, 1, 2, 3, mesh_interval=[1, 1, 1], element_type=cupy.element_type.hex8
    )
    material = get_default_test_solid_material(material_type="st_venant_kirchhoff")
    dumped_material_dict = material.dump_to_list()
    dumped_material_dict["MAT"] = 1
    cubit.fourc_input["MATERIALS"] = [dumped_material_dict]

    cubit.dump(input_file_path)


def create_solid_brick(input_file_path, get_default_test_solid_material):
    """Create an a brick with the dimensions 1x3x15."""

    cubit = CubitPy()
    brick = create_brick(
        cubit,
        1,
        3,
        15,
        element_type=cupy.element_type.hex8,
        mesh_interval=[2, 1, 10],
        mesh=False,
    )
    cubit.cmd(f"move volume {brick.id()} x 0 y 0 z 7.5 include_merged")
    brick.mesh()
    material = get_default_test_solid_material(material_type="st_venant_kirchhoff")
    dumped_material_dict = material.dump_to_list()
    dumped_material_dict["MAT"] = 1
    cubit.fourc_input["MATERIALS"] = [dumped_material_dict]
    cubit.dump(input_file_path)


def create_multiple_solid_bricks():
    """Create blocks with all supported element types.

    Also add some node sets to ensure their correct representation in
    BeamMe.
    """

    if not cupy.is_coreform():
        raise ValueError("This script requires Cubit Coreform")

    cubit = CubitPy()

    brick_element_types = [
        cupy.element_type.hex8,
        cupy.element_type.hex20,
        cupy.element_type.hex27,
    ]
    for i_brick, element_type in enumerate(brick_element_types):
        brick = create_brick(
            cubit,
            1,
            1,
            1,
            element_type=element_type,
            mesh_interval=[2, 2, 2],
            mesh=False,
            name=f"hex_{i_brick}",
        )
        cubit.cmd(f"move {fmt(brick)} x {i_brick * 2} y 0 z 0 include_merged")
        brick.mesh()
        cubit.add_node_set(brick.surfaces()[0], name=f"surface_brick_{i_brick}_top")
        cubit.add_node_set(brick.surfaces()[1], name=f"surface_brick_{i_brick}_bottom")

    def get_wedge_indices(cubit) -> set[int]:
        """Return the wedge element indices in cubit."""
        wedge_group = cubit.group(add_value="add wedge all")
        return set(cubit.get_group_wedges(wedge_group.id()))

    wedge_element_types = [cupy.element_type.wedge6]
    for i_wedge, element_type in enumerate(wedge_element_types):
        return_dict = cubit.cmd_return_dict(
            "create surface rectangle width 1 height 1 zplane",
            [cupy.geometry.vertex, cupy.geometry.surface],
        )
        surface = return_dict[cupy.geometry.surface][0]
        owning_volume = cubit.volume(cubit.get_owning_volume("surface", surface.id()))
        cubit.cmd(
            f"move {fmt(surface)} x {(len(brick_element_types) + i_wedge) * 2} y 0 z -0.5 include_merged"
        )
        vertices = return_dict[cupy.geometry.vertex]
        curve_1 = cubit.create_curve(vertices[0], vertices[2])
        curve_2 = cubit.create_curve(vertices[1], vertices[3])
        cubit.cmd(f"partition create {fmt(surface)} {fmt([curve_1, curve_2])}")
        surfaces = owning_volume.surfaces()
        cubit.cmd(f"{fmt(surfaces)} scheme trimesh")
        cubit.cmd(f"{fmt(surfaces)} size 2")
        tri_indices = set()
        for surface in surfaces:
            surface.mesh()
            tri_indices.update(cubit.get_surface_tris(surface.id()))
        tri_indices_str = " ".join(str(i) for i in tri_indices)
        wedge_indices_prior = get_wedge_indices(cubit)
        cubit.cmd(f"create element offset tri {tri_indices_str} distance 1.0 layers 2")
        wedge_indices_after = get_wedge_indices(cubit)

        wedges_added_str = " ".join(
            str(i) for i in wedge_indices_after - wedge_indices_prior
        )
        wedge_group = cubit.group(add_value=f"add wedge {wedges_added_str}")
        cubit.add_element_type(wedge_group, element_type, name=f"wedge_{i_wedge}")
        volume = cubit.cmd_return(
            f"create mesh geometry wedge {wedges_added_str} feature_angle 135.0",
            cupy.geometry.volume,
        )
        for name, condition in [
            ("top", "z_coord > 0.1"),
            ("bottom", "z_coord < -0.1"),
        ]:
            cubit.add_node_set(
                cubit.group(add_value=f"add surface in {fmt(volume)} with {condition}"),
                name=f"surface_wedge_{i_wedge}_{name}",
            )

    tet_element_types = [
        cupy.element_type.tet4,
        cupy.element_type.tet10,
    ]
    for i_tet, element_type in enumerate(tet_element_types):
        pyramid = cubit.pyramid(1.0, 3, 1.0, 1.0)
        cubit.cmd(
            f"move {fmt(pyramid)} x {(len(brick_element_types) + len(wedge_element_types) + i_tet) * 2} y 0 z 0 include_merged"
        )
        cubit.cmd(f"{fmt(pyramid)} size 2")
        cubit.cmd(f"{fmt(pyramid)} scheme tetmesh")
        pyramid.mesh()

        cubit.add_element_type(pyramid.volumes()[0], element_type, name=f"tet_{i_tet}")
        cubit.add_node_set(
            pyramid.surfaces()[0], name=f"surface_pyramid_{i_tet}_bottom"
        )

    # Set the material.
    cubit.fourc_input["MATERIALS"] = [
        {"MAT": 1, "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2}}
    ]

    return cubit
