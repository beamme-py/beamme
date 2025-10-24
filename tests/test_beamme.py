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
"""This script is used to test the functionality of the core modules."""

import os
import random
import warnings
from unittest.mock import patch

import autograd.numpy as npAD
import numpy as np
import pytest
import splinepy
import vtk

import beamme.core.vtk_writer as vtk_writer
from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.coupling import Coupling
from beamme.core.element_beam import Beam
from beamme.core.function import Function
from beamme.core.geometry_set import GeometryName, GeometrySet, GeometrySetNodes
from beamme.core.mesh import Mesh
from beamme.core.node import Node, NodeCosserat
from beamme.core.rotation import Rotation
from beamme.core.vtk_writer import VTKWriter
from beamme.four_c.element_beam import (
    Beam3eb,
    Beam3rHerm2Line3,
)
from beamme.four_c.header_functions import (
    add_result_description,
    set_beam_to_solid_meshtying,
    set_header_static,
    set_runtime_output,
)
from beamme.four_c.input_file import InputFile
from beamme.four_c.material import (
    MaterialKirchhoff,
    MaterialReissnerElastoplastic,
    MaterialStVenantKirchhoff,
)
from beamme.four_c.model_importer import import_cubitpy_model, import_four_c_model
from beamme.mesh_creation_functions.applications.beam_honeycomb import (
    create_beam_mesh_honeycomb,
)
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)
from beamme.mesh_creation_functions.nurbs_generic import add_splinepy_nurbs_to_mesh
from beamme.utils.nodes import get_min_max_coordinates
from tests.create_cubit_input import create_tube_cubit


def create_test_mesh(get_default_test_beam_material):
    """Fill the mesh with a couple of test nodes and elements."""

    # Set the seed for the pseudo random numbers
    random.seed(0)

    # Add material to mesh.
    mesh = Mesh()
    material = get_default_test_beam_material(material_type="reissner")
    mesh.add(material)

    # Add three test nodes and add them to a beam element
    for _j in range(3):
        mesh.add(
            NodeCosserat(
                [100 * random.uniform(-1, 1) for _i in range(3)],
                Rotation(
                    [100 * random.uniform(-1, 1) for _i in range(3)],
                    100 * random.uniform(-1, 1),
                ),
            )
        )
    beam = Beam3rHerm2Line3(material=material, nodes=mesh.nodes)
    mesh.add(beam)

    # Add a beam line with three elements
    create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        material,
        [100 * random.uniform(-1, 1) for _i in range(3)],
        [100 * random.uniform(-1, 1) for _i in range(3)],
        n_el=3,
    )

    return mesh


def test_rotations(get_default_test_beam_material, assert_results_close):
    """Check if the Mesh function rotation gives the same results as rotating
    each node it self."""

    mesh_1 = create_test_mesh(get_default_test_beam_material)
    mesh_2 = create_test_mesh(get_default_test_beam_material)

    # Set the seed for the pseudo random numbers
    random.seed(0)
    rot = Rotation(
        [100 * random.uniform(-1, 1) for _i in range(3)],
        100 * random.uniform(-1, 1),
    )
    origin = [100 * random.uniform(-1, 1) for _i in range(3)]

    for node in mesh_1.nodes:
        node.rotate(rot, origin=origin)

    mesh_2.rotate(rot, origin=origin)

    # Compare the output for the two meshes.
    assert_results_close(mesh_1, mesh_2)


def test_mesh_rotations_individual(
    get_default_test_beam_material, assert_results_close
):
    """Check if the Mesh function rotation gives the same results as rotating
    each node it self, when an array is passed with different rotations."""

    mesh_1 = create_test_mesh(get_default_test_beam_material)
    mesh_2 = create_test_mesh(get_default_test_beam_material)

    # Set the seed for the pseudo random numbers
    random.seed(0)

    # Rotate each node with a different rotation
    rotations = np.zeros([len(mesh_1.nodes), 4])
    origin = [100 * random.uniform(-1, 1) for _i in range(3)]
    for j, node in enumerate(mesh_1.nodes):
        rot = Rotation(
            [100 * random.uniform(-1, 1) for _i in range(3)],
            100 * random.uniform(-1, 1),
        )
        rotations[j, :] = rot.get_quaternion()
        node.rotate(rot, origin=origin)

    mesh_2.rotate(rotations, origin=origin)

    # Compare the output for the two meshes.
    assert_results_close(mesh_1, mesh_2)


@pytest.mark.parametrize("origin", [False, True])
@pytest.mark.parametrize("flip", [False, True])
def test_mesh_reflection(
    origin, flip, get_default_test_beam_material, assert_results_close
):
    """Create a mesh, and its mirrored counterpart and then compare the input
    files."""

    # Rotations to be applied.
    rot_1 = Rotation([0, 1, 1], np.pi / 6)
    rot_2 = Rotation([1, 2.455, -1.2324], 1.2342352)

    mesh_ref = Mesh()
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    # Create the reference mesh.
    if not flip:
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0], n_el=1
        )
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0], n_el=1
        )
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [1, 1, 0], [1, 1, 1], n_el=1
        )
    else:
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [0, 0, 0], n_el=1
        )
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [1, 1, 0], [1, 0, 0], n_el=1
        )
        create_beam_mesh_line(
            mesh_ref, Beam3rHerm2Line3, mat, [1, 1, 1], [1, 1, 0], n_el=1
        )

        # Reorder the internal nodes.
        old = mesh_ref.nodes.copy()
        mesh_ref.nodes[0] = old[2]
        mesh_ref.nodes[2] = old[0]
        mesh_ref.nodes[3] = old[5]
        mesh_ref.nodes[5] = old[3]
        mesh_ref.nodes[6] = old[8]
        mesh_ref.nodes[8] = old[6]

    mesh_ref.rotate(rot_1)

    # Create the mesh that will be mirrored.
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [-1, 0, 0], n_el=1)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [-1, 0, 0], [-1, 1, 0], n_el=1)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [-1, 1, 0], [-1, 1, 1], n_el=1)
    mesh.rotate(rot_1.inv())

    # Rotate everything, to show generalized reflection.
    mesh_ref.rotate(rot_2)
    mesh.rotate(rot_2)

    if origin:
        # Translate everything so the reflection plane is not in the
        # origin.
        r = [1, 2.455, -1.2324]
        mesh_ref.translate(r)
        mesh.translate(r)
        mesh.reflect(2 * (rot_2 * [1, 0, 0]), origin=r, flip_beams=flip)
    else:
        mesh.reflect(2 * (rot_2 * [1, 0, 0]), flip_beams=flip)

    # Compare the input files.
    # TODO: Also add fixed result files to compare the tests with
    assert_results_close(mesh_ref, mesh)


def test_mesh_transformations_with_solid(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the different mesh transformation methods in combination with solid
    elements."""

    # TODO use pytest fixtures for the different setups

    def base_test_mesh_translations(*, import_full=False, radius=None, reflect=True):
        """Create the line and wrap it with passing radius to the wrap
        function."""

        # Create the mesh.
        input_file, mesh = import_four_c_model(
            input_file_path=get_corresponding_reference_file_path(
                reference_file_base_name="4C_input_solid_cuboid"
            ),
            convert_input_to_mesh=import_full,
        )

        mat = get_default_test_beam_material(material_type="reissner")

        # Create the line.
        create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0.2, 0, 0],
            [0.2, 5 * 0.2 * 2 * np.pi, 4],
            n_el=3,
        )

        # Transform the mesh.
        mesh.wrap_around_cylinder(radius=radius)
        mesh.translate([1, 2, 3])
        mesh.rotate(Rotation([1, 2, 3], np.pi * 17.0 / 27.0))
        if reflect:
            mesh.reflect([0.1, -2, 1])

        input_file.add(mesh)

        # Check the output.
        assert_results_close(
            get_corresponding_reference_file_path(
                additional_identifier="full" if import_full else "yaml"
            ),
            input_file,
        )

    base_test_mesh_translations(import_full=False, radius=None)
    base_test_mesh_translations(import_full=False, radius=0.2)
    base_test_mesh_translations(import_full=True, radius=0.2, reflect=False)

    # Not specifying or specifying the wrong radius should raise an error
    # In this case because everything is on one plane ("no" solid nodes in this case)
    # and we specify the radius
    with pytest.raises(ValueError):
        base_test_mesh_translations(import_full=False, radius=666, reflect=False)

    # In this case because we need to specify a radius because with the solid nodes there
    # is no clear radius
    with pytest.raises(ValueError):
        base_test_mesh_translations(
            import_full=True,
            radius=None,
            reflect=False,
        )


def test_fluid_element_section(
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
    material = get_default_test_beam_material(material_type="euler_bernoulli")
    beam_mesh.add(material)

    create_beam_mesh_line(
        beam_mesh, Beam3eb, material, [0, -0.5, 0], [0, 0.2, 0], n_el=5
    )

    input_file.add(beam_mesh)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_wrap_cylinder_not_on_same_plane(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix that is itself wrapped around a cylinder."""

    # Ignore the warnings from wrap around cylinder.
    warnings.filterwarnings("ignore")

    # Create the mesh.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    # Create the line and bend it to a helix.
    create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0.2, 0, 0],
        [0.2, 5 * 0.2 * 2 * np.pi, 4],
        n_el=20,
    )
    mesh.wrap_around_cylinder()

    # Move the helix so its axis is in the y direction and goes through
    # (2 0 0). The helix is also moved by a lot in y-direction, this only
    # affects the angle phi when wrapping around a cylinder, not the shape
    # of the beam.
    mesh.rotate(Rotation([1, 0, 0], -0.5 * np.pi))
    mesh.translate([2, 666.666, 0])

    # Wrap the helix again.
    mesh.wrap_around_cylinder(radius=2.0)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_get_nodes_by_function(get_default_test_beam_material, assert_results_close):
    """Check if the get_nodes_by_function method of Mesh works properly."""

    def get_nodes_at_x(node, x_value):
        """True for all coordinates at a certain x value."""
        if np.abs(node.coordinates[0] - x_value) < 1e-10:
            return True
        else:
            return False

    mat = get_default_test_beam_material(material_type="reissner")

    mesh = Mesh()
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [5, 0, 0], n_el=5)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 1, 0], [10, 1, 0], n_el=10)

    nodes = mesh.get_nodes_by_function(get_nodes_at_x, 1.0)
    assert 2 == len(nodes)
    for node in nodes:
        assert_results_close(1.0, node.coordinates[0])


def test_get_min_max_coordinates(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test if the get_min_max_coordinates function works properly."""

    # Create the mesh.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="4C_input_solid_cuboid"
        ),
        convert_input_to_mesh=True,
    )

    mat = get_default_test_beam_material(material_type="reissner")
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 3, 4], n_el=10)

    # Check the results.
    min_max = get_min_max_coordinates(mesh.nodes)
    ref_solution = [-0.5, -1.0, -1.5, 2.0, 3.0, 4.0]
    assert_results_close(min_max, ref_solution)


def test_geometry_sets(assert_results_close, get_corresponding_reference_file_path):
    """Test functionality of the GeometrySet objects."""

    mesh = Mesh()
    for i in range(6):
        mesh.add(NodeCosserat([i, 2 * i, 3 * i], Rotation()))

    set_1 = GeometrySetNodes(
        bme.geo.point, [mesh.nodes[0], mesh.nodes[1], mesh.nodes[2]]
    )
    set_2 = GeometrySetNodes(
        bme.geo.point, [mesh.nodes[2], mesh.nodes[3], mesh.nodes[4]]
    )
    set_12 = GeometrySetNodes(bme.geo.point)
    set_12.add(set_1)
    set_12.add(set_2)
    set_3 = GeometrySet(set_1.get_points())

    mesh.add(set_1, set_2, set_12, set_3)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_unique_ordering_of_get_all_nodes_for_line_condition(
    get_bc_data,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """This test ensures that the ordering of the nodes returned from the
    function get_all_nodes is unique for line sets."""

    # set up a beam mesh with material
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=10
    )

    # apply different Dirichlet conditions to all nodes within this condition
    for i, node in enumerate(beam_set["line"].get_all_nodes()):
        # add different condition value for each node
        mesh.add(
            BoundaryCondition(
                GeometrySet(node),
                get_bc_data(identifier=node.coordinates[0]),
                bc_type=bme.bc.dirichlet,
            )
        )

    # Check the input file
    assert_results_close(
        get_corresponding_reference_file_path(),
        mesh,
    )


def test_reissner_elasto_plastic(assert_results_close):
    """Test the elasto plastic Reissner beam material."""

    kwargs = {
        "radius": 0.1,
        "nu": 1.0,
        "density": 1.0,
        "youngs_modulus": 1000,
        "interaction_radius": 2.0,
        "shear_correction": 5.0 / 6.0,
        "yield_moment": 2.3,
        "isohardening_modulus_moment": 4.5,
        "torsion_plasticity": False,
    }

    ref_dict = {
        "MAT": 69,
        "MAT_BeamReissnerElastPlastic": {
            "YOUNG": 1000,
            "POISSONRATIO": 1.0,
            "DENS": 1.0,
            "CROSSAREA": 0.031415926535897934,
            "SHEARCORR": 0.8333333333333334,
            "MOMINPOL": 0.00015707963267948968,
            "MOMIN2": 7.853981633974484e-05,
            "MOMIN3": 7.853981633974484e-05,
            "INTERACTIONRADIUS": 2.0,
            "YIELDM": 2.3,
            "ISOHARDM": 4.5,
            "TORSIONPLAST": False,
        },
    }

    mat = MaterialReissnerElastoplastic(**kwargs)
    mat.i_global = 68

    assert_results_close(mat.dump_to_list(), ref_dict)

    ref_dict["MAT_BeamReissnerElastPlastic"]["TORSIONPLAST"] = True
    kwargs["torsion_plasticity"] = True
    mat = MaterialReissnerElastoplastic(**kwargs)
    mat.i_global = 68
    assert_results_close(mat.dump_to_list(), ref_dict)


def test_kirchhoff_material(assert_results_close):
    """Test the Kirchhoff Love beam material."""

    def set_stiff(material):
        """Set the material properties for the beam material."""
        material.area = 2.0
        material.mom2 = 3.0
        material.mom3 = 4.0
        material.polar = 5.0

    material = MaterialKirchhoff(
        youngs_modulus=1000, radius=1.0, nu=1.0, density=1.0, is_fad=True
    )
    material.i_global = 26
    set_stiff(material)
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 27,
            "MAT_BeamKirchhoffElastHyper": {
                "YOUNG": 1000,
                "SHEARMOD": 250.0,
                "DENS": 1.0,
                "CROSSAREA": 2.0,
                "MOMINPOL": 5.0,
                "MOMIN2": 3.0,
                "MOMIN3": 4.0,
                "FAD": True,
            },
        },
    )

    material = MaterialKirchhoff(
        youngs_modulus=1000, radius=1.0, nu=1.0, density=1.0, is_fad=False
    )
    material.i_global = 26
    set_stiff(material)
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 27,
            "MAT_BeamKirchhoffElastHyper": {
                "YOUNG": 1000,
                "SHEARMOD": 250.0,
                "DENS": 1.0,
                "CROSSAREA": 2.0,
                "MOMINPOL": 5.0,
                "MOMIN2": 3.0,
                "MOMIN3": 4.0,
                "FAD": False,
            },
        },
    )

    material = MaterialKirchhoff(
        youngs_modulus=1000, radius=1.0, nu=1.0, density=1.0, interaction_radius=1.1
    )
    material.i_global = 26
    set_stiff(material)
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 27,
            "MAT_BeamKirchhoffElastHyper": {
                "YOUNG": 1000,
                "SHEARMOD": 250.0,
                "DENS": 1.0,
                "CROSSAREA": 2.0,
                "MOMINPOL": 5.0,
                "MOMIN2": 3.0,
                "MOMIN3": 4.0,
                "FAD": False,
                "INTERACTIONRADIUS": 1.1,
            },
        },
    )


def test_close_beam(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """
    Create a circle with different methods.
    - Create the mesh manually by creating the nodes and connecting them to
        the elements.
    - Create one full circle and connect it to its beginning.
    - Create two half circle and connect their start / end nodes.
    All of those methods should give the exact same mesh.
    Both variants are also tried with different rotations at the beginning.
    """

    # Parameters for this test case.
    n_el = 3
    R = 1.235
    additional_rotation = Rotation([0, 1, 0], 0.5)

    # Define material.
    mat = get_default_test_beam_material(material_type="reissner")

    def create_mesh_manually(start_rotation):
        """Create the full circle manually."""
        mesh = Mesh()
        mesh.add(mat)

        # Add nodes.
        for i in range(4 * n_el):
            basis = start_rotation * Rotation([0, 0, 1], np.pi * 0.5)
            r = [R, 0, 0]
            node = NodeCosserat(r, basis)
            rotation = Rotation([0, 0, 1], 0.5 * i * np.pi / n_el)
            node.rotate(rotation, origin=[0, 0, 0])
            mesh.nodes.append(node)

        # Add elements.
        for i in range(2 * n_el):
            node_index = [2 * i, 2 * i + 1, 2 * i + 2]
            nodes = []
            for index in node_index:
                if index == len(mesh.nodes):
                    nodes.append(mesh.nodes[0])
                else:
                    nodes.append(mesh.nodes[index])
            element = Beam3rHerm2Line3(mat, nodes)
            mesh.add(element)

        # Add sets.
        geom_set = GeometryName()
        geom_set["start"] = GeometrySet(mesh.nodes[0])
        geom_set["end"] = GeometrySet(mesh.nodes[0])
        geom_set["line"] = GeometrySet(mesh.elements)
        mesh.add(geom_set)
        return mesh

    def one_full_circle_closed(function, argument_list, additional_rotation=None):
        """Create one full circle and connect it to itself."""

        mesh = Mesh()

        if additional_rotation is not None:
            start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
            mesh.add(NodeCosserat([R, 0, 0], start_rotation))
            beam_sets = function(
                mesh,
                start_node=mesh.nodes[0],
                close_beam=True,
                **(argument_list),
            )
        else:
            beam_sets = function(mesh, close_beam=True, **(argument_list))
        mesh.add(beam_sets)
        return mesh

    def two_half_circles_closed(function, argument_list, additional_rotation=None):
        """Create two half circles and close them, by reusing the connecting
        nodes."""

        mesh = Mesh()

        if additional_rotation is not None:
            start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
            mesh.add(NodeCosserat([R, 0, 0], start_rotation))
            set_1 = function(mesh, start_node=mesh.nodes[0], **(argument_list[0]))
        else:
            set_1 = function(mesh, **(argument_list[0]))

        set_2 = function(
            mesh,
            start_node=set_1["end"],
            end_node=set_1["start"],
            **(argument_list[1]),
        )

        # Add sets.
        geom_set = GeometryName()
        geom_set["start"] = GeometrySet(set_1["start"])
        geom_set["end"] = GeometrySet(set_2["end"])
        geom_set["line"] = GeometrySet([set_1["line"], set_2["line"]])
        mesh.add(geom_set)

        return mesh

    def get_arguments_arc_segment(circle_type):
        """Return the arguments for the arc segment function."""
        if circle_type == 0:
            # Full circle.
            arg_rot_angle = np.pi / 2
            arg_angle = 2 * np.pi
            arg_n_el = 2 * n_el
        elif circle_type == 1:
            # First half circle.
            arg_rot_angle = np.pi / 2
            arg_angle = np.pi
            arg_n_el = n_el
        elif circle_type == 2:
            # Second half circle.
            arg_rot_angle = 3 * np.pi / 2
            arg_angle = np.pi
            arg_n_el = n_el
        return {
            "beam_class": Beam3rHerm2Line3,
            "material": mat,
            "center": [0, 0, 0],
            "axis_rotation": Rotation([0, 0, 1], arg_rot_angle),
            "radius": R,
            "angle": arg_angle,
            "n_el": arg_n_el,
        }

    def circle_function(t):
        """Function for the circle."""
        return R * npAD.array([npAD.cos(t), npAD.sin(t)])

    def get_arguments_curve(circle_type):
        """Return the arguments for the curve function."""
        if circle_type == 0:
            # Full circle.
            arg_interval = [0, 2 * np.pi]
            arg_n_el = 2 * n_el
        elif circle_type == 1:
            # First half circle.
            arg_interval = [0, np.pi]
            arg_n_el = n_el
        elif circle_type == 2:
            # Second half circle.
            arg_interval = [np.pi, 2 * np.pi]
            arg_n_el = n_el
        return {
            "beam_class": Beam3rHerm2Line3,
            "material": mat,
            "function": circle_function,
            "interval": arg_interval,
            "n_el": arg_n_el,
        }

    # Check the meshes without additional rotation.
    assert_results_close(
        get_corresponding_reference_file_path(), create_mesh_manually(Rotation())
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        one_full_circle_closed(
            create_beam_mesh_arc_segment_via_rotation, get_arguments_arc_segment(0)
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        two_half_circles_closed(
            create_beam_mesh_arc_segment_via_rotation,
            [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        one_full_circle_closed(
            create_beam_mesh_parametric_curve, get_arguments_curve(0)
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(),
        two_half_circles_closed(
            create_beam_mesh_parametric_curve,
            [get_arguments_curve(1), get_arguments_curve(2)],
        ),
    )

    # Check the meshes with additional rotation.
    additional_identifier = "rotation"
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        create_mesh_manually(additional_rotation),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        one_full_circle_closed(
            create_beam_mesh_arc_segment_via_rotation,
            get_arguments_arc_segment(0),
            additional_rotation=additional_rotation,
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        two_half_circles_closed(
            create_beam_mesh_arc_segment_via_rotation,
            [get_arguments_arc_segment(1), get_arguments_arc_segment(2)],
            additional_rotation=additional_rotation,
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        one_full_circle_closed(
            create_beam_mesh_parametric_curve,
            get_arguments_curve(0),
            additional_rotation=additional_rotation,
        ),
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        two_half_circles_closed(
            create_beam_mesh_parametric_curve,
            [get_arguments_curve(1), get_arguments_curve(2)],
            additional_rotation=additional_rotation,
        ),
    )


def test_geometry_set_get_geometry_objects(get_default_test_beam_material):
    """Test if the geometry set returns the objects(elements) in the correct
    order."""

    # Initialize material and mesh
    mat = get_default_test_beam_material(material_type="reissner")
    mesh = Mesh()

    # number of elements
    n_el = 5

    # Create a simple beam.
    geometry = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=n_el
    )

    # Get all elements from the geometry set.
    elements_of_geometry = geometry["line"].get_geometry_objects()

    # Check number of elements.
    assert len(elements_of_geometry) == n_el

    # Check if the order of the elements from the geometry set is the same as for the mesh.
    for i_element, element in enumerate(elements_of_geometry):
        assert element == mesh.elements[i_element]


@pytest.mark.parametrize("use_nodal_geometry_sets", [True, False])
def test_replace_nodes_geometry_set(
    get_bc_data,
    get_default_test_beam_material,
    use_nodal_geometry_sets,
    assert_results_close,
):
    """Test case for coupling of nodes, and reusing the identical nodes."""

    bme.check_overlapping_elements = False

    mat = get_default_test_beam_material(material_type="reissner")
    rot = Rotation([1, 2, 43], 213123)

    # Create a beam with two elements. Once immediately and once as two
    # beams with couplings.
    mesh_ref = Mesh()
    mesh_couple = Mesh()

    # Create a simple beam.
    create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])

    ref_nodes = list(mesh_ref.nodes)
    coupling_nodes = list(mesh_couple.nodes)

    # Add a set with all nodes, to check that the nodes in the
    # boundary condition are replaced correctly.
    if use_nodal_geometry_sets:
        mesh_ref.add(GeometrySetNodes(bme.geo.line, ref_nodes))
        mesh_ref.add(GeometrySetNodes(bme.geo.point, ref_nodes))
        mesh_couple.add(GeometrySetNodes(bme.geo.line, coupling_nodes))
        mesh_couple.add(GeometrySetNodes(bme.geo.point, coupling_nodes))
    else:
        mesh_ref.add(GeometrySet(mesh_ref.elements))
        mesh_ref.add(GeometrySet(ref_nodes))
        mesh_couple.add(GeometrySet(mesh_couple.elements))
        mesh_couple.add(GeometrySet(coupling_nodes))

    # Add another set with all nodes, this time only the coupling node
    # that will be kept is in this set.
    coupling_nodes_without_replace_node = list(coupling_nodes)
    del coupling_nodes_without_replace_node[3]
    if use_nodal_geometry_sets:
        mesh_ref.add(GeometrySetNodes(bme.geo.point, ref_nodes))
        mesh_couple.add(
            GeometrySetNodes(bme.geo.point, coupling_nodes_without_replace_node)
        )
    else:
        mesh_ref.add(GeometrySet(ref_nodes))
        mesh_couple.add(GeometrySet(coupling_nodes_without_replace_node))

    # Add another set with all nodes, this time only the coupling node
    # that will be replaced is in this set.
    coupling_nodes_without_replace_node = list(coupling_nodes)
    del coupling_nodes_without_replace_node[2]
    if use_nodal_geometry_sets:
        mesh_ref.add(GeometrySetNodes(bme.geo.point, ref_nodes))
        mesh_couple.add(
            GeometrySetNodes(bme.geo.point, coupling_nodes_without_replace_node)
        )
    else:
        mesh_ref.add(GeometrySet(ref_nodes))
        mesh_couple.add(GeometrySet(coupling_nodes_without_replace_node))

    # Rotate both meshes
    mesh_ref.rotate(rot)
    mesh_couple.rotate(rot)

    # Couple the coupling mesh.
    mesh_couple.couple_nodes(
        coupling_dof_type=bme.coupling_dof.fix, reuse_matching_nodes=True
    )

    # Compare the meshes.
    assert_results_close(mesh_ref, mesh_couple)

    # Create two overlapping beams. This is to test that the middle nodes
    # are not coupled.
    mesh_ref = Mesh()
    mesh_couple = Mesh()

    # Create a simple beam.
    set_ref = create_beam_mesh_line(
        mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0]
    )
    create_beam_mesh_line(
        mesh_ref,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0],
        [1, 0, 0],
        start_node=set_ref["start"],
        end_node=set_ref["end"],
    )
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])

    # Rotate both meshes
    mesh_ref.rotate(rot)
    mesh_couple.rotate(rot)

    # Couple the coupling mesh.
    mesh_couple.couple_nodes(
        coupling_dof_type=bme.coupling_dof.fix, reuse_matching_nodes=True
    )

    # Compare the meshes.
    assert_results_close(mesh_ref, mesh_couple)

    # Create a beam with two elements. Once immediately and once as two
    # beams with couplings.
    mesh_ref = Mesh()
    mesh_couple = Mesh()

    # Create a simple beam.
    create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 0, 0])

    # Create set with all the beam nodes.
    if use_nodal_geometry_sets:
        node_set_1_ref = GeometrySetNodes(bme.geo.line, mesh_ref.nodes)
        node_set_2_ref = GeometrySetNodes(bme.geo.line, mesh_ref.nodes)
        node_set_1_couple = GeometrySetNodes(bme.geo.line, mesh_couple.nodes)
        node_set_2_couple = GeometrySetNodes(bme.geo.line, mesh_couple.nodes)
    else:
        node_set_1_ref = GeometrySet(mesh_ref.elements)
        node_set_2_ref = GeometrySet(mesh_ref.elements)
        node_set_1_couple = GeometrySet(mesh_couple.elements)
        node_set_2_couple = GeometrySet(mesh_couple.elements)

    # Create connecting beams.
    create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 2, 2])
    create_beam_mesh_line(mesh_ref, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -2, -2])
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, 2, 2])
    create_beam_mesh_line(mesh_couple, Beam3rHerm2Line3, mat, [1, 0, 0], [2, -2, -2])

    # Rotate both meshes
    mesh_ref.rotate(rot)
    mesh_couple.rotate(rot)

    # Couple the mesh.
    mesh_ref.couple_nodes(coupling_dof_type=bme.coupling_dof.fix)
    mesh_couple.couple_nodes(
        coupling_dof_type=bme.coupling_dof.fix, reuse_matching_nodes=True
    )

    # Add the node sets.
    mesh_ref.add(node_set_1_ref)
    mesh_couple.add(node_set_1_couple)

    # Add BCs.
    mesh_ref.add(
        BoundaryCondition(node_set_2_ref, get_bc_data(), bc_type=bme.bc.neumann)
    )
    mesh_couple.add(
        BoundaryCondition(node_set_2_couple, get_bc_data(), bc_type=bme.bc.neumann)
    )

    # Compare the meshes.
    # TODO: Add reference mesh for this test
    assert_results_close(mesh_ref, mesh_couple)


def create_beam_to_solid_conditions_model(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    full_import: bool,
):
    """Create the input file for the beam-to-solid input conditions tests."""

    # Create input file
    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_block"
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


# TODO: Standardize test parameterization for (full_import, additional_identifier).
# Currently, different tests use inconsistent patterns for parametrize:
#   - (False, "dict_import"), (True, "full_import")
#   - (False, None), (True, "full")
#   - Only "full_import" as a boolean param
# Consider unifying these under a shared fixture or helper to reduce redundancy
# and improve readability across tests. Also adjust reference file names.
@pytest.mark.parametrize(
    ("full_import", "additional_identifier"),
    [(False, None), (True, "full")],
)
def test_beam_to_solid_conditions(
    full_import,
    additional_identifier,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create the input file for the beam-to-solid input conditions tests."""

    # Get the input file.
    input_file, mesh = create_beam_to_solid_conditions_model(
        get_default_test_beam_material,
        get_corresponding_reference_file_path,
        full_import=full_import,
    )
    input_file.add(mesh)

    # Check results
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        input_file,
    )


def test_surface_to_surface_contact_import(
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


def test_nurbs_import(
    get_default_test_beam_material,
    get_default_test_solid_material,
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
    element_description = {"KINEM": "nonlinear"}

    volume_set = GeometrySetNodes(geometry_type=bme.geo.volume)
    fix_set = GeometrySetNodes(geometry_type=bme.geo.surface)
    for i in range(3):
        patch_set = add_splinepy_nurbs_to_mesh(
            mesh, extruded, material=mat, data=element_description
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
    # We need this to math the "old" result description
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


def test_stvenantkirchhoff_solid(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test that the input file for a solid with St.

    Venant Kirchhoff material properties is generated correctly
    """

    # Create materials
    material_1 = MaterialStVenantKirchhoff(youngs_modulus=157, nu=0.17, density=6.1e-7)

    material_2 = MaterialStVenantKirchhoff(youngs_modulus=370, nu=0.20, density=5.2e-7)

    # Create mesh
    mesh = Mesh()
    mesh.add(material_1)
    mesh.add(material_2)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize(
    "coupling_type",
    [
        ["exact", bme.bc.point_coupling, bme.coupling_dof.fix],
        [
            "penalty",
            bme.bc.point_coupling_penalty,
            {
                "POSITIONAL_PENALTY_PARAMETER": 10000,
                "ROTATIONAL_PENALTY_PARAMETER": 0,
            },
        ],
    ],
)
def test_point_couplings(
    get_default_test_beam_material,
    coupling_type,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create the input file for the test_point_couplings method."""

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
        coupling_type=coupling_type[1],
        coupling_dof_type=coupling_type[2],
    )

    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=coupling_type[0]),
        mesh,
    )


def test_point_couplings_check():
    """Test that the check for points at the same spatial position works for
    point couplings."""

    def get_nodes(scale_factor):
        """Return a list with nodes to be added to a coupling condition.

        The coordinates are modified such that they are close to each
        other within a radius of bme.eps_pos * scale_factor
        """
        coordinates = np.zeros((10, 3))
        ref_point = [1, 2, 3]
        coordinates[0] = ref_point
        for i in range(1, 10):
            coordinates[i] = ref_point
            for i_dir in range(3):
                factor = 2 * ((i + i_dir % 3) % 2) - 1
                # Multiply with 0.5 here, because we add the tolerance in + and - direction
                coordinates[i, i_dir] += factor * bme.eps_pos * scale_factor
        return [Node(coord) for coord in coordinates]

    # This should work, as the points are within the global tolerance of each
    # other
    Coupling(get_nodes(0.5), None, None)

    with pytest.raises(ValueError):
        # This should fail, as the points are not within the global tolerance
        # of each other
        Coupling(get_nodes(1.0), None, None)

    # This should work, as the points are not within the global tolerance of
    # each other but we dont perform the check
    Coupling(get_nodes(1.0), None, None, check_overlapping_nodes=False)


def test_vtk_writer(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Test the output created by the VTK writer."""

    # Initialize writer.
    writer = VTKWriter()

    # Add poly line.
    indices = writer.add_points([[0, 0, -2], [1, 1, -2], [2, 2, -1]])
    writer.add_cell(vtk.vtkPolyLine, indices)

    # Add quadratic quad.
    cell_data = {}
    cell_data["cell_data_1"] = 3
    cell_data["cell_data_2"] = [66, 0, 1]
    point_data = {}
    point_data["point_data_1"] = [1, 2, 3, 4, 5, -2, -3, 0]
    point_data["point_data_2"] = [
        [0.25, 0, -0.25],
        [1, 0.25, 0],
        [2, 0, 0],
        [2.25, 1.25, 0.5],
        [2, 2.25, 0],
        [1, 2, 0.5],
        [0, 2.25, 0],
        [0, 1, 0.5],
    ]
    indices = writer.add_points(
        [
            [0.25, 0, -0.25],
            [1, 0.25, 0],
            [2, 0, 0],
            [2.25, 1.25, 0.5],
            [2, 2.25, 0],
            [1, 2, 0.5],
            [0, 2.25, 0],
            [0, 1, 0.5],
        ],
        point_data=point_data,
    )
    writer.add_cell(
        vtk.vtkQuadraticQuad, indices[[0, 2, 4, 6, 1, 3, 5, 7]], cell_data=cell_data
    )

    # Add tetrahedron.
    cell_data = {}
    cell_data["cell_data_2"] = [5, 0, 10]
    point_data = {}
    point_data["point_data_1"] = [1, 2, 3, 4]
    indices = writer.add_points(
        [[3, 3, 3], [4, 4, 3], [4, 3, 3], [4, 4, 4]], point_data=point_data
    )
    writer.add_cell(vtk.vtkTetra, indices[[0, 2, 1, 3]], cell_data=cell_data)

    # Before we can write the data to file we have to store the cell and
    # point data in the grid
    writer.complete_data()

    # Write to file.
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / ref_file.name
    writer.write_vtk(vtk_file, binary=False)

    # Compare the vtk files.
    assert_results_close(ref_file, vtk_file)


def test_vtk_writer_beam(
    assert_results_close,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Create a sample mesh and check the VTK output."""

    # Create the mesh.
    mesh = Mesh()

    # Add content to the mesh.
    honeycomb_set = create_beam_mesh_honeycomb(
        mesh,
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="base"),
        2.0,
        2,
        3,
        n_el=2,
    )
    mesh.add(honeycomb_set)

    # Write VTK output, with coupling sets."""
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / ref_file.name
    mesh.write_vtk(
        output_name="test_vtk_writer",
        coupling_sets=True,
        output_directory=tmp_path,
        binary=False,
    )
    assert_results_close(ref_file, vtk_file, atol=bme.eps_pos)

    # Write VTK output, without coupling sets."""
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="no_coupling_beam", extension="vtu"
    )
    vtk_file = tmp_path / ref_file.name
    mesh.write_vtk(
        output_name="test_vtk_writer_beam_no_coupling",
        coupling_sets=False,
        output_directory=tmp_path,
        binary=False,
    )
    assert_results_close(ref_file, vtk_file, atol=bme.eps_pos)

    # Write VTK output, with coupling sets and additional points for visualization."""
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="smooth_centerline_beam", extension="vtu"
    )
    vtk_file = tmp_path / ref_file.name
    mesh.write_vtk(
        output_name="test_vtk_writer_beam_smooth_centerline",
        coupling_sets=True,
        output_directory=tmp_path,
        binary=False,
        beam_centerline_visualization_segments=3,
    )
    assert_results_close(ref_file, vtk_file, atol=bme.eps_pos)


def test_vtk_writer_solid(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Import a solid mesh and check the VTK output."""

    # Convert the solid mesh to beamme objects.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_tube"
        ),
        convert_input_to_mesh=True,
    )

    # Write VTK output.
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / ref_file.name
    if os.path.isfile(vtk_file):  # Todo: Can this check be removed?
        os.remove(vtk_file)
    mesh.write_vtk(
        output_name="test_vtk_writer", output_directory=tmp_path, binary=False
    )

    # Compare the vtk files.
    assert_results_close(ref_file, vtk_file)


def test_vtk_writer_solid_elements(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Import a solid mesh with all solid types and check the VTK output."""

    # Convert the solid mesh to beamme objects.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="import"
        ),
        convert_input_to_mesh=True,
    )

    # Write VTK output.
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="solid", extension="vtu"
    )
    vtk_file = tmp_path / ref_file.name
    if os.path.isfile(vtk_file):  # Todo: Can this check be removed?
        os.remove(vtk_file)
    mesh.write_vtk(
        output_name="test_vtk_writer_solid_elements",
        output_directory=tmp_path,
        binary=False,
    )

    # Compare the vtk files.
    assert_results_close(ref_file, vtk_file)


def test_vtk_curve_cell_data(
    assert_results_close,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Test that when creating a beam, cell data can be given.

    This test also checks, that the nan values in vtk can be explicitly
    given.
    """

    with (
        patch("beamme.core.vtk_writer.VTK_NAN_FLOAT", 69.69),
        patch("beamme.core.vtk_writer.VTK_NAN_INT", 69),
    ):
        # Create the mesh.
        mesh = Mesh()

        # Add content to the mesh.
        mat = get_default_test_beam_material(material_type="base")
        create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
        create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 1, 0],
            [2, 1, 0],
            n_el=2,
            vtk_cell_data={"cell_data": (1, vtk_writer.VTKType.int)},
        )
        create_beam_mesh_arc_segment_via_rotation(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 2, 0],
            Rotation([1, 0, 0], np.pi),
            1.5,
            np.pi / 2.0,
            n_el=2,
            vtk_cell_data={"cell_data": (2, vtk_writer.VTKType.int), "other_data": 69},
        )

        # Write VTK output, with coupling sets."""
        ref_file = get_corresponding_reference_file_path(
            additional_identifier="beam", extension="vtu"
        )
        vtk_file = tmp_path / ref_file.name
        mesh.write_vtk(
            output_name="test_vtk_curve_cell_data",
            output_directory=tmp_path,
            binary=False,
        )

        # Compare the vtk files.
        assert_results_close(ref_file, vtk_file)


@pytest.mark.parametrize("full_import", [False, True])
@pytest.mark.cubitpy
def test_cubitpy_import(
    full_import, assert_results_close, get_corresponding_reference_file_path
):
    """Check that an import from a cubitpy object works as expected."""

    cubit = create_tube_cubit()
    input_file_cubit, mesh = import_cubitpy_model(
        cubit, convert_input_to_mesh=full_import
    )
    if full_import:
        input_file_cubit.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_create_cubit_input_tube"
        ),
        input_file_cubit,
    )


def test_deep_copy(get_bc_data, get_default_test_beam_material, assert_results_close):
    """This test checks that the deep copy function on a mesh does not copy the
    materials or functions."""

    # Create material and function object.
    mat = get_default_test_beam_material(material_type="reissner")
    fun = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")

    def create_mesh(mesh):
        """Add material and function to the mesh and create a beam."""
        mesh.add(fun, mat)
        set1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
        set2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0])
        mesh.add(
            BoundaryCondition(
                set1["line"], get_bc_data(identifier=1), bc_type=bme.bc.dirichlet
            )
        )
        mesh.add(
            BoundaryCondition(
                set2["line"], get_bc_data(identifier=2), bc_type=bme.bc.neumann
            )
        )
        mesh.couple_nodes()

    # The second mesh will be translated and rotated with those vales.
    translate = [1.0, 2.34535435, 3.345353]
    rotation = Rotation([1, 0.2342342423, -2.234234], np.pi / 15 * 27)

    # First create the mesh twice, move one and get the input file.
    mesh_ref_1 = Mesh()
    mesh_ref_2 = Mesh()
    create_mesh(mesh_ref_1)
    create_mesh(mesh_ref_2)
    mesh_ref_2.rotate(rotation)
    mesh_ref_2.translate(translate)

    mesh = Mesh()
    mesh.add(mesh_ref_1, mesh_ref_2)

    # Now copy the first mesh and add them together in the input file.
    mesh_copy_1 = Mesh()
    create_mesh(mesh_copy_1)
    mesh_copy_2 = mesh_copy_1.copy()
    mesh_copy_2.rotate(rotation)
    mesh_copy_2.translate(translate)

    mesh_copy = Mesh()
    mesh_copy.add(mesh_copy_1, mesh_copy_2)

    # Check that the input files are the same.
    # TODO: add reference file check here as well
    assert_results_close(mesh, mesh_copy)


def test_mesh_add_checks():
    """This test checks that Mesh raises an error when double objects are added
    to the mesh."""

    # Mesh instance for this test.
    mesh = Mesh()

    # Create basic objects that will be added to the mesh.
    node = Node([0, 1.0, 2.0])
    element = Beam()
    mesh.add(node)
    mesh.add(element)

    # Create objects based on basic mesh items.
    coupling = Coupling(mesh.nodes, bme.bc.point_coupling, bme.coupling_dof.fix)
    coupling_penalty = Coupling(
        mesh.nodes, bme.bc.point_coupling_penalty, bme.coupling_dof.fix
    )
    geometry_set = GeometrySet(mesh.elements)
    mesh.add(coupling)
    mesh.add(coupling_penalty)
    mesh.add(geometry_set)

    # Add the objects again and check for errors.
    # TODO catch and test error messages
    with pytest.raises(ValueError):
        mesh.add(node)
    with pytest.raises(ValueError):
        mesh.add(element)
    with pytest.raises(ValueError):
        mesh.add(coupling)
    with pytest.raises(ValueError):
        mesh.add(coupling_penalty)
    with pytest.raises(ValueError):
        mesh.add(geometry_set)


def test_check_two_couplings(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """The current implementation can handle more than one coupling on a node
    correctly, therefore we check this here."""

    # Create mesh object
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, 0], [1, 1, 0])

    # Call coupling twice -> this will create two coupling objects for the
    # corner node
    mesh.couple_nodes()
    mesh.couple_nodes()

    # Create the input file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize("reuse_nodes", [[None, False], ["reuse", True]])
def test_check_multiple_node_penalty_coupling(
    get_default_test_beam_material,
    reuse_nodes,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """For point penalty coupling constraints, we add multiple coupling
    conditions.

    This is checked in this test case. This method creates the flag
    reuse_nodes decides if equal nodes are unified to a single node.
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
        reuse_matching_nodes=reuse_nodes[1],
        coupling_type=bme.bc.point_coupling_penalty,
        coupling_dof_type={
            "POSITIONAL_PENALTY_PARAMETER": 10000,
            "ROTATIONAL_PENALTY_PARAMETER": 0,
        },
    )
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=reuse_nodes[0]),
        mesh,
    )


def test_check_double_elements(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Check if there are overlapping elements in a mesh."""

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection.
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])

    # Rotate the mesh with an arbitrary rotation.
    mesh.rotate(Rotation([1, 2, 3.24313], 2.2323423), [1, 3, -2.23232323])

    # The elements in the created mesh are overlapping, check that an error
    # is thrown.
    with pytest.raises(ValueError):
        mesh.check_overlapping_elements()

    # Check if the overlapping elements are written to the vtk output.
    warnings.filterwarnings("ignore")
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="beam", extension="vtu"
    )
    vtk_file = tmp_path / ref_file.name
    mesh.write_vtk(
        output_name="test_check_double_elements",
        output_directory=tmp_path,
        binary=False,
        overlapping_elements=True,
    )

    # Compare the vtk files.
    assert_results_close(ref_file, vtk_file)


@pytest.mark.parametrize("check", [True, False])
def test_check_overlapping_coupling_nodes(get_default_test_beam_material, check):
    """Per default, we check that coupling nodes are at the same physical
    position.

    This check can be deactivated with the keyword
    check_overlapping_nodes when creating a Coupling.
    """

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection.
    set_1 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0])
    set_2 = create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [2, 0, 0], [3, 0, 0])

    # Couple two nodes that are not at the same position.

    # Create the input file. This will cause an error, as there are two
    # couplings for one node.
    args = [
        [set_1["start"], set_2["end"]],
        bme.bc.point_coupling,
        "coupling_type_string",
    ]
    if check:
        with pytest.raises(ValueError):
            Coupling(*args)
    else:
        Coupling(*args, check_overlapping_nodes=False)


def test_check_start_end_node_error(
    get_default_test_beam_material,
):
    """Check that an error is raised if wrong start and end nodes are given to
    a mesh creation function."""

    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    # Try to create a line with a starting node that is not in the mesh.
    node = NodeCosserat([0, 0, 0], Rotation())
    args = [mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [1, 0, 0]]
    kwargs = {"start_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)
    node.coordinates = [1, 0, 0]
    kwargs = {"end_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)


def test_userdefined_boundary_condition(
    get_bc_data,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Check if an user defined boundary condition can be added."""

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


def test_display_pyvista(
    get_default_test_beam_material, get_corresponding_reference_file_path
):
    """Test that the display in pyvista function does not lead to errors.

    TODO: Add a check for the created visualziation
    """

    _, mesh = create_beam_to_solid_conditions_model(
        get_default_test_beam_material,
        get_corresponding_reference_file_path,
        full_import=True,
    )

    mesh.display_pyvista(resolution=3)
