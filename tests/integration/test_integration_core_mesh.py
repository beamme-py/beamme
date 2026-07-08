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
"""This script is used to test general functionality of the core mesh class with end-to-
end integration tests."""

import copy
import random
import warnings
from contextlib import nullcontext

import numpy as np
import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_2d,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def create_test_mesh(get_default_test_beam_material):
    """Create a mesh with a couple of test nodes and elements."""
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


def test_integration_core_mesh_rotation(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Check if the Mesh function rotation gives the same results as rotating each node
    it self."""
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

    # Compare with reference results.
    assert_results_close(mesh_1, get_corresponding_reference_file_path())


def test_integration_core_mesh_rotation_individual(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Check if the Mesh function rotation gives the same results as rotating each node
    it self, when an array is passed with different rotations."""
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

    # Compare with reference results.
    assert_results_close(mesh_1, get_corresponding_reference_file_path())


@pytest.mark.parametrize("origin", [False, True])
@pytest.mark.parametrize("flip", [False, True])
def test_integration_core_mesh_reflection(
    origin,
    flip,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Create a mesh, and its mirrored counterpart and then compare the input files."""
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
    assert_results_close(mesh_ref, mesh)

    # Compare with reference file.
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=("origin" if origin else "no_origin")
            + ("_flip" if flip else "_no_flip")
        ),
        mesh,
    )


def test_integration_core_mesh_couple_nodes(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the couple nodes functionality when replacing nodes."""
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    # Create the beam structure
    beam_set_1 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 10, 0], [0, 0, 0], n_el=2
    )
    beam_set_2 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 0, 0], [10, 0, 0], n_el=2
    )
    beam_set_3 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [10, 0, 0], [20, 0, 0], n_el=2
    )
    beam_set_4 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [20, 0, 0], [30, 0, 0], n_el=2
    )

    def create_arc():
        """Create an arc that starts at the end position of beam set 3."""
        mesh_arc = Mesh()
        arc_set = create_beam_mesh_arc_segment_2d(
            mesh_arc, Beam3rHerm2Line3, mat, [20, 20, 0], 20, -np.pi * 0.5, 0.0, n_el=2
        )
        return mesh_arc, arc_set

    arc_mesh, beam_set_5 = create_arc()
    mesh.add(arc_mesh)

    arc_mesh, beam_set_6 = create_arc()
    arc_mesh.rotate(Rotation([1, 0, 0], 0.5 * np.pi), origin=[20, 0, 0])
    mesh.add(arc_mesh)

    # Add all created sets to the mesh.
    mesh.add(beam_set_1, beam_set_2, beam_set_3, beam_set_4, beam_set_5, beam_set_6)

    # Couple the nodes and replace equal nodes.
    mesh.couple_nodes(reuse_matching_nodes=True)

    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize(
    ("import_full", "radius", "reflect", "context"),
    [
        (False, None, True, nullcontext()),
        (False, 0.2, True, nullcontext()),
        (True, 0.2, False, nullcontext()),
        (False, 666, False, pytest.raises(ValueError)),
        (True, None, False, pytest.raises(ValueError)),
    ],
)
def test_integration_core_mesh_transformations_with_solid(
    import_full,
    radius,
    reflect,
    context,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the different mesh transformation methods in combination with solid
    elements."""
    with context:
        # First, we create a line and wrap it with passing radius to the wrap function.

        # Create the mesh.
        input_file, mesh = import_four_c_model(
            input_file_path=get_corresponding_reference_file_path(
                reference_file_base_name="test_other_create_cubit_input_files_single_solid_element_brick"
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


def test_integration_core_mesh_wrap_cylinder_not_on_same_plane(
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


def test_integration_core_mesh_deep_copy(
    get_bc_data,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """This test checks that the deep copy function on a mesh does not copy the
    materials or functions."""
    # Create material and function object.
    mat = get_default_test_beam_material(material_type="reissner")
    fun = Function([{"COMPONENT": 0, "SYMBOLIC_FUNCTION_OF_SPACE_TIME": "t"}])

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
    assert_results_close(mesh, mesh_copy)
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_core_mesh_deep_copy_with_geometry_sets(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that deep-copying a mesh together with geometry sets referencing it works
    such that the copied geometry sets also reference the copied mesh."""
    mesh = Mesh()
    beam_set = create_beam_mesh_line(
        mesh=mesh,
        beam_class=Beam3rHerm2Line3,
        material=get_default_test_beam_material(material_type="reissner"),
        start_point=[0, 0, 0],
        end_point=[1, 0, 0],
    )

    # Deep-copy both mesh and beam_set to keep node/element references consistent
    mesh_copy, beam_set_copy = copy.deepcopy((mesh, beam_set))

    mesh.add(mesh_copy)
    mesh.add(beam_set)
    mesh.add(beam_set_copy)
    assert_results_close(get_corresponding_reference_file_path(), mesh)
