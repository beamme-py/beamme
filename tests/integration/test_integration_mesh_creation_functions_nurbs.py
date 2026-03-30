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
"""This script is used to test the NURBS mesh creation functions."""

import numpy as np
import splinepy

from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_solid import get_four_c_solid
from beamme.mesh_creation_functions.nurbs_generic import (
    add_geomdl_nurbs_to_mesh,
    add_splinepy_nurbs_to_mesh,
)
from beamme.mesh_creation_functions.nurbs_geometries import (
    create_nurbs_brick,
    create_nurbs_cylindrical_shell_sector,
    create_nurbs_flat_plate_2d,
    create_nurbs_hemisphere_surface,
    create_nurbs_hollow_cylinder_segment_2d,
    create_nurbs_sphere_surface,
    create_nurbs_torus_surface,
)
from beamme.mesh_creation_functions.nurbs_utils import (
    ensure_3d_splinepy_object,
    translate_splinepy,
)


def test_integration_mesh_creation_functions_nurbs_hollow_cylinder_segment_2d(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a two dimensional hollow cylinder segment."""

    # Create the surface of a quarter of a hollow cylinder
    surf_obj = create_nurbs_hollow_cylinder_segment_2d(
        1.74, 2.46, np.pi * 5 / 6, n_ele_u=2, n_ele_v=3
    )

    # Create mesh
    mesh = Mesh()

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        surf_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_flat_plate_2d(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a two dimensional flat plate."""

    # Create the surface of a flat plate
    surf_obj = create_nurbs_flat_plate_2d(0.75, 0.91, n_ele_u=2, n_ele_v=5)

    # Create mesh
    mesh = Mesh()

    # Add material
    mat = get_default_test_solid_material(material_type="2d_shell")

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_shell")
    patch_set = add_geomdl_nurbs_to_mesh(mesh, element_type, surf_obj, material=mat)

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_flat_plate_2d_splinepy(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a two dimensional flat plate with splinepy."""

    # Create the surface of a flat plate
    n_ele_u = 2
    n_ele_v = 5
    surf_obj = splinepy.helpme.create.box(0.75, 0.91).nurbs
    surf_obj.elevate_degrees([0, 1])
    surf_obj.insert_knots(0, np.linspace(0, 1, n_ele_u + 1))
    surf_obj.insert_knots(1, np.linspace(0, 1, n_ele_v + 1))
    ensure_3d_splinepy_object(surf_obj)
    translate_splinepy(surf_obj, -0.5 * np.array([0.75, 0.91, 0]))

    # Create the shell mesh
    mesh = Mesh()
    mat = get_default_test_solid_material(material_type="2d_shell")
    element_type = get_default_test_solid_element("nurbs_shell")
    patch_set = add_splinepy_nurbs_to_mesh(mesh, element_type, surf_obj, material=mat)
    mesh.add(patch_set)
    assert_results_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_integration_mesh_creation_functions_nurbs_flat_plate_2d"
        ),
        mesh,
    )


def test_integration_mesh_creation_functions_nurbs_flat_plate_2d_splinepy_copy(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that a mesh created from a splinepy NURBS can be copied."""

    # Create a flat plate
    surf_obj = splinepy.helpme.create.box(0.5, 1.0).nurbs
    surf_obj.elevate_degrees([0, 1])
    ensure_3d_splinepy_object(surf_obj)

    # Create mesh
    mesh = Mesh()
    mat = get_default_test_solid_material(material_type="2d_shell")
    element_type = get_default_test_solid_element("nurbs_shell")
    add_splinepy_nurbs_to_mesh(mesh, element_type, surf_obj, material=mat)

    mesh_copy = mesh.copy()
    mesh_copy.translate([3, 0, 0])
    mesh.add(mesh_copy)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_brick(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a brick."""

    # Create a brick
    vol_obj = create_nurbs_brick(1.5, 3.0, 2.4, n_ele_u=2, n_ele_v=3, n_ele_w=4)

    # Create mesh
    mesh = Mesh()

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_3d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        vol_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_brick_splinepy(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a brick with splinepy."""

    # Create a brick
    n_el_dim = [2, 3, 4]
    box_dimensions = [1.5, 3.0, 2.4]
    vol_obj = splinepy.helpme.create.box(*box_dimensions).nurbs
    vol_obj.elevate_degrees([0, 1, 2])
    for i_dim, n_el in enumerate(n_el_dim):
        vol_obj.insert_knots(i_dim, np.linspace(0, 1, n_el + 1))

    control_points_3d = np.zeros([len(vol_obj.control_points), 3])
    control_points_3d[:, :3] = vol_obj.control_points
    vol_obj.control_points = control_points_3d - 0.5 * np.array(box_dimensions)

    # Create mesh
    mesh = Mesh()

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_3d")
    patch_set = add_splinepy_nurbs_to_mesh(
        mesh,
        element_type,
        vol_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_integration_mesh_creation_functions_nurbs_brick"
        ),
        mesh,
    )


def test_integration_mesh_creation_functions_nurbs_rotation_nurbs_surface(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the rotation of a NURBS mesh."""

    # Create the surface
    surf_obj = create_nurbs_hollow_cylinder_segment_2d(
        1.74, 2.46, np.pi * 3 / 4, n_ele_u=5, n_ele_v=2
    )

    # Create mesh
    mesh = Mesh()

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        surf_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    mesh.rotate(Rotation([1, 2, 3], np.pi * 7 / 6))

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_translate_nurbs_surface(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the translation of a NURBS surface mesh."""

    # Create the surface
    surf_obj = create_nurbs_flat_plate_2d(0.87, 1.35, n_ele_u=2, n_ele_v=3)

    # Create mesh
    mesh = Mesh()

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        surf_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    mesh.translate([-1.6, -2.3, 3.7])

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_cylindrical_shell_sector(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a 3-dimensional cylindrical shell sector."""

    # Create the surface of a quarter of a hollow cylinder
    surf_obj = create_nurbs_cylindrical_shell_sector(
        2.3, np.pi / 3, 1.7, n_ele_u=3, n_ele_v=5
    )

    # Create mesh
    mesh = Mesh()

    # Create patch set
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        surf_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_couple_nurbs_meshes(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the coupling of NURBS surface meshes."""

    # Create mesh
    mesh = Mesh()

    # Create the first surface object
    surf_obj_1 = create_nurbs_hollow_cylinder_segment_2d(
        0.65, 1.46, np.pi * 2 / 3, n_ele_u=3, n_ele_v=2
    )

    # Create first patch set
    mat = get_default_test_solid_material(material_type="st_venant_kirchhoff")
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set_1 = add_geomdl_nurbs_to_mesh(mesh, element_type, surf_obj_1, material=mat)

    mesh.add(patch_set_1)

    mesh.rotate(Rotation([0, 0, 1], np.pi / 3))

    # Create the second surface object
    surf_obj_2 = create_nurbs_hollow_cylinder_segment_2d(
        0.65, 1.46, np.pi / 3, n_ele_u=3, n_ele_v=2
    )

    patch_set_2 = add_geomdl_nurbs_to_mesh(mesh, element_type, surf_obj_2, material=mat)

    mesh.add(patch_set_2)

    mesh.couple_nodes(reuse_matching_nodes=True)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_sphere_surface(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creating of the base patch of the surface of a sphere."""

    # Create mesh
    mesh = Mesh()

    # Create the base of a sphere
    surf_obj = create_nurbs_sphere_surface(1, n_ele_u=3, n_ele_v=2)

    # Create first patch set
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        surf_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_string_types(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creating of a NURBS with strings for the element and material
    definition."""

    # Create mesh
    mesh = Mesh()

    # Create the base of a sphere
    surf_obj = create_nurbs_flat_plate_2d(1, 3, n_ele_u=3, n_ele_v=2)

    # Create first patch set
    element_type = get_default_test_solid_element("nurbs_2d")
    patch_set = add_geomdl_nurbs_to_mesh(
        mesh,
        element_type,
        surf_obj,
        material=get_default_test_solid_material(material_type="st_venant_kirchhoff"),
    )

    mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_hemisphere_surface(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of the surface of a hemisphere."""

    # Create mesh
    mesh = Mesh()

    # Create the base of a sphere
    surfs = create_nurbs_hemisphere_surface(2.5, n_ele_uv=2)

    # Create first patch set
    mat = get_default_test_solid_material(material_type="st_venant_kirchhoff")
    element_type = get_default_test_solid_element("nurbs_2d")

    # Add the patch sets of every surface section of the hemisphere to the input file
    for surf in surfs:
        patch_set = add_geomdl_nurbs_to_mesh(
            mesh,
            element_type,
            surf,
            material=mat,
        )

        mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_torus_surface(
    get_default_test_solid_element,
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the creation of a torus."""

    # Create mesh
    mesh = Mesh()

    # Create the surface of a torus
    surfs = create_nurbs_torus_surface(1, 0.5, n_ele_u=2, n_ele_v=3)

    # Define element description
    mat = get_default_test_solid_material(material_type="st_venant_kirchhoff")
    element_type = get_default_test_solid_element("nurbs_2d")

    # Add the patch sets of every surface section of the torus to the input file
    for surf in surfs:
        patch_set = add_geomdl_nurbs_to_mesh(
            mesh,
            element_type,
            surf,
            material=mat,
        )

        mesh.add(patch_set)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_nurbs_empty_knot_spans(
    get_default_test_solid_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that NURBS patches with empty knot spans are handled correctly."""

    # Create the pipe geometry with splinepy
    disk = splinepy.helpme.create.disk(
        outer_radius=2.3, inner_radius=1.7, angle=360, n_knot_spans=1
    )
    pipe = disk.create.extruded(extrusion_vector=[0.0, 0.0, 2.0])
    pipe.elevate_degrees([0, 2])
    pipe.uniform_refine(2, 2)

    # Create mesh
    mesh = Mesh()
    mat = get_default_test_solid_material(material_type="st_venant_kirchhoff")
    element_type = get_four_c_solid("nurbs_3d", n_nodes=27)
    patch_set = add_splinepy_nurbs_to_mesh(mesh, element_type, pipe, material=mat)
    mesh.add(patch_set)
    mesh.couple_nodes(reuse_matching_nodes=True)

    # Compare with the reference file
    assert_results_close(get_corresponding_reference_file_path(), mesh)
