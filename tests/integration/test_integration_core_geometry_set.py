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
"""This script is used to test general functionality of the core geometry set class with
end-to-end integration tests."""

import pytest

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet, GeometrySetNodes
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


@pytest.mark.parametrize("use_nodal_geometry_sets", [True, False])
def test_integration_core_geometry_set_replace_nodes_geometry_set(
    get_bc_data,
    get_default_test_beam_material,
    use_nodal_geometry_sets,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test case for coupling of nodes, and reusing the identical nodes."""
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
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="1"), mesh_couple
    )

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
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="2"), mesh_couple
    )

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
    assert_results_close(mesh_ref, mesh_couple)
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="3"), mesh_couple
    )
