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
"""Unit tests for the generic beam mesh creation functions."""

import re

import numpy as np
import pytest

from beamme.core.element_beam import Beam3
from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.mesh_creation_functions.beam_arc import create_beam_mesh_arc_segment_2d
from beamme.mesh_creation_functions.beam_generic import create_beam_mesh_generic
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)


def test_beamme_mesh_creation_functions_beam_generic_start_end_node_error(
    get_default_test_beam_material,
):
    """Check that an error is raised if wrong start and end nodes are given to a mesh
    creation function."""
    # Create mesh object.
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="base")
    mesh.add(mat)

    # Try to create a line with a starting node that is not in the mesh.
    node = NodeCosserat([0, 0, 0], Rotation())
    args = [mesh, Beam3, mat, [0, 0, 0], [1, 0, 0]]
    kwargs = {"start_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)
    node.coordinates = [1, 0, 0]
    kwargs = {"end_node": node}
    with pytest.raises(ValueError):
        create_beam_mesh_line(*args, **kwargs)


def test_beamme_mesh_creation_functions_beam_generic_argument_checks():
    """Test that wrong input values leads to failure."""
    dummy_arg = "dummy"

    # Check error messages for input parameters
    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error since we dont allow `n_el` and `l_el`
        # to be set at the same time.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            n_el=1,
            l_el=1.5,
        )
    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error because node_positions_of_elements can not be used with l_el.

        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            l_el=1,
            node_positions_of_elements=[0.0, 0.5, 1.0],
        )

    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error because node_positions_of_elements can not be used with n_el.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            n_el=1,
            node_positions_of_elements=[0.0, 0.5, 1.0],
        )

    with pytest.raises(
        ValueError, match='The parameter "l_el" requires "interval_length" to be set.'
    ):
        mesh = Mesh()
        # This should raise an error because we set `l_el` but don't provide
        # `interval_length`.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            l_el=2.0,
        )

    with pytest.raises(
        ValueError,
        match="First entry of node_positions_of_elements must be 0, got -1.0",
    ):
        mesh = Mesh()
        # This should raise an error because the interval [0,1] is violated.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            node_positions_of_elements=[-1.0, 0.0, 1.0],
        )

    with pytest.raises(
        ValueError, match="Last entry of node_positions_of_elements must be 1, got 2.0"
    ):
        mesh = Mesh()
        # This should raise an error because the interval [0,1] is violated.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            node_positions_of_elements=[0.0, 1.0, 2.0],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The given node_positions_of_elements must be in ascending order. Got [0.0, 0.2, 0.1, 1.0]"
        ),
    ):
        mesh = Mesh()
        # This should raise an error because the interval is not ordered.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            node_positions_of_elements=[0.0, 0.2, 0.1, 1.0],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            'The arguments "close_beam" and "end_node" are mutually exclusive'
        ),
    ):
        mesh = Mesh()
        # This should raise an error because the interval is not ordered.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            n_el=1,
            close_beam=True,
            end_node=dummy_arg,
        )


def test_beamme_mesh_creation_functions_beam_generic_arc_length_argument_checks(
    get_default_test_beam_material,
):
    """Test that wrong input values leads to failure."""
    dummy_arg = "dummy"

    # Check error messages for input parameters
    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error since we dont allow `n_el` and `l_el`
        # to be set at the same time.
        create_beam_mesh_line(
            mesh,
            Beam3,
            get_default_test_beam_material(material_type="reissner"),
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 6.0],
            n_el=1,
            l_el=1.5,
        )
    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error because node_positions_of_elements can not be used with l_el.

        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            l_el=1,
            node_positions_of_elements=[0.0, 0.5, 1.0],
        )

    with pytest.raises(
        ValueError,
        match='The arguments "n_el", "l_el" and "node_positions_of_elements" are mutually exclusive',
    ):
        mesh = Mesh()
        # This should raise an error because node_positions_of_elements can not be used with n_el.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            n_el=1,
            node_positions_of_elements=[0.0, 0.5, 1.0],
        )

    with pytest.raises(
        ValueError, match='The parameter "l_el" requires "interval_length" to be set.'
    ):
        mesh = Mesh()
        # This should raise an error because we set `l_el` but don't provide
        # `interval_length`.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            l_el=2.0,
        )

    with pytest.raises(
        ValueError,
        match="First entry of node_positions_of_elements must be 0, got -1.0",
    ):
        mesh = Mesh()
        # This should raise an error because the interval [0,1] is violated.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            node_positions_of_elements=[-1.0, 0.0, 1.0],
        )

    with pytest.raises(
        ValueError, match="Last entry of node_positions_of_elements must be 1, got 2.0"
    ):
        mesh = Mesh()
        # This should raise an error because the interval [0,1] is violated.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            node_positions_of_elements=[0.0, 1.0, 2.0],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The given node_positions_of_elements must be in ascending order. Got [0.0, 0.2, 0.1, 1.0]"
        ),
    ):
        mesh = Mesh()
        # This should raise an error because the interval is not ordered.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            node_positions_of_elements=[0.0, 0.2, 0.1, 1.0],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            'The arguments "close_beam" and "end_node" are mutually exclusive'
        ),
    ):
        mesh = Mesh()
        # This should raise an error because the interval is not ordered.
        create_beam_mesh_generic(
            mesh,
            beam_class=dummy_arg,
            material=dummy_arg,
            beam_function=dummy_arg,
            interval=[0, 1],
            n_el=1,
            close_beam=True,
            end_node=dummy_arg,
        )


@pytest.mark.parametrize(
    "basic_creation_function,arc_length_ref",
    (
        ("arc", None),
        ("line", None),
        ("parametric_curve_arc-length", None),
        (
            "parametric_curve_parametric",
            np.array(
                [
                    0.0,
                    0.1432291807294664,
                    0.3678198188308245,
                    1.9466864902724,
                    6.5938166349472915,
                ]
            ),
        ),
        (
            "parametric_curve_parametric_consistent_middle_nodes",
            np.array(
                [
                    0.0,
                    0.18390990941541224,
                    0.3678198188308245,
                    3.480818226889058,
                    6.5938166349472915,
                ]
            ),
        ),
    ),
)
def test_beamme_mesh_creation_functions_beam_generic_arc_length(
    get_parametric_function,
    basic_creation_function,
    arc_length_ref,
    get_default_test_beam_material,
    assert_results_close,
):
    """Test that the arc length can be stored in the nodes when creating a filament."""
    node_positions_of_elements = [0, 0.25, 1]
    mat = get_default_test_beam_material(material_type="reissner")
    offset = 3.0

    if basic_creation_function == "line":
        length = 2.5
        start_pos = [0, 0, 0]
        end_pos = [length, 0, 0]
        start_rot = Rotation()
        end_rot = Rotation()

        def create_beam(mesh, **kwargs):
            """Wrapper for the common arguments in the call to create the line."""
            create_beam_mesh_line(
                mesh,
                Beam3,
                mat,
                start_pos,
                end_pos,
                node_positions_of_elements=node_positions_of_elements,
                **kwargs,
            )

    elif basic_creation_function == "arc":
        length = np.pi * 0.5
        start_pos = [0, 1, 0]
        end_pos = [-1, 0, 0]
        start_rot = Rotation([0, 0, 1], np.pi)
        end_rot = Rotation([0, 0, 1], 1.5 * np.pi)

        def create_beam(mesh, **kwargs):
            """Wrapper for the common arguments in the call to create the arc."""
            create_beam_mesh_arc_segment_2d(
                mesh,
                Beam3,
                mat,
                [0, 0, 0],
                1.0,
                np.pi * 0.5,
                np.pi,
                node_positions_of_elements=node_positions_of_elements,
                **kwargs,
            )

    elif basic_creation_function.startswith("parametric_curve_"):
        arc_length_integration_method = "_".join(basic_creation_function.split("_")[2:])

        length = 6.5938166349472915
        start_pos = [2, 0, 0]
        end_pos = [-2, 0, 2]
        start_rot = Rotation.from_rotation_vector(
            [1.432490763274102, 1.0473349985879405, 1.432490763274102]
        )
        end_rot = Rotation.from_rotation_vector(
            [-1.405448691974458, 1.014988339931174, -1.4449555589051493]
        )

        R = 2.0
        tz = 4.0
        n = 0.5

        def create_beam(mesh, **kwargs):
            """Wrapper for the common arguments in the call to create the parametric
            curve beam mesh (helix)."""
            create_beam_mesh_parametric_curve(
                mesh,
                Beam3,
                mat,
                get_parametric_function(
                    "helix", R, tz, transformation_factor=2.0, number_of_turns=n
                ),
                [0.0, 2.0 * np.pi * n],
                node_positions_of_elements=node_positions_of_elements,
                arc_length_integrator_kwargs={
                    "method": arc_length_integration_method,
                    "scipy_integrate": False,
                },
                **kwargs,
            )

    else:
        raise ValueError(f"Unknown basic_creation_function '{basic_creation_function}'")

    if arc_length_ref is None:
        arc_length_ref = np.array([0, 0.125, 0.25, 0.625, 1.0]) * length
    arc_length_offset_ref = arc_length_ref + offset

    def get_start_and_end_node(*, arc_length_start=None, arc_length_end=None):
        """Return the explicitly created start and end node."""
        start_node = NodeCosserat(start_pos, start_rot, arc_length=arc_length_start)
        end_node = NodeCosserat(end_pos, end_rot, arc_length=arc_length_end)
        return (start_node, end_node)

    def check_arc_length(mesh, arc_length_ref):
        """Compare the arc lengths of the nodes in mesh with reference values."""
        arc_length_from_mesh = np.array([node.arc_length for node in mesh.nodes])
        assert_results_close(
            {"arc_length": arc_length_from_mesh}, {"arc_length": arc_length_ref}
        )

    # Standard arc length calculation
    mesh = Mesh()
    create_beam(mesh, set_nodal_arc_length=True)
    check_arc_length(mesh, arc_length_ref)

    # Standard arc length calculation with offset
    mesh = Mesh()
    create_beam(mesh, set_nodal_arc_length=True, nodal_arc_length_offset=offset)
    check_arc_length(mesh, arc_length_offset_ref)

    # Arc length calculation based on start node
    mesh = Mesh()
    start_node, _ = get_start_and_end_node(arc_length_start=offset)
    mesh.add(start_node)
    create_beam(mesh, set_nodal_arc_length=True, start_node=start_node)
    check_arc_length(mesh, arc_length_offset_ref)

    # Arc length calculation based on end node
    mesh = Mesh()
    _, end_node = get_start_and_end_node(arc_length_end=offset + length)
    mesh.add(end_node)
    create_beam(mesh, set_nodal_arc_length=True, end_node=end_node)
    # The nodes are in different order here, so we need to reorder the reference result
    check_arc_length(
        mesh, [arc_length_offset_ref[-1]] + arc_length_offset_ref.tolist()[:-1]
    )

    # Arc length calculation based on start and end node (if the values don't
    # match an error will be raised)
    mesh = Mesh()
    start_node, end_node = get_start_and_end_node(
        arc_length_start=offset, arc_length_end=offset + length
    )
    mesh.add(start_node, end_node)
    create_beam(
        mesh, set_nodal_arc_length=True, start_node=start_node, end_node=end_node
    )
    # The nodes are in different order here, so we need to reorder the reference result
    check_arc_length(
        mesh,
        [arc_length_offset_ref[0]]
        + [arc_length_offset_ref[-1]]
        + arc_length_offset_ref.tolist()[1:-1],
    )

    # Arc length calculation based on all possible parameters (if the values don't
    # match an error will be raised)
    mesh = Mesh()
    start_node, end_node = get_start_and_end_node(
        arc_length_start=offset, arc_length_end=offset + length
    )
    mesh.add(start_node, end_node)
    create_beam(
        mesh,
        set_nodal_arc_length=True,
        start_node=start_node,
        end_node=end_node,
        nodal_arc_length_offset=offset,
    )
    # The nodes are in different order here, so we need to reorder the reference result
    check_arc_length(
        mesh,
        [arc_length_offset_ref[0]]
        + [arc_length_offset_ref[-1]]
        + arc_length_offset_ref.tolist()[1:-1],
    )
