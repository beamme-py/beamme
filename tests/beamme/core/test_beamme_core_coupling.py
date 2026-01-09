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
"""This script is used to unittest the functionality of the couplings."""

import numpy as np
import pytest

from beamme.core.conf import bme
from beamme.core.coupling import Coupling, coupling_factory
from beamme.core.geometry_set import GeometrySet
from beamme.core.node import Node, NodeCosserat
from beamme.core.rotation import Rotation


def test_beamme_core_coupling_factory():
    """Test that the coupling factory can be used with the desired inputs."""

    nodes = [Node([0, 0, 0]) for i in range(4)]
    node_set = GeometrySet(nodes)

    # Single coupling for all nodes, list of nodes given
    couplings = coupling_factory(nodes, bme.bc.point_coupling, {})
    assert len(couplings) == 1
    coupling_points = couplings[0].geometry_set.get_points()
    assert len(coupling_points) == 4
    for i in range(4):
        assert coupling_points[i] is nodes[i]

    # Single coupling for all nodes, geometry set given
    couplings = coupling_factory(node_set, bme.bc.point_coupling, {})
    assert len(couplings) == 1
    assert couplings[0].geometry_set is node_set

    # Pairwise couplings, list of nodes given
    couplings = coupling_factory(nodes, bme.bc.point_coupling_penalty, {})
    assert len(couplings) == 3
    for i_coupling in range(3):
        coupling_nodes = couplings[i_coupling].geometry_set.get_points()
        assert len(coupling_nodes) == 2
        assert coupling_nodes[0] is nodes[0]
        assert coupling_nodes[1] is nodes[i_coupling + 1]

    # Pairwise couplings, geometry set given
    couplings = coupling_factory(node_set, bme.bc.point_coupling_penalty, {})
    assert len(couplings) == 3
    reference_geometry_set_nodes = node_set.get_points()
    for i_coupling in range(3):
        coupling_nodes = couplings[i_coupling].geometry_set.get_points()
        assert len(coupling_nodes) == 2
        assert coupling_nodes[0] is reference_geometry_set_nodes[0]
        assert coupling_nodes[1] is reference_geometry_set_nodes[i_coupling + 1]


@pytest.mark.parametrize("within_tolerance", (True, False))
@pytest.mark.parametrize("check_overlapping_nodes", (True, False))
def test_beamme_core_coupling_check_overlapping_coupling_nodes(
    within_tolerance, check_overlapping_nodes
):
    """Per default, we check that coupling nodes are at the same physical
    position.

    This check can be deactivated with the keyword
    check_overlapping_nodes when creating a Coupling.
    """

    # Create the nodes
    factor = 0.5 if within_tolerance else 2.0
    ref_position = [0.0, 1.5, 3.0]
    node_1 = NodeCosserat(ref_position, Rotation())
    node_2 = NodeCosserat(
        ref_position + factor * bme.eps_pos * np.array([1.0, 1.0, 1.0]), Rotation()
    )

    # Couple two nodes whose positions may be within or outside the positional
    # tolerance. When check_overlapping_nodes is enabled and the nodes are
    # outside the tolerance, creating the coupling is expected to raise an error.
    args = [[node_1, node_2], bme.bc.point_coupling, "coupling_type_string"]
    if check_overlapping_nodes and not within_tolerance:
        with pytest.raises(ValueError):
            Coupling(*args)
    else:
        Coupling(*args, check_overlapping_nodes=False)
