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
"""Test the performance of the geometric space time module."""

import pytest

from beamme.core.element_beam import generate_beam_class
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.space_time.beam_to_space_time import beam_to_space_time


@pytest.fixture(scope="module")
def mesh_in_space(evaluate_execution_time):
    """Provide mesh in space direction."""
    mesh = Mesh()
    beam_type = generate_beam_class(3)

    evaluate_execution_time(
        "BeamMe: Space-Time: Create mesh in space",
        create_beam_mesh_line,
        kwargs={
            "mesh": mesh,
            "beam_class": beam_type,
            "material": MaterialBeamBase(),
            "start_point": [0, 0, 0],
            "end_point": [1, 0, 0],
            "n_el": 100,
        },
        expected_time=0.01,
    )

    return mesh


@pytest.mark.performance
def test_performance_space_time_create_mesh_in_space(mesh_in_space):
    """Test the performance of the mesh creation in space.

    The test is run in the fixture, so we don't need to do anything here.
    """
    pass


@pytest.mark.performance
def test_performance_space_time_create_mesh_in_time(
    mesh_in_space, evaluate_execution_time
):
    """Test the performance of the mesh creation in time."""
    evaluate_execution_time(
        "BeamMe: Space-Time: Create mesh in time",
        beam_to_space_time,
        kwargs={
            "mesh_space_or_generator": mesh_in_space,
            "time_duration": 6.9,
            "number_of_elements_in_time": 1000,
            "time_start": 1.69,
        },
        expected_time=3.0,
    )
