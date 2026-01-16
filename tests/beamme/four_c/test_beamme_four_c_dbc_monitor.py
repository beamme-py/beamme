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
"""Unit tests for the dbc monitor."""

import numpy as np

from beamme.four_c.dbc_monitor import linear_time_transformation, read_dbc_monitor_file


def test_beamme_four_c_dbc_monitor_read_dbc_monitor_file(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test that a dbc monitor file can be read correctly."""

    nodes, time, force, moment = read_dbc_monitor_file(
        get_corresponding_reference_file_path(extension="yaml")
    )
    assert_results_close(nodes, [2, 4, 5, 9, 10])
    assert_results_close(time, [0.0, 0.1, 0.2])
    assert_results_close(force, [[0, 0, 0], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    assert_results_close(moment, [[0, 0, 0], [0.4, 0.5, 0.6], [1.0, 1.1, 1.2]])


def test_beamme_four_c_dbc_monitor_linear_time_transformation_scaling():
    """Test the scaling of the interval for the function.

    Starts with a function within the interval [0,1] and transforms
    them.
    """

    # starting time array
    time = np.array([0, 0.5, 0.75, 1.0])

    # corresponding values 3 values per time step
    force = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # with the result vector
    force_result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # base case no scaling only
    time_trans, force_trans = linear_time_transformation(
        time, force, [0, 1], flip=False
    )

    # first result is simply the attached
    time_result = np.array([0, 0.5, 0.75, 1.0])

    # check solution
    assert time_trans.tolist() == time_result.tolist()
    assert force_trans.tolist() == force_result.tolist()

    # transform to interval [0, 2]
    time_trans, force_trans = linear_time_transformation(
        time, force, [0, 2], flip=False
    )

    # time values should double
    assert time_trans.tolist() == (2 * time_result).tolist()
    assert force_trans.tolist() == force_result.tolist()

    # new result
    force_result = np.array(
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [10, 11, 12]]
    )

    # shift to the interval [1 ,2] and add valid start end point
    time_trans, force_trans = linear_time_transformation(
        time, force, [1, 2, 5], flip=False, valid_start_and_end_point=True
    )
    assert time_trans.tolist() == np.array([0, 1.0, 1.5, 1.75, 2.0, 5.0]).tolist()
    assert force_trans.tolist() == force_result.tolist()


def test_beamme_four_c_dbc_monitor_linear_time_transformation_flip():
    """Test the flip flag option of linear_time_transformation to mirror the
    function."""

    # base case no scaling no end points should be attached
    # starting time array
    time = np.array([0, 0.5, 0.75, 1.0])

    # corresponding values:  3 values per time step
    force = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # first result is simply the attached point at the end
    time_result = np.array([0, 0.25, 0.5, 1.0])

    # with the value vector:
    force_result = np.array([[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]])

    # base case no scaling only end points should be attached
    time_trans, force_trans = linear_time_transformation(time, force, [0, 1], flip=True)

    # check solution
    assert time_result.tolist() == time_trans.tolist()
    assert force_trans.tolist() == force_result.tolist()

    # new force result
    force_result = np.array([[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]])

    time_result = np.array([0, 0.25, 0.5, 1.0]) + 1

    # test now an shift to the interval [1 ,2]
    time_trans, force_trans = linear_time_transformation(time, force, [1, 2], flip=True)
    assert time_result.tolist() == time_trans.tolist()
    assert force_trans.tolist() == force_result.tolist()

    # same trick as above but with 2
    time_result = np.array([0, 2.0, 2.25, 2.5, 3.0, 5.0])
    # new force result
    force_result = np.array(
        [[10, 11, 12], [10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3], [1, 2, 3]]
    )

    # test offset and scaling and add valid start and end point
    time_trans, force_trans = linear_time_transformation(
        time, force, [2, 3, 5], flip=True, valid_start_and_end_point=True
    )
    assert time_result.tolist() == time_trans.tolist()
    assert force_trans.tolist() == force_result.tolist()
