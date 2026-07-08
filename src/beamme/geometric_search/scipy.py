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
"""This file defines the interface to the Scipy spatial geometric search
functionality."""

from scipy.spatial import KDTree as _KDTree


def pairs_to_partner_list(pairs, n_points):
    """Convert the pairs to a partner list."""
    # Sort the pairs by the first column
    pairs = pairs[pairs[:, 0].argsort()]

    n_partners = 0
    partner_index_list = [-1 for i in range(n_points)]
    for pair in pairs:
        pair_partners = [partner_index_list[pair[i]] for i in range(2)]
        if pair_partners[0] == -1 and pair_partners[1] == -1:
            # We have a new partner index
            for i in range(2):
                partner_index_list[pair[i]] = n_partners
            n_partners += 1
        elif pair_partners[0] == -1:
            partner_index_list[pair[0]] = pair_partners[1]
        elif pair_partners[1] == -1:
            partner_index_list[pair[1]] = pair_partners[0]
        else:
            # Both points already have a partner index. Since we order
            # the pairs this one has to be equal to each other or we
            # have multiple clusters.
            if not pair_partners[0] == pair_partners[1]:
                raise ValueError(
                    "Two points are connected to different partner indices."
                )
    return partner_index_list, n_partners


def find_close_points_scipy(point_coordinates, tol):
    """Call the Scipy implementation of find close_points."""
    kd_tree = _KDTree(point_coordinates)
    pairs = kd_tree.query_pairs(r=tol, output_type="ndarray")
    return pairs_to_partner_list(pairs, len(point_coordinates))
