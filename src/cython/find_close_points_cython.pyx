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
# cython: language_level=3
"""
This module contains functions that calculate points that are close together.
It uses cython to gain performance.
"""


import numpy as np
from libc.math cimport sqrt

cimport cython
cimport numpy as np

@cython.boundscheck(False) # Deactivate checking bounds of arrays.
@cython.wraparound(False)  # Deactivate negative indexing.
def find_close_points(np.ndarray[double, ndim=2] coords, double eps):
    """
    Finds coordinates that are within a tolerance of each other.

    Args
    ----
    coords: numpy array
        Array with the coordinates of the nodes.
    eps: float
        Tolerance to look for neighbors.

    Return
    ----
    has_partner: numpy array
        An array with integers, marking the set a node is part of. -1 means the
        node does not have a partner.
    partner: int
        Number of clusters found.
    """

    # Define types of variables for this function.
    cdef np.int64_t n_nodes, partner, this_is_partner, i, j, k, n_dim
    cdef double distance_square, eps_square

    # Square of the tolerance
    eps_square = eps**2

    # Number of nodes and dimension of coordinates.
    n_nodes = np.shape(coords)[0]
    n_dim = np.shape(coords)[1]

    # This vector is -1 if a node does not belong to a pair, otherwise it is
    # the number of the pair.
    cdef np.ndarray[np.int64_t, ndim=1] has_partner = (
        np.zeros(n_nodes, dtype=np.int64) - 1)

    # Loop over nodes.
    partner = 0
    for i in range(n_nodes):
        this_is_partner = 0
        if has_partner[i] == -1:
            for j in range(i + 1, n_nodes):
                # Calculate the squared distance between the two nodes.
                distance_square = 0.0
                for k in range(n_dim):
                    distance_square += (coords[i, k] - coords[j, k])**2
                # Check if the distance is smaller than the threshold, and add
                # to has_partner list.
                if distance_square < eps_square:
                    this_is_partner = 1
                    if not has_partner[j] == -1:
                        raise RuntimeError('The case where a node connects two'
                            ' other nodes that are more than eps apart is not'
                            ' yet implemented. Check if the value for eps'
                            ' makes physical sense!')
                    has_partner[j] = partner
            # If this one has a partner set this node too.
            if this_is_partner == 1:
                has_partner[i] = partner
                partner += 1

    return has_partner, partner
