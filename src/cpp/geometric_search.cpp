// The MIT License (MIT)
//
// Copyright (c) 2018-2026 BeamMe Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <pybind11/pybind11.h>
#include <Kokkos_Core.hpp>

#include "find_close_points.h"


namespace GeometricSearch
{
    /**
     * @brief Initialize Kokkos
     *
     */
    void kokkos_initialize() { Kokkos::initialize(); };

    /**
     * @brief Finalize Kokkos
     *
     */
    void kokkos_finalize() { Kokkos::finalize(); };
}  // namespace GeometricSearch


/**
 * @brief Construct the arborx_lib pybind11 module object
 *
 */
PYBIND11_MODULE(arborx_lib, py_module)
{
    py_module.doc() = "Python wrapper for ArborX";
    py_module.def("kokkos_initialize", &GeometricSearch::kokkos_initialize, "Initialize Kokkos");
    py_module.def("kokkos_finalize", &GeometricSearch::kokkos_finalize, "Finalize Kokkos");
    py_module.def("find_close_points", &GeometricSearch::find_close_points,
        "Find sets of points that are within the spatial radius tol of each other.\n"
        "\n"
        "Args\n"
        "----\n"
        "points: np.array\n"
        "    Two-dimensional array with point coordinates.\n"
        "tol: double\n"
        "    Tolerance for closest point search.\n"
        "\n"
        "Return\n"
        "----\n"
        "has_partner: numpy array\n"
        "    An array with integers, marking the set a node is part of. -1 means the\n"
        "    node does not have a partner.\n"
        "partner: int\n"
        "    Number of clusters found.\n");
}
