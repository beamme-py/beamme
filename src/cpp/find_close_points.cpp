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

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include "find_close_points.h"

namespace GeometricSearch
{
    /*
     * Define wrappers for the search data.
     * TODO: Template this also on the scalar type of the numpy array.
     */
    template <int n_dim>
    struct PointsPrimitives
    {
        const pybind11::detail::unchecked_reference<double, 2>* coordinates_;
    };

    template <int n_dim>
    struct PointsPredicates
    {
        const pybind11::detail::unchecked_reference<double, 2>* coordinates_;
        double tol_;
    };
}  // namespace GeometricSearch

/*
 * Define how ArborX accesses the point data provided by the find_close_points function.
 */
namespace ArborX
{
    template <int n_dim>
    struct AccessTraits<GeometricSearch::PointsPrimitives<n_dim>>
    {
        static KOKKOS_FUNCTION std::size_t size(
            const GeometricSearch::PointsPrimitives<n_dim>& points)
        {
            return points.coordinates_->shape(0);
        }
        static KOKKOS_FUNCTION auto get(
            const GeometricSearch::PointsPrimitives<n_dim>& points, std::size_t i)
        {
            Point<n_dim, double> point;
            for (int d = 0; d < n_dim; ++d)
            {
                point[d] = points.coordinates_->operator()(i, d);
            }
            return point;
        }
        using memory_space = Kokkos::HostSpace;
    };

    template <int n_dim>
    struct AccessTraits<GeometricSearch::PointsPredicates<n_dim>>
    {
        static KOKKOS_FUNCTION std::size_t size(
            const GeometricSearch::PointsPredicates<n_dim>& points)
        {
            return points.coordinates_->shape(0);
        }
        static KOKKOS_FUNCTION auto get(
            const GeometricSearch::PointsPredicates<n_dim>& points, std::size_t i)
        {
            // Create an ArborX sphere at the coordinates of the point here. The radius is the
            // tolerance for the closes point search.
            Point<n_dim, double> point;
            for (int d = 0; d < n_dim; ++d)
            {
                point[d] = points.coordinates_->operator()(i, d);
            }
            return attach(intersects(ArborX::Sphere<n_dim, double>{point, points.tol_}), (int)i);
        }
        using memory_space = Kokkos::HostSpace;
    };
}  // namespace ArborX


namespace GeometricSearch
{
    template <int n_dim>
    std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> find_close_points_template(
        const pybind11::array_t<double>& coordinates, double tol)
    {
        using memory_space = Kokkos::HostSpace;

        // Get the data from the numpy array and create the primitives and predicates.
        const auto& coordinates_unchecked = coordinates.unchecked<2>();
        const PointsPrimitives<n_dim> points_primitives{.coordinates_ = &coordinates_unchecked};
        const PointsPredicates<n_dim> points_predicates{
            .coordinates_ = &coordinates_unchecked, .tol_ = tol};

        // Build tree structure containing all points
        ArborX::BoundingVolumeHierarchy bounding_volume_hierarchy{
            Kokkos::DefaultHostExecutionSpace{},
            ArborX::Experimental::attach_indices(points_primitives)};

        // Perform the collision check
        auto ensure_unique_pairs =
            KOKKOS_LAMBDA(const auto predicate, const auto& value, const auto& out)->void
        {
            int const i = value.index;
            int const j = getData(predicate);
            if (i > j)
            {
                out(i);
            }
        };
        Kokkos::View<int*, Kokkos::HostSpace> indices_arborx("indices_arborx", 0);
        Kokkos::View<int*, Kokkos::HostSpace> offset_arborx("offset_arborx", 0);
        bounding_volume_hierarchy.query(Kokkos::DefaultExecutionSpace{}, points_predicates,
            ensure_unique_pairs, indices_arborx, offset_arborx);

        // Copy everything into numpy arrays
        auto copy_kokkos_array = [&](const Kokkos::View<int*, Kokkos::HostSpace>& vec)
        {
            auto np_array = pybind11::array_t<int>(vec.size());
            pybind11::buffer_info np_array_buffer = np_array.request();
            int* np_array_ptr = (int*)np_array_buffer.ptr;
            for (unsigned int i = 0; i < vec.size(); i++)
            {
                np_array_ptr[i] = vec[i];
            }
            return np_array;
        };

        // Return the search results
        return std::make_tuple(copy_kokkos_array(indices_arborx), copy_kokkos_array(offset_arborx));
    }


    std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> find_close_points_factory(
        const pybind11::array_t<double>& coordinates, double tol)
    {
        switch (coordinates.shape(1))
        {
            case 1:
                return find_close_points_template<1>(coordinates, tol);
            case 2:
                return find_close_points_template<2>(coordinates, tol);
            case 3:
                return find_close_points_template<3>(coordinates, tol);
            case 4:
                return find_close_points_template<4>(coordinates, tol);
            case 5:
                return find_close_points_template<5>(coordinates, tol);
            case 6:
                return find_close_points_template<6>(coordinates, tol);
            case 7:
                return find_close_points_template<7>(coordinates, tol);
            case 8:
                return find_close_points_template<8>(coordinates, tol);
            default:
                throw std::out_of_range("Got unexpected number of spatial dimensions");
        }
    }


    std::tuple<pybind11::array_t<int>, int> find_close_points(
        const pybind11::array_t<double>& coordinates, const double tol)
    {
        const auto& [indices, offsets] = find_close_points_factory(coordinates, tol);
        const auto& indices_unchecked = indices.unchecked<1>();
        const auto& offsets_unchecked = offsets.unchecked<1>();

        // Create empty data array.
        const auto n_points = coordinates.shape(0);
        auto clusters = pybind11::array_t<int>(n_points);
        pybind11::buffer_info clusters_buffer = clusters.request();
        int* clusters_ptr = (int*)clusters_buffer.ptr;
        for (unsigned int i_point = 0; i_point < n_points; i_point++)
        {
            clusters_ptr[i_point] = -1;
        }

        // Loop over ArborX data
        int n_clusters = 0;
        for (unsigned int i_point = 0; i_point < n_points; i_point++)
        {
            const auto start = offsets_unchecked(i_point);
            const auto end = offsets_unchecked(i_point + 1);

            const bool has_cluster = 0 < (end - start);
            const auto my_cluster_id = clusters_ptr[i_point];

            if (has_cluster && my_cluster_id == -1)
            {
                // Loop over matching points and set them to the ID of this cluster.
                for (int j = start; j < end; j++)
                {
                    int j_point = indices_unchecked(j);
                    if (clusters_ptr[j_point] == -1)
                    {
                        clusters_ptr[j_point] = n_clusters;
                    }
                    else
                    {
                        throw std::out_of_range("No point can be part of two clusters");
                    }
                }
                clusters_ptr[i_point] = n_clusters;
                n_clusters += 1;
            }
            else if (has_cluster)
            {
                // The point is part of a cluster and the cluster ID is already set. Check
                // that all partners match the ID.
                for (int j = start; j < end; j++)
                {
                    int j_point = indices_unchecked(j);
                    if (clusters_ptr[j_point] != my_cluster_id)
                    {
                        throw std::out_of_range("No point can be part of two clusters");
                    }
                }
            }
        }

        return std::make_tuple(clusters, n_clusters);
    }
}  // namespace GeometricSearch
