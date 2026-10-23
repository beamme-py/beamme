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

cmake_minimum_required(VERSION 3.24)
enable_language(CXX)

# Compiler options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Kokkos is linked into the extension statically so the wheel does not depend
# on a separately installed Kokkos shared library. Static libraries linked into
# a Python extension need position-independent code on Unix platforms.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dependencies as static libraries" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable the Kokkos Serial backend" FORCE)
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests" FORCE)
set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Kokkos examples" FORCE)
set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "Disable Kokkos benchmarks" FORCE)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
  URL_HASH SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Kokkos)

# Get ArborX after Kokkos so ArborX uses the statically built Kokkos targets.
FetchContent_Declare(
  arborx
  GIT_REPOSITORY https://github.com/kokkos/ArborX.git
  GIT_TAG v2.1
)
FetchContent_MakeAvailable(arborx)

# Build the python extension module
set(lib_name "arborx_lib")
pybind11_add_module(${lib_name}
  geometric_search.cpp
  find_close_points.cpp
)
target_compile_options(${lib_name} PRIVATE -Wall -Wextra -fvisibility=hidden)
target_link_libraries(${lib_name} PRIVATE ArborX::ArborX)

# Install into the Python package. scikit-build-core collects installed CMake
# artifacts into wheels and editable installs.
install(
  TARGETS ${lib_name}
  DESTINATION beamme/geometric_search
  COMPONENT beamme
)
