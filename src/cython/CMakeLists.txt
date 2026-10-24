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

enable_language(C)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(Cython 3.0 MODULE REQUIRED)
include(UseCython)

cython_transpile(
  "${CMAKE_CURRENT_SOURCE_DIR}/find_close_points_cython.pyx"
  LANGUAGE C
  OUTPUT_VARIABLE find_close_points_cython_c
)
python_add_library(find_close_points_cython MODULE "${find_close_points_cython_c}" WITH_SOABI)
target_link_libraries(find_close_points_cython PRIVATE Python::NumPy)
install(
  TARGETS find_close_points_cython
  DESTINATION beamme/geometric_search
  COMPONENT beamme
)
