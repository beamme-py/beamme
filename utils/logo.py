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
"""Create the BeamMe logo."""

from pathlib import Path

import numpy as np
import pyvista as pv
import splinepy
from vistools.pyvista.camera import apply_camera_settings
from vistools.pyvista.polyline_cross_section import polyline_cross_section
from vistools.vtk.merge_polylines import merge_polylines

from beamme.core.conf import bme
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rLine2Line2
from beamme.four_c.material import MaterialReissner
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_splinepy import (
    create_beam_mesh_from_splinepy,
    get_curve_function_and_jacobian_for_integration,
)
from beamme.utils.environment import cubitpy_is_available, is_testing

if cubitpy_is_available():
    from cubitpy.conf import cupy
    from cubitpy.cubitpy import CubitPy
    from cubitpy.geometry_creation_functions import create_spline_interpolation_curve

# Splinepy data for characters.
# The "curve" font data is taken from the font Relief-SingleLine (https://github.com/isdat-type/Relief-SingleLine?tab=OFL-1.1-1-ov-file)
# The "2D" font data is taken from Source Sans Pro (https://github.com/adobe-fonts/source-sans)
CHARACTER_SPLINEPY = {
    "B": [
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [0.2, -0.3],
                [6.633333333333334, -0.3],
                [13.066666666666666, -0.3],
                [19.5, -0.3],
                [19.5, -0.3],
                [24.6, -0.3],
                [28.8, -0.8],
                [32.1, -1.9000000000000001],
                [32.1, -1.9000000000000001],
                [35.4, -2.9000000000000004],
                [38.0, -4.3],
                [39.9, -6.1000000000000005],
                [39.9, -6.1000000000000005],
                [41.8, -7.9],
                [43.199999999999996, -9.9],
                [43.9, -12.100000000000001],
                [43.9, -12.100000000000001],
                [44.699999999999996, -14.3],
                [45.0, -16.6],
                [45.0, -18.900000000000002],
                [45.0, -18.900000000000002],
                [45.0, -22.3],
                [44.4, -25.1],
                [43.2, -27.5],
                [43.2, -27.5],
                [42.0, -29.9],
                [40.400000000000006, -31.9],
                [38.300000000000004, -33.4],
                [38.300000000000004, -33.4],
                [36.300000000000004, -35.0],
                [33.900000000000006, -36.1],
                [31.300000000000004, -36.8],
                [31.300000000000004, -36.8],
                [28.700000000000003, -37.5],
                [25.900000000000006, -37.9],
                [22.900000000000006, -37.9],
                [22.900000000000006, -37.9],
                [22.10000000000001, -37.900000000000006],
                [21.300000000000008, -37.9],
                [20.500000000000007, -37.9],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                ]
            ],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [0.2, -79.4],
                [6.666666666666666, -79.4],
                [13.133333333333333, -79.4],
                [19.599999999999998, -79.4],
                [19.599999999999998, -79.4],
                [24.799999999999997, -79.4],
                [29.199999999999996, -78.9],
                [32.699999999999996, -77.80000000000001],
                [32.699999999999996, -77.80000000000001],
                [36.3, -76.70000000000002],
                [39.099999999999994, -75.30000000000001],
                [41.3, -73.4],
                [41.3, -73.4],
                [43.5, -71.60000000000001],
                [45.0, -69.4],
                [46.0, -66.80000000000001],
                [46.0, -66.80000000000001],
                [47.0, -64.20000000000002],
                [47.4, -61.500000000000014],
                [47.4, -58.500000000000014],
                [47.4, -58.500000000000014],
                [47.4, -52.30000000000001],
                [45.3, -47.30000000000001],
                [41.1, -43.500000000000014],
                [41.1, -43.500000000000014],
                [36.9, -39.70000000000002],
                [30.0, -37.90000000000001],
                [20.5, -37.90000000000001],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                ]
            ],
        ),
        splinepy.bspline.BSpline(
            degrees=[1],
            control_points=[[0.2, -79.4], [0.2, -37.9]],
            knot_vectors=[[0.0, 0.0, 1.0, 1.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[1],
            control_points=[[0.2, -37.9], [0.2, -0.3]],
            knot_vectors=[[0.0, 0.0, 1.0, 1.0]],
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[20.7, -37.9], [0.2, -37.9]]
        ),
    ],
    "e": [
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [61.4, -44.0],
                [61.6, -40.2],
                [62.199999999999996, -36.6],
                [63.0, -33.4],
                [63.0, -33.4],
                [64.2, -29.2],
                [65.8, -25.7],
                [67.9, -22.799999999999997],
                [67.9, -22.799999999999997],
                [70.00000000000001, -19.899999999999995],
                [72.60000000000001, -17.699999999999996],
                [75.60000000000001, -16.199999999999996],
                [75.60000000000001, -16.199999999999996],
                [78.60000000000001, -14.699999999999996],
                [82.00000000000001, -13.899999999999995],
                [85.7, -13.899999999999995],
                [85.7, -13.899999999999995],
                [89.39999999999999, -13.899999999999995],
                [93.60000000000001, -14.799999999999995],
                [96.4, -16.699999999999996],
                [96.4, -16.699999999999996],
                [99.2, -18.599999999999994],
                [101.5, -20.999999999999996],
                [103.10000000000001, -23.799999999999997],
                [103.10000000000001, -23.799999999999997],
                [104.70000000000002, -26.599999999999998],
                [105.9, -29.9],
                [106.50000000000001, -33.5],
                [106.50000000000001, -33.5],
                [107.10000000000001, -37.1],
                [107.50000000000001, -40.5],
                [107.50000000000001, -43.9],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                ]
            ],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [106.7, -67.4],
                [106.0, -69.10000000000001],
                [105.0, -70.8],
                [103.8, -72.4],
                [103.8, -72.4],
                [102.6, -74.00000000000001],
                [101.2, -75.4],
                [99.5, -76.60000000000001],
                [99.5, -76.60000000000001],
                [97.8, -77.80000000000001],
                [95.9, -78.80000000000001],
                [93.7, -79.50000000000001],
                [93.7, -79.50000000000001],
                [91.5, -80.2],
                [90.60000000000001, -80.60000000000001],
                [86.2, -80.60000000000001],
                [86.2, -80.60000000000001],
                [81.8, -80.60000000000001],
                [78.1, -79.7],
                [75.0, -78.10000000000001],
                [75.0, -78.10000000000001],
                [71.89999999999999, -76.5],
                [69.3, -74.2],
                [67.3, -71.2],
                [67.3, -71.2],
                [65.3, -68.3],
                [63.8, -64.8],
                [62.8, -60.8],
                [62.8, -60.8],
                [61.9, -56.8],
                [61.4, -52.4],
                [61.4, -47.6],
                [61.4, -47.6],
                [61.4, -46.4],
                [61.4, -45.2],
                [61.4, -44.0],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                ]
            ],
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[107.6, -44.0], [61.4, -44.0]]
        ),
    ],
    "a": [
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[164.3, -65.03], [164.3, -43.67]]
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [164.3, -43.68],
                [161.86, -43.68000000000001],
                [159.42000000000002, -43.68],
                [156.98000000000002, -43.68],
                [156.98000000000002, -43.68],
                [144.9, -43.68],
                [136.16000000000003, -45.3],
                [130.76000000000002, -48.54],
                [130.76000000000002, -48.54],
                [125.36000000000001, -51.78],
                [122.66000000000003, -56.84],
                [122.66000000000003, -63.72],
                [122.66000000000003, -63.72],
                [122.66000000000003, -65.88],
                [123.00000000000003, -67.98],
                [123.68000000000002, -70.02],
                [123.68000000000002, -70.02],
                [124.36000000000003, -72.06],
                [125.38000000000002, -73.86],
                [126.74000000000002, -75.42],
                [126.74000000000002, -75.42],
                [128.10000000000002, -76.98],
                [129.82000000000002, -78.24],
                [131.90000000000003, -79.2],
                [131.90000000000003, -79.2],
                [133.98000000000005, -80.16000000000001],
                [136.46000000000004, -80.64],
                [139.34000000000003, -80.64],
                [139.34000000000003, -80.64],
                [144.54000000000002, -80.64],
                [149.16000000000003, -79.14],
                [153.20000000000005, -76.14],
                [153.20000000000005, -76.14],
                [157.24000000000007, -73.14],
                [160.94000000000005, -69.44],
                [164.30000000000004, -65.04],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                    8.0,
                    8.0,
                    9.0,
                    9.0,
                    9.0,
                    9.0,
                ]
            ],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [164.3, -65.03],
                [164.3, -67.27],
                [164.35000000000002, -69.77],
                [164.42000000000002, -72.53],
                [164.42000000000002, -72.53],
                [164.50000000000003, -75.69],
                [164.62, -78.39],
                [164.78000000000003, -80.63],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [164.3, -43.68],
                [164.3, -40.88],
                [164.3, -38.08],
                [164.3, -35.28],
                [164.3, -35.28],
                [164.3, -32.4],
                [163.98000000000002, -29.66],
                [163.34, -27.060000000000002],
                [163.34, -27.060000000000002],
                [162.7, -24.460000000000004],
                [161.66, -22.200000000000003],
                [160.22, -20.28],
                [160.22, -20.28],
                [158.78, -18.36],
                [156.88, -16.84],
                [154.52, -15.720000000000002],
                [154.52, -15.720000000000002],
                [152.16000000000003, -14.600000000000005],
                [149.22, -14.040000000000003],
                [145.70000000000002, -14.040000000000003],
                [145.70000000000002, -14.040000000000003],
                [140.58, -14.040000000000003],
                [136.44000000000003, -15.200000000000003],
                [133.28000000000003, -17.520000000000003],
                [133.28000000000003, -17.520000000000003],
                [130.12000000000003, -19.840000000000003],
                [127.54000000000003, -23.28],
                [125.54000000000003, -27.840000000000003],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                ]
            ],
        ),
    ],
    "m": [
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [264.79999999999995, -80.69999999999999],
                [264.79999999999995, -64.1],
                [264.79999999999995, -47.5],
                [264.79999999999995, -30.9],
                [264.79999999999995, -30.9],
                [264.79999999999995, -25.599999999999994],
                [263.69999999999993, -21.4],
                [261.29999999999995, -18.499999999999996],
                [261.29999999999995, -18.499999999999996],
                [258.9, -15.599999999999996],
                [255.39999999999998, -14.099999999999996],
                [250.79999999999998, -14.099999999999996],
                [250.79999999999998, -14.099999999999996],
                [248.2, -14.099999999999996],
                [245.7, -14.499999999999996],
                [243.29999999999998, -15.499999999999996],
                [243.29999999999998, -15.499999999999996],
                [241.0, -16.499999999999996],
                [238.7, -17.699999999999996],
                [236.6, -19.299999999999997],
                [236.6, -19.299999999999997],
                [234.4, -20.9],
                [232.4, -22.7],
                [230.5, -24.7],
                [230.5, -24.7],
                [228.6, -26.799999999999997],
                [226.70000000000002, -28.799999999999997],
                [224.9, -30.9],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    7.0,
                    7.0,
                ]
            ],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [224.9, -30.9],
                [224.9, -25.599999999999998],
                [223.70000000000002, -21.5],
                [221.4, -18.5],
                [221.4, -18.5],
                [219.0, -15.5],
                [215.5, -14.1],
                [210.9, -14.1],
                [210.9, -14.1],
                [206.3, -14.1],
                [205.8, -14.6],
                [203.4, -15.5],
                [203.4, -15.5],
                [201.0, -16.5],
                [198.8, -17.7],
                [196.70000000000002, -19.3],
                [196.70000000000002, -19.3],
                [194.60000000000002, -20.900000000000002],
                [192.60000000000002, -22.7],
                [190.60000000000002, -24.700000000000003],
                [190.60000000000002, -24.700000000000003],
                [188.70000000000002, -26.700000000000003],
                [186.8, -28.800000000000004],
                [185.00000000000003, -30.900000000000002],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    6.0,
                    6.0,
                ]
            ],
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[185.0, -30.9], [185.0, -14.0]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[224.9, -80.7], [224.9, -30.9]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[185.0, -80.7], [185.0, -30.9]]
        ),
    ],
    "M_2d": [
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [329.6, -55.9],
                [327.90000000000003, -50.8],
                [326.0, -45.5],
                [324.3, -40.5],
                [324.3, -40.5],
                [319.6666666666667, -27.733333333333334],
                [315.03333333333336, -14.966666666666669],
                [310.40000000000003, -2.200000000000003],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [348.4, -2.2],
                [343.8666666666667, -14.966666666666669],
                [339.33333333333326, -27.733333333333334],
                [334.79999999999995, -40.5],
                [334.79999999999995, -40.5],
                [333.09999999999997, -45.5],
                [331.29999999999995, -50.8],
                [329.59999999999997, -55.9],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [352.6, -19.8],
                [351.90000000000003, -27.1],
                [351.3, -37.5],
                [351.3, -44.7],
                [351.3, -44.7],
                [351.30000000000007, -56.66666666666667],
                [351.3, -68.63333333333333],
                [351.3, -80.6],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [307.3, -80.7],
                [307.30000000000007, -68.73333333333333],
                [307.3, -56.766666666666666],
                [307.3, -44.800000000000004],
                [307.3, -44.800000000000004],
                [307.3, -37.6],
                [306.7, -27.100000000000005],
                [306.0, -19.900000000000006],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[1],
            control_points=[
                [333.4, -73.3],
                [346.3, -38.0],
                [346.3, -38.0],
                [352.6, -19.8],
            ],
            knot_vectors=[[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[1],
            control_points=[
                [306.0, -19.8],
                [312.2, -38.0],
                [312.2, -38.0],
                [325.2, -73.3],
            ],
            knot_vectors=[[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]],
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[310.5, -2.2], [294.7, -2.2]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[364.1, -2.2], [348.4, -2.2]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[364.1, -80.7], [364.1, -2.2]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[351.4, -80.7], [364.1, -80.7]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[325.2, -73.3], [333.4, -73.3]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[294.7, -80.7], [307.3, -80.7]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[294.7, -2.2], [294.7, -80.7]]
        ),
    ],
    "e_2d": [
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [392.41, -45.57],
                [393.61, -36.120000000000005],
                [399.65000000000003, -30.990000000000002],
                [406.51000000000005, -30.990000000000002],
                [406.51000000000005, -30.990000000000002],
                [414.57000000000005, -30.990000000000002],
                [418.6, -36.510000000000005],
                [418.6, -45.57],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [423.59, -67.1],
                [419.27, -69.78999999999999],
                [414.95, -71.36999999999999],
                [409.91999999999996, -71.36999999999999],
                [409.91999999999996, -71.36999999999999],
                [400.41999999999996, -71.36999999999999],
                [393.69999999999993, -65.41999999999999],
                [392.54999999999995, -54.959999999999994],
            ],
            knot_vectors=[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        ),
        splinepy.bspline.BSpline(
            degrees=[3],
            control_points=[
                [429.94, -54.97],
                [430.25, -53.53],
                [430.54, -50.89],
                [430.54, -48.25],
                [430.54, -48.25],
                [430.54, -31.79],
                [422.15000000000003, -20.28],
                [406.22, -20.28],
                [406.22, -20.28],
                [392.36, -20.28],
                [379.07000000000005, -32.08],
                [379.07000000000005, -51.22],
                [379.07000000000005, -51.22],
                [379.07000000000005, -70.36],
                [391.83000000000004, -82.07],
                [408.09000000000003, -82.07],
                [408.09000000000003, -82.07],
                [415.58000000000004, -82.07],
                [422.67, -79.47999999999999],
                [428.29, -75.69],
            ],
            knot_vectors=[
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ]
            ],
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[418.6, -45.57], [392.41, -45.57]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[392.55, -54.97], [429.94, -54.97]]
        ),
        splinepy.bezier.Bezier(
            degrees=[1], control_points=[[428.3, -75.69], [423.59, -67.1]]
        ),
    ],
}

# The approximated "NURBS" control points for character a
CHARACTER_A_CONTROL_POINTS = [
    np.array(
        [
            [164.35324483, -43.58243023],
            [142.77495122, -42.60237254],
            [119.98822797, -52.25170688],
            [123.55368612, -80.44125075],
            [149.82412744, -82.44491517],
            [164.36604489, -65.10479724],
        ]
    ),
    np.array(
        [
            [164.29371849, -43.47329983],
            [165.92980009, -16.55819675],
            [132.69848732, -10.0654419],
            [125.60390606, -27.90025592],
        ]
    ),
    np.array([[164.3, -65.03], [164.78000000000003, -80.63]]),
]


def create_curve(curve, **kwargs):
    """Create the beam mesh for a given splinepy curve."""
    curve_tmp = curve.copy()
    mesh = Mesh()

    # The Jacobian along the given curves might not be continuous, which causes the
    # BeamMe curve integration to fail. We scale the knot span here, such
    # that the Jacobian is continuous.
    _, jacobian, _, curve_end = get_curve_function_and_jacobian_for_integration(
        curve_tmp, tol=1
    )
    n_seg = int(np.round(curve_end))
    if isinstance(curve_tmp, splinepy.BSpline):
        n_knot = len(np.array(curve_tmp.knot_vectors)[0])
        n_double = int(n_knot / (n_seg + 1))
        knot_scaled = [0.0] * n_double + [1.0] * n_double
        factor_old = 1.0
        for i in range(1, n_seg):
            last_jacobian = np.linalg.norm(jacobian(i - bme.eps_knot_vector))
            my_jacobian = np.linalg.norm(jacobian(i + bme.eps_knot_vector))
            factor = factor_old * my_jacobian / last_jacobian
            knot_scaled.extend([knot_scaled[-1] + factor] * n_double)
            factor_old = factor
        curve_tmp.knot_vectors = [knot_scaled]

    _, length = create_beam_mesh_from_splinepy(
        mesh,
        Beam3rLine2Line2,
        MaterialReissner(),
        curve_tmp,
        tol=10,
        output_length=True,
        **kwargs,
    )
    return mesh, length


def create_mesh_curves(curves, l_el) -> Mesh:
    """Create a BeamMe mesh for all given curves and return a single mesh
    containing all curves."""
    mesh = Mesh()
    for i in range(len(curves)):
        curve_mesh, _ = create_curve(curves[i], l_el=l_el)
        mesh.add(curve_mesh)
    return mesh


def add_letter_B(plotter, plot_data, lighting=True):
    """Add the letter B to the plotter."""

    color_letter = "#1E3A8A"
    color_nodes = "#FFD43B"  # "#F43F5E"
    curves = CHARACTER_SPLINEPY["B"]
    mesh_coarse, mesh_fine = [
        create_mesh_curves(curves, l_el)
        for l_el in [plot_data["l_el_coarse"], plot_data["l_el_fine"]]
    ]

    # Tube part
    vtk_curve = pv.UnstructuredGrid(
        merge_polylines(mesh_fine.get_vtk_representation()[0].grid)
    )
    surface = vtk_curve.extract_surface()
    surface = surface.cell_data_to_point_data()
    tube = surface.tube(
        radius=plot_data["beam_radius"], n_sides=plot_data["resolution"]
    )
    plotter.add_mesh(tube, color=color_letter, lighting=lighting)

    # Nodes
    coarse_nodes = pv.UnstructuredGrid(
        mesh_coarse.get_vtk_representation()[0].grid
    ).cell_data_to_point_data()
    nodes_glyph = coarse_nodes.glyph(
        geom=plot_data["sphere"],
        factor=1.5 * plot_data["beam_radius"],
        orient=False,
        scale=False,
    )
    plotter.add_mesh(nodes_glyph, color=color_nodes, lighting=lighting)


def add_letter_e(plotter, plot_data):
    """Add the letter e to the plotter."""

    color = "#60A5FA"
    mesh = Mesh()
    cross_section = 3.0 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    curves = CHARACTER_SPLINEPY["e"]
    for i in range(len(curves)):
        curve_mesh, length = create_curve(curves[i], l_el=plot_data["l_el_fine"])
        if length < 70:
            angle = np.pi
        else:
            angle = -2.0 * np.pi
        n_nodes = len(curve_mesh.nodes)
        for i_node, node in enumerate(curve_mesh.nodes):
            rot_new = node.rotation * Rotation(
                [1, 0, 0], angle / (n_nodes - 1) * i_node
            )
            node.rotation = rot_new
        mesh.add(curve_mesh)
    curve = pv.UnstructuredGrid(
        merge_polylines(mesh.get_vtk_representation()[0].grid, tol=1e-4)
    )
    curve_with_cross_section = polyline_cross_section(
        curve, cross_section, separate_surfaces=True
    )
    curve_edges = curve_with_cross_section.extract_feature_edges()
    plotter.add_mesh(curve_with_cross_section, smooth_shading=True, color=color)
    plotter.add_mesh(curve_edges, color="black", line_width=2)


def add_letter_a(plotter, plot_data):
    """Add the letter a to the plotter."""

    color_letter = "black"
    color_control_polygon = "blue"
    color_control_points = "red"
    curves = CHARACTER_SPLINEPY["a"]
    curve_mesh = create_mesh_curves(curves, plot_data["l_el_fine"])
    vtk_curve = pv.UnstructuredGrid(
        merge_polylines(curve_mesh.get_vtk_representation()[0].grid)
    )
    surface = vtk_curve.extract_surface()
    surface = surface.cell_data_to_point_data()
    plotter.add_mesh(
        surface.tube(radius=0.75),
        color=color_letter,
        lighting=False,
    )

    # Draw the control point grid
    for control_points in CHARACTER_A_CONTROL_POINTS:
        if control_points.shape[1] == 2:
            control_points = np.hstack(
                [control_points, np.zeros((control_points.shape[0], 1))]
            )

        # Create line connectivity (polyline)
        n_pts = control_points.shape[0]
        lines = np.hstack([[n_pts], np.arange(n_pts)])

        # Create polyline object
        polyline = pv.PolyData()
        polyline.points = control_points
        polyline.lines = lines

        plotter.add_mesh(
            polyline.tube(radius=0.25),
            color=color_control_polygon,
            label="Control Polygon",
            lighting=False,
        )
        nodes_glyph = polyline.glyph(
            geom=plot_data["cube"],
            factor=1.0 * plot_data["beam_radius"],
            orient=False,
            scale=False,
        )
        plotter.add_mesh(nodes_glyph, color=color_control_points)


def add_letter_m(plotter, plot_data):
    """Add the letter m to the plotter."""

    color_letter = "#FBBF24"
    mesh = Mesh()
    factor = 4.0
    t = 0.5
    cross_section = factor * np.array(
        [
            [-1, -1],
            [1, -1],
            [1, -1 + t],
            [0.5 * t, -1 + t],
            [0.5 * t, 1 - t],
            [1, 1 - t],
            [1, 1],
            [-1, 1],
            [-1, 1 - t],
            [-0.5 * t, 1 - t],
            [-0.5 * t, -1 + t],
            [-1, -1 + t],
            [-1, -1],
        ]
    )
    curves = CHARACTER_SPLINEPY["m"]
    for i in range(len(curves)):
        curve_mesh, _ = create_curve(curves[i], l_el=plot_data["l_el_fine"])
        vtk_curve = merge_polylines(
            pv.UnstructuredGrid(curve_mesh.get_vtk_representation()[0].grid)
        )
        cross_section_mesh = pv.UnstructuredGrid(
            polyline_cross_section(vtk_curve, cross_section, closed=False)
        )
        cross_section_mesh.save("test.vtu")
        plotter.add_mesh(cross_section_mesh, color=color_letter)
        mesh.add(curve_mesh)

        if i == 2:
            top_position = mesh.nodes[-1].coordinates
            top_rotation_matrix = (
                mesh.nodes[-1].rotation
                * Rotation([1, 0, 0], 0.5 * np.pi)
                * Rotation([0, 1, 0], 0.5 * np.pi)
            ).get_rotation_matrix()

    # Top face
    polygon_points = np.array(
        [
            np.dot(top_rotation_matrix, [cross_section[i, 0], cross_section[i, 1], 0.0])
            for i in range(len(cross_section))
        ]
    )
    polygon_points += top_position
    faces = np.hstack(
        [
            [6, 0, 1, 2, 3, 10, 11],  # bottom
            [4, 3, 4, 9, 10],  # middle
            [6, 4, 5, 6, 7, 8, 9],  # top
        ]
    )
    mesh = pv.PolyData(polygon_points, faces)
    plotter.add_mesh(mesh, color=color_letter)


def create_2d_mesh_cubit(plot_data):
    """Create the 2D mesh for the letters M and e."""

    if not cupy.is_coreform():
        raise ValueError("This script requires Cubit Coreform")

    cubit = CubitPy()

    for key in ["M_2d", "e_2d"]:
        curves = CHARACTER_SPLINEPY[key]
        cubit_curves = []
        for curve in curves:
            # We use the BeamMe curve integration function to get the points along the curve.
            curve_mesh = create_mesh_curves([curve], plot_data["l_el_mid"])
            vtk_curve = pv.UnstructuredGrid(
                merge_polylines(curve_mesh.get_vtk_representation()[0].grid)
            )
            curve_points = vtk_curve.points
            cubit_curves.append(create_spline_interpolation_curve(cubit, curve_points))
        cubit.cmd(
            f"create surface curve {' '.join([str(curve.id()) for curve in cubit_curves])}"
        )
        surface = cubit.surface(cubit.get_last_id(cupy.geometry.surface))

        if key == "M_2d":
            cubit.cmd(f"surface {surface.id()} scheme pave")
            cubit.cmd(f"surface {surface.id()} size auto factor 4")
            surface.mesh()
            cubit.cmd(
                f"sweep surface {surface.id()} perpendicular distance {plot_data['extrude_distance']}"
            )
            volume = cubit.volume(cubit.get_last_id(cupy.geometry.volume))
            volume.mesh()
            cubit.add_element_type(volume, cupy.element_type.hex8)
        else:
            cubit.cmd(
                f"sweep surface {surface.id()} perpendicular distance {plot_data['extrude_distance']}"
            )
            volume = cubit.volume(cubit.get_last_id(cupy.geometry.volume))
            cubit.cmd(f"volume {volume.id()} scheme tetmesh")
            cubit.cmd(f"volume {volume.id()} size auto factor 7")
            volume.mesh()
            cubit.add_element_type(volume, cupy.element_type.tet4)

    # Set the material.
    cubit.fourc_input["MATERIALS"] = [{"MAT": 1, "dummy": {}}]

    cubit.dump(plot_data["input_file_name"])


def get_letter_2d_grid(plot_data):
    """Return the PyVista grids for the 2D letters."""

    _, solid_mesh = import_four_c_model(
        plot_data["input_file_name"], convert_input_to_mesh=True
    )
    solid_mesh.translate([0, 0, 0.5 * plot_data["extrude_distance"]])
    solid_grid = pv.UnstructuredGrid(
        solid_mesh.get_vtk_representation()[1].grid
    ).clean()
    solid_grid = solid_grid.connectivity()
    return [
        solid_grid.threshold(
            scalars="RegionId", value=(region_id - 1e-1, region_id + 1e-1)
        )
        for region_id in [
            1,
            0,
        ]  # Inverse order here, so M is the first and e is the second.
    ]


def add_letter_2d_full(plotter, plot_data):
    """Add both 2D letters to the plotter."""

    color_letters = {0: "#10B981", 1: "#F472B6"}
    grids = get_letter_2d_grid(plot_data)
    translate_vector = np.array([294.7, -80.7, 0.0])
    translate_letter = {0: [-8, 0, 0], 1: [-13, 0, 0]}
    for i_grid, grid in enumerate(grids):
        grid.translate(-translate_vector, inplace=True)
        grid.scale(1.05, inplace=True)
        grid.translate(translate_vector, inplace=True)
        grid.translate(translate_letter[i_grid], inplace=True)
        plotter.add_mesh(grid, show_edges=True, color=color_letters[i_grid])


def add_letter_2d_small(plotter, plot_data, lighting=False):
    """Add just the M of the 2D letters to the plotter."""

    color = "#10B981"
    grid_M, _ = get_letter_2d_grid(plot_data)
    grid_M.translate([-235, 0, 0], inplace=True)
    translate_vector = 0.5 * (
        np.min(grid_M.points, axis=0) + np.max(grid_M.points, axis=0)
    )
    grid_M.translate(-translate_vector, inplace=True)
    grid_M.scale(1.075, inplace=True)
    y_mid = 0.5 * (-79.4 + (-0.3))  # Measured from letter B
    translate_vector[1] = y_mid
    grid_M.translate(translate_vector, inplace=True)
    plotter.add_mesh(grid_M, show_edges=False, color=color, lighting=lighting)


def create_beamme_logo_full(plotter, plot_data):
    """Create the full BeamMe logo."""

    # Create the letters
    add_letter_B(plotter, plot_data)
    add_letter_e(plotter, plot_data)
    add_letter_a(plotter, plot_data)
    add_letter_m(plotter, plot_data)
    add_letter_2d_full(plotter, plot_data)

    view = {
        "2D": {
            "window_size": [3300, 700],
            "camera_position": [
                210.1369786266167,
                -39.635466910711216,
                99.99994357785772,
            ],
            "camera_focal_point": [210.24320680213364, -39.635466910711216, 0.0],
            "camera_view_up": [0.0, 1.0, 0.0],
            "parallel_projection": 1,
            "parallel_scale": 45.64392169861635,
            "view_angle": 30.0,
        }
    }
    apply_camera_settings(plotter, view["2D"])


def create_beamme_logo_small(plotter, plot_data, square=True):
    """Create the small BeamMe logo."""

    # Create the letters
    add_letter_B(plotter, plot_data, lighting=False)
    add_letter_2d_small(plotter, plot_data, lighting=False)

    view = {
        False: {
            "window_size": [1280, 640],
            "camera_position": [64.32596930324353, -39.72129035167604, 100.0],
            "camera_focal_point": [64.32596930324353, -39.72129035167604, 0.0],
            "camera_view_up": [0.0, 1.0, 0.0],
            "parallel_projection": 1,
            "parallel_scale": 44.88496648212754,
            "view_angle": 30.0,
        },
        True: {
            "window_size": [1080, 1080],
            "camera_position": [64.32596930324353, -39.72129035167604, 100.0],
            "camera_focal_point": [64.32596930324353, -39.72129035167604, 0.0],
            "camera_view_up": [0.0, 1.0, 0.0],
            "parallel_projection": 1,
            "parallel_scale": 70.58560796483074,
            "view_angle": 30.0,
        },
    }
    apply_camera_settings(plotter, view[square])


def create_beamme_logo(base_dir, create_cubit=True):
    """Create the BeamMe logo images."""

    # General parameters
    plot_data = {
        "l_el_coarse": 20.0,
        "l_el_fine": 2.0,
        "l_el_mid": 5.0,
        "resolution": 30,
        "beam_radius": 3.0,
        "input_file_name": base_dir / "beamme_logo_2d_characters.4C.yaml",
        "extrude_distance": 8.0,
    }
    plot_data["sphere"] = pv.Sphere(
        radius=1.0,
        theta_resolution=plot_data["resolution"],
        phi_resolution=plot_data["resolution"],
    )
    plot_data["cube"] = pv.Cube()

    # If required create the mesh with cubit
    if create_cubit:
        create_2d_mesh_cubit(plot_data)

    # Large logo
    plotter = pv.Plotter(off_screen=True)
    create_beamme_logo_full(plotter, plot_data)
    if not is_testing():
        plotter.screenshot(
            base_dir / "beamme_logo_wide.png", transparent_background=True
        )

    # Small logo
    for square in [False, True]:
        plotter = pv.Plotter(off_screen=True)
        create_beamme_logo_small(plotter, plot_data, square=square)
        if not is_testing():
            plotter.screenshot(
                base_dir / f"beamme_logo_icon_{'square' if square else 'wide'}.png",
                transparent_background=True,
            )


if __name__ == "__main__":
    """Create the logo."""
    create_beamme_logo(Path(__file__).parent, create_cubit=True)
