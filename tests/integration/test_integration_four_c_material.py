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
"""Integration tests for materials in 4C."""

from beamme.core.mesh import Mesh
from beamme.four_c.input_file import InputFile


def test_integration_four_c_sub_materials(
    get_default_test_solid_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Check if sub-materials are handled correctly."""

    # Add a nested material to the mesh and check the result.
    mesh = Mesh()
    material = get_default_test_solid_material(material_type="solid_nested")
    mesh.add(material)
    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # Add the material again and check that the result is the same.
    mesh.add(material)
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_four_c_sub_materials_material_numbering(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that materials can be added as structured data (dictionaries) to an
    input file (as is done when importing input files) and that the numbering
    with other added materials does not lead to materials with double IDs."""

    input_file = InputFile()
    input_file.add(
        {
            "MATERIALS": [
                {
                    "MAT": 1,
                    "MAT_ViscoElastHyper": {
                        "NUMMAT": 4,
                        "MATIDS": [10, 11, 12, 13],
                        "DENS": 1.3e-06,
                    },
                },
                {
                    "MAT": 10,
                    "ELAST_CoupNeoHooke": {
                        "YOUNG": 0.16,
                        "NUE": 0.45,
                    },
                },
                {
                    "MAT": 11,
                    "VISCO_GenMax": {
                        "TAU": 0.1,
                        "BETA": 0.4,
                        "SOLVE": "OST",
                    },
                },
                {
                    "MAT": 12,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0024,
                        "K2": 0.14,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 100,
                        "INIT": 3,
                    },
                },
                {
                    "MAT": 13,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0054,
                        "K2": 1.24,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 100,
                        "INIT": 3,
                        "FIBER_ID": 2,
                    },
                },
                {
                    "MAT": 100,
                    "ELAST_StructuralTensor": {
                        "STRATEGY": "Standard",
                    },
                },
                {
                    "MAT": 2,
                    "MAT_ElastHyper": {
                        "NUMMAT": 3,
                        "MATIDS": [20, 21, 22],
                        "DENS": 1.3e-06,
                    },
                },
                {
                    "MAT": 20,
                    "ELAST_CoupNeoHooke": {
                        "YOUNG": 1.23,
                        "NUE": 0.45,
                    },
                },
                {
                    "MAT": 21,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0004,
                        "K2": 12,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 200,
                        "INIT": 3,
                    },
                },
                {
                    "MAT": 22,
                    "ELAST_CoupAnisoExpo": {
                        "K1": 0.0502,
                        "K2": 10,
                        "GAMMA": 0,
                        "K1COMP": 0,
                        "K2COMP": 1,
                        "STR_TENS_ID": 200,
                        "INIT": 3,
                        "FIBER_ID": 2,
                    },
                },
                {
                    "MAT": 200,
                    "ELAST_StructuralTensor": {
                        "STRATEGY": "Standard",
                    },
                },
            ]
        }
    )

    mesh = Mesh()
    mesh.add(get_default_test_beam_material(material_type="reissner"))

    input_file.add(mesh)

    assert_results_close(get_corresponding_reference_file_path(), input_file)
