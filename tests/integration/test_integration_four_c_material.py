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
"""Integration tests for materials in 4C."""

import pytest

from beamme.core.mesh import Mesh
from beamme.four_c.material import MaterialSolid


def test_integration_four_c_sub_materials(
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Check if sub-materials are handled correctly."""

    # Add a nested material to the mesh and check the result.
    mesh = Mesh()
    material = MaterialSolid(
        material_string="MAT_ElastHyper",
        data={
            "NUMMAT": 2,
            "MATIDS": [
                MaterialSolid(
                    material_string="ELAST_CoupSVK", data={"YOUNG": 1.0, "NUE": 0.0}
                ),
                MaterialSolid(
                    material_string="MAT_ElastHyper",
                    data={
                        "NUMMAT": 1,
                        "MATIDS": [
                            MaterialSolid(
                                material_string="ELAST_CoupSVK",
                                data={"YOUNG": 2.0, "NUE": 0.0},
                            )
                        ],
                        "DENS": 1.0,
                    },
                ),
            ],
            "DENS": 1.0,
        },
    )
    mesh.add(material)
    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # Add the material again and check that the result is the same.
    mesh.add(material)
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_four_c_sub_materials_error():
    """Check the error for incorrectly added sub-materials."""

    mesh = Mesh()
    material_sub = MaterialSolid(
        material_string="ELAST_CoupSVK", data={"YOUNG": 1.0, "NUE": 0.0}
    )
    mesh.add(material_sub)
    material = MaterialSolid(
        material_string="MAT_ElastHyper",
        data={
            "NUMMAT": 1,
            "MATIDS": [material_sub],
            "DENS": 1.0,
        },
    )
    with pytest.raises(
        ValueError,
        match="The material to be added has some "
        "materials that are already in the mesh and some that are not. "
        "This is not allowed.",
    ):
        mesh.add(material)
