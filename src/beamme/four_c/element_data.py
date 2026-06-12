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
"""Define a data class for 4C element data."""

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import Any as _Any

import numpy as _np
from numpy.typing import NDArray as _NDArray

from beamme.utils.data_structures import (
    compare_nested_dicts_or_lists as _compare_nested_dicts_or_lists,
)


@_dataclass(eq=False)
class FourCElementData:
    """Class that contains the data for a 4C element block."""

    four_c_type: str
    four_c_cell: str
    element_technology: dict[str, _Any] = _field(default_factory=dict)

    def get_block_dict(
        self, i_block: int, i_material: int, additional_data: dict | None = None
    ) -> dict:
        """Return the dictionary containing the data for this element block.

        Args:
            i_block: The index of the element block.
            i_material: The index of the material. If -1, no material will be assigned
                to this element block.
            additional_data: Additional data to add to the element block dictionary.

        Returns: The dictionary to write this element block data to the input file for
            mesh based output.
        """
        return {
            "ID": i_block,
            self.four_c_type: {
                self.four_c_cell: {
                    **({"MAT": i_material + 1} if i_material != -1 else {}),
                    **self.element_technology,
                    **(additional_data or {}),
                }
            },
        }

    def get_yaml_dict(
        self, element_id, connectivity, element_material_id, additional_element_data
    ) -> dict:
        """Return the dictionary to write this element data to a yaml element
        definition in the input file."""

        return {
            "id": element_id + 1,
            "cell": {
                "type": self.four_c_cell,
                "connectivity": connectivity + 1,
            },
            "data": {
                "type": self.four_c_type,
                **(
                    {"MAT": element_material_id + 1}
                    if element_material_id != -1
                    else {}
                ),
                **self.element_technology,
                **(additional_element_data or {}),
            },
        }

    def __eq__(self, other) -> bool:
        """Check if two 4C element data objects are equal."""
        if not isinstance(other, FourCElementData):
            return False
        if self.four_c_cell != other.four_c_cell:
            return False
        if self.four_c_type != other.four_c_type:
            return False
        if not _compare_nested_dicts_or_lists(
            self.element_technology, other.element_technology
        ):
            return False
        return True


def four_c_element_data_from_yaml_dict(
    yaml_dict: dict,
) -> tuple[FourCElementData, int, _NDArray, int]:
    """Extract the 4C element data from a yaml element definition in the input
    file.

    Args:
        yaml_dict: The yaml element definition in the input file, will be modified in place.

    Returns:
        A tuple containing the 4C element data, the element ID, the connectivity, and the material ID.
    """

    element_id = yaml_dict["id"]
    connectivity = _np.array(yaml_dict["cell"]["connectivity"], dtype=int) - 1
    four_c_cell = yaml_dict["cell"]["type"]

    # Since we directly store `yaml_dict["data"]` as the element technology data,
    # the material ID and four_c_type have to be removed from the `yaml_dict["data"]`
    # dictionary, thus the `pop`.
    material_id = yaml_dict["data"].pop("MAT", 0) - 1
    four_c_type = yaml_dict["data"].pop("type")

    data = FourCElementData(
        four_c_type=four_c_type,
        four_c_cell=four_c_cell,
        element_technology=yaml_dict["data"],
    )

    return data, element_id, connectivity, material_id


def four_c_element_data_from_exo_dict(exo_dict: dict) -> tuple[FourCElementData, int]:
    """Extract the 4C element data from a exodus element definition in the
    input file.

    Args:
        exo_dict: The exodus element definition in the input file, will be modified in
            place.

    Returns:
        A tuple containing the 4C element data and the material ID.
    """

    # First, we have to remove the ID entry
    exo_dict.pop("ID")

    four_c_type = list(exo_dict.keys())[0]
    four_c_cell = list(exo_dict[four_c_type].keys())[0]
    element_technology = exo_dict[four_c_type][four_c_cell]
    material_id = element_technology.pop("MAT", 0) - 1

    data = FourCElementData(
        four_c_type=four_c_type,
        four_c_cell=four_c_cell,
        element_technology=element_technology,
    )

    return data, material_id
