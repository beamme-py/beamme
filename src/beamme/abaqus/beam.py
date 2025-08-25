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
"""This file provides functions to create Abaqus beam element classes."""

from beamme.core.element_beam import Beam2 as _Beam2
from beamme.core.element_beam import Beam3 as _Beam3


def generate_abaqus_beam(beam_type: str):
    """Return a class representing a beam in Abaqus. This class can be used in
    the standard mesh generation functions.

    Args:
        beam_type: Abaqus identifier for this beam element. For more details,
            have a look at the Abaqus manual on "Choosing a beam element".

    Returns:
        A class representing the Abaqus beam element. The class inherits from
        the BeamX class, depending on the number of nodes.
    """

    if not beam_type[0].lower() == "b":
        raise TypeError("Could not identify the given Abaqus beam element")

    n_dim = int(beam_type[1])
    element_type = int(beam_type[2])

    if not n_dim == 3:
        raise ValueError("Currently only 3D beams in Abaqus are supported")
    if element_type == 1:
        base_class = _Beam2
    elif element_type == 2:
        base_class = _Beam3
    elif element_type == 3:
        base_class = _Beam2
    else:
        raise ValueError(f"Got unexpected element_type {element_type}")

    # Create the Abaqus beam class.
    return type(
        "BeamAbaqus" + beam_type,
        (base_class,),
        {
            "beam_type": beam_type,
            "n_dim": n_dim,
        },
    )
