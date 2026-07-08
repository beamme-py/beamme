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
"""This module implements containers to manage boundary conditions and geometry sets in
one object."""

# This comment avoids docformatter's issue: https://github.com/PyCQA/docformatter/issues/350


class ContainerBase(dict):
    """A base class for containers."""

    def append(self, key, item):
        """Append item to this container and check if the item is already in the list
        corresponding to key."""
        type_ok = False
        for item_type in self.item_types:
            if isinstance(item, item_type):
                type_ok = True
                break
        if not type_ok:
            raise TypeError(
                f"You tried to add an item of type {type(item)}, but only types derived "
                + f"from {self.item_types} can be added"
            )
        if key not in self.keys():
            self[key] = []
        else:
            if item in self[key]:
                raise ValueError("The item is already in this container!")
        self[key].append(item)

    def extend(self, container):
        """Add all items of another container to this container."""
        if not isinstance(container, self.__class__):
            raise TypeError(
                f"Only containers of type {self.__class__} can be merged here, you tried "
                + f"add {type(container)}"
            )
        for key, items in container.items():
            for item in items:
                self.append(key, item)
