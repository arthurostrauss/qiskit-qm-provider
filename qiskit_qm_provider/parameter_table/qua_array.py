# Copyright 2026 Arthur Strauss
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QUAArray: N-dimensional view over a 1D QUA array with slicing and variable lists.

Author: Arthur Strauss
Date: 2026-02-08
"""

from numbers import Number
from typing import Tuple, Union, List, Sequence, Literal, Optional, Any
import numpy as np

from qm.qua import declare, assign as qua_assign, fixed, for_
from qm.jobs.running_qm_job import RunningQmJob
from quam.utils.qua_types import QuaVariableInt, Scalar, ScalarInt

from .parameter import Parameter


class QUAArray(Parameter):
    """
    A generic N-dimensional view over one big 1D QUA array.
    Supports arbitrary shape and slicing, returning lists of variables or sub-array views.
    """

    def __init__(
        self,
        name: str,
        shape_or_value: Union[Tuple[int, ...], List[Any], np.ndarray],
        qua_type: Optional[Union[str, type]] = None,
        input_type=None,
        direction=None,
        units: str = "",
    ):
        """
        Args:
            name: Name of the parameter.
            shape_or_value: Either a tuple representing the shape (for empty init)
                            or a multi-dimensional array/list of values.
            qua_type: QUA type of the elements.
            input_type: Input type of the parameter.
            direction: Direction of the parameter (only for DGX Quantum).
            units: Units of the parameter.
        """
        self.shape: Tuple[int, ...]

        # Determine shape and initial list for Parameter
        if (
            isinstance(shape_or_value, (tuple, list))
            and all(isinstance(x, int) for x in shape_or_value)
            and not isinstance(shape_or_value, np.ndarray)
        ):
            # Treated as shape if it's a tuple/list of ints and not a numpy array
            # Ambiguity: [1, 2] could be shape (1, 2) or value [1, 2].
            # Convention: If passed as tuple, it's shape. If list, check contents.
            # Actually, safe bet: if it looks like shape, treat as shape ONLY if tuple.
            # If list of ints, treat as 1D value?
            # The prompt implies generalizing QUA2DArray. QUA2DArray distinction was explicit (n_rows vs value).
            # Let's assume tuple -> shape, list/ndarray -> value.
            if isinstance(shape_or_value, tuple):
                self.shape = shape_or_value
                length = int(np.prod(self.shape))
                init_list = [0] * length
            else:
                # List of values
                val_arr = np.array(shape_or_value)
                self.shape = val_arr.shape
                init_list = val_arr.flatten().tolist()

        elif isinstance(shape_or_value, np.ndarray):
            self.shape = shape_or_value.shape
            init_list = shape_or_value.flatten().tolist()
        else:
            raise TypeError(
                "shape_or_value must be a tuple (shape) or list/ndarray (value)"
            )

        # Calculate strides
        # Stride for dimension i is product of shape[i+1:]
        strides = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        self.strides = tuple(strides)

        super().__init__(
            name=name,
            value=init_list,
            qua_type=qua_type,
            input_type=input_type,
            direction=direction,
            units=units,
        )

    def _flat_index(self, *indices: ScalarInt) -> ScalarInt:
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        flat_idx = 0
        for i, (ind, stride, dim_size) in enumerate(
            zip(indices, self.strides, self.shape)
        ):
            # Runtime check for python integers
            if isinstance(ind, int):
                if not (0 <= ind < dim_size):
                    raise IndexError(
                        f"Index {ind} out of bounds for dimension {i} with size {dim_size}"
                    )
                if ind == 0:
                    continue

            term = ind
            if stride != 1:
                term = term * stride

            if isinstance(flat_idx, int) and flat_idx == 0:
                flat_idx = term
            else:
                flat_idx = flat_idx + term

        return flat_idx

    def __getitem__(self, key):
        """
        Access elements, sub-arrays, or slices.
        - arr[i, j, k] -> QUA variable (if full rank)
        - arr[i] -> Sub-array View (if partial rank)
        - arr[i, :] -> List of results (recursive expansion)
        """
        if self.var is None:
            raise RuntimeError(f"{self.name} not declared yet")

        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Handle slicing via expansion
        # We expand the first slice found and recurse
        for i, k in enumerate(key):
            if isinstance(k, slice):
                # Determine range from slice and shape
                start, stop, step = k.indices(self.shape[i])
                res = []
                # We must iterate in Python to return a list of variables/views
                for val in range(start, stop, step):
                    # Construct new key with integer at this position
                    new_key = key[:i] + (val,) + key[i + 1 :]
                    res.append(self[new_key])
                return res

        # If we are here, 'key' contains no slices, only indices (int or QUA)
        if len(key) == len(self.shape):
            return self.var[self._flat_index(*key)]
        elif len(key) < len(self.shape):
            return _QUAArrayView(self, key)
        else:
            raise IndexError(f"Too many indices: {len(key)} for shape {self.shape}")

    def assign(self, indices_or_val, val=None):
        """
        Flexible assign:
        - assign(val) -> delegates to Parameter.assign (fills whole array)
        - assign(indices..., val) -> assigns specific element
        """
        if val is None:
            # Case: assign(val) where val is a list/array for the whole parameter
            super().assign(indices_or_val)
        else:
            # Case: assign(i, j, ..., val) or assign((i, j), val)
            if isinstance(indices_or_val, tuple):
                indices = indices_or_val
            elif isinstance(indices_or_val, (int, QuaVariableInt, list)):
                # ambiguous if list can be an index? Assuming list is not a single index.
                # If arguments are packed: assign(i, j, val) -> handled by python args?
                # No, this method signature takes 2 args.
                # The user likely calls arr.assign((i, j), val) or uses the View.assign
                indices = (indices_or_val,)
            else:
                indices = (indices_or_val,)

            # We only support assigning to a single element via this method if full indices provided
            # Or if it's a View assign, we handle it there.
            # Here we assume indices fully specify an element.
            if len(indices) == len(self.shape):
                qua_assign(self[indices], val)
            else:
                # Partial indices -> get view and assign?
                # But value would need to be array-like.
                # Simpler to just support full indexing here or rely on views.
                raise ValueError(
                    "Use __getitem__ to get a View for partial assignment or provide full indices."
                )


class _QUAArrayView:
    """
    Proxy object representing a sub-array of a QUAArray.
    Allows further indexing or assignment.
    """

    def __init__(self, parent: QUAArray, indices: Tuple):
        self._parent = parent
        self._indices = indices

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Combine current indices with new key
        new_full_key = self._indices + key
        return self._parent[new_full_key]

    def assign(self, val):
        """
        Assign a value (scalar or array) to this view.
        """
        # Determine the shape of this view
        view_shape = self._parent.shape[len(self._indices) :]
        view_size = int(np.prod(view_shape))

        if view_size == 1:
            # It's a single element (shouldn't happen for View usually, but possible)
            qua_assign(self._parent[self._indices], val)
            return

        # If val is scalar, assign to all? or assume val matches shape?
        # Parameter.assign supports list/array/qua_array.

        # Implementation: iterate over the view's domain and assign
        # This requires generating indices for the view.
        # This is complex for arbitrary N-D in QUA (nested loops).
        # We can flatten the view logic.

        # Simple case: 1D view (row) -> assign list
        if len(view_shape) == 1:
            # Logic similar to QUA2DArray row assignment
            seq = val
            if isinstance(seq, np.ndarray):
                seq = seq.tolist()

            if isinstance(seq, list):
                if len(seq) != view_shape[0]:
                    raise ValueError(f"Length mismatch: {len(seq)} vs {view_shape[0]}")
                for i, item in enumerate(seq):
                    qua_assign(self[i], item)
                return

            if isinstance(seq, QuaArrayVariable):  # QUA array
                # Use a for_ loop
                # We need a counter. Use parent's ctr or new one?
                # Parameter has _ctr.
                ctr = self._parent._ctr
                if ctr is None:
                    ctr = declare(int)

                with for_(ctr, 0, ctr < view_shape[0], ctr + 1):
                    qua_assign(self[ctr], seq[ctr])
                return

        raise NotImplementedError(
            "Assignment to N-D views > 1D not fully implemented yet."
        )
