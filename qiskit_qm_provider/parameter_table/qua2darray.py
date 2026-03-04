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

"""QUA2DArray: 2D view over a 1D QUA array for row/column indexing.

Author: Arthur Strauss
Date: 2026-02-08
"""

from numbers import Number

from qm.jobs.running_qm_job import RunningQmJob
from qm.qua import declare, assign as qua_assign, fixed, for_
from quam.utils.qua_types import QuaVariableInt, Scalar, ScalarInt

from .parameter import Parameter
from typing import Tuple, Union, List, Sequence, Literal, Optional
import numpy as np
from qm.qua._expressions import QuaArrayVariable


class QUA2DArray(Parameter):
    """
    A 2D view over one big 1D QUA array.
    QUA array of length n_rows * n_cols and gives 2D indexing.
    """

    def __init__(
        self,
        name: str,
        n_rows_or_value: int | List[List[Number]] | np.ndarray,
        n_cols: Optional[int] = None,
        qua_type: Optional[Union[str, type]] = None,
        input_type=None,
        direction=None,
        units: str = "",
    ):
        """
        Class to create a 2D array of QUA variables.
        Args:
            name: Name of the parameter.
            n_rows_or_value: Number of rows or a 2D array of values.
            n_cols: Number of columns.
            qua_type: QUA type of the elements.
            input_type: Input type of the parameter.
            direction: Direction of the parameter (only for DGX Quantum).
            units: Units of the parameter.
        Example:
            >>> fake_data = QUA2DArray("fake_data", 50, 2, 8)
            >>> fake_data[0, 0]
            >>> fake_data[0]
            >>> fake_data.assign(0, [1, 0, 1, 0, 1, 0, 1, 0])
        """
        # prepare an "initial" 1D list of zeros so that Parameter can infer length/type
        if isinstance(n_rows_or_value, int):
            n_rows = n_rows_or_value
            if n_cols is None:
                raise ValueError(
                    "n_cols must be provided if n_rows_or_value is an integer"
                )
            if n_rows < 1 or n_cols < 1:
                raise ValueError("n_rows and n_cols must be strictly positive integers")
            length = n_rows * n_cols
            init_list = [0] * length
        elif isinstance(n_rows_or_value, np.ndarray):
            if n_rows_or_value.ndim != 2:
                raise ValueError(f"Value must be a 2D array")
            n_rows, n_cols = n_rows_or_value.shape
            init_list = n_rows_or_value.flatten().tolist()
        elif isinstance(n_rows_or_value, list) and all(
            isinstance(row, list) for row in n_rows_or_value
        ):
            n_rows = len(n_rows_or_value)
            n_cols = len(n_rows_or_value[0])
            if len(n_rows_or_value) != n_rows or any(
                len(row) != n_cols for row in n_rows_or_value
            ):
                raise ValueError(
                    f"Value must be a 2D list of shape ({n_rows}, {n_cols})"
                )
            init_list = [item for row in n_rows_or_value for item in row]
        else:
            raise TypeError("Value must be a 2D numpy array or a list of lists")

        # let Parameter.__init__ infer or take the qua_type you pass
        super().__init__(
            name=name,
            value=init_list,
            qua_type=qua_type,
            input_type=input_type,
            direction=direction,
            units=units,
        )

        self.n_rows = n_rows
        self.n_cols = n_cols

    def _flat_index(self, i: ScalarInt, j: ScalarInt) -> ScalarInt:
        if isinstance(i, int) and (i < 0 or i >= self.n_rows):
            raise IndexError(f"Row index {i} out of bounds for n_rows={self.n_rows}")
        if isinstance(j, int) and (j < 0 or j >= self.n_cols):
            raise IndexError(f"Column index {j} out of bounds for n_cols={self.n_cols}")

        if isinstance(i, int) and i * self.n_cols == 0:
            return j

        if isinstance(j, int) and j == 0:
            return i * self.n_cols if self.n_cols != 1 else i

        return i * self.n_cols + j if self.n_cols != 1 else i + j

    def __getitem__(
        self,
        key: Union[
            ScalarInt, slice, Tuple[Union[ScalarInt, slice], Union[ScalarInt, slice]]
        ],
    ):
        """
        2D indexing:
         • arr[i, j] → returns the QUA‐VarRef for slot (i,j)
         • arr[i] → returns a RowView so you can do arr[i][j]
         • arr[i, :] → returns a list of QUA variables for row i
         • arr[:, j] → returns a list of QUA variables for column j
        """
        if self.var is None:
            raise RuntimeError(f"{self.name} not declared yet")

        if isinstance(key, tuple):
            row_spec, col_spec = key

            # Check if we have slices
            row_is_slice = isinstance(row_spec, slice)
            col_is_slice = isinstance(col_spec, slice)

            if not row_is_slice and not col_is_slice:
                return self.var[self._flat_index(row_spec, col_spec)]

            # Handle slicing
            if row_is_slice:
                row_indices = range(*row_spec.indices(self.n_rows))
            else:
                row_indices = [row_spec]

            if col_is_slice:
                col_indices = range(*col_spec.indices(self.n_cols))
            else:
                col_indices = [col_spec]

            # Return lists based on slice structure
            if row_is_slice and not col_is_slice:
                # Column extraction: [:, j] -> list of size n_rows
                return [self.var[self._flat_index(r, col_spec)] for r in row_indices]

            elif not row_is_slice and col_is_slice:
                # Row extraction: [i, :] -> list of size n_cols
                return [self.var[self._flat_index(row_spec, c)] for c in col_indices]

            else:
                # 2D slice: [:, :] -> list of lists
                return [
                    [self.var[self._flat_index(r, c)] for c in col_indices]
                    for r in row_indices
                ]

        else:
            # Single index
            if isinstance(key, slice):
                # arr[start:stop] -> returns list of RowViews
                rows = range(*key.indices(self.n_rows))
                return [_QUA2DRow(self, r) for r in rows]

            # return a small proxy so you can do arr[i][j]
            return _QUA2DRow(self, key)

    def assign(
        self,
        row: ScalarInt,
        col_or_vals: Union[ScalarInt, Sequence, QuaArrayVariable],
        val: Scalar = None,
    ):
        """
        Generalized assign:
        - assign(row, col, value) → one element
        - assign(row, [v0, v1, …]) → entire row from Python list/ndarray
        - assign(row, qua_array) → entire row from a QuaArray-like

        This allows you to assign a single value to a specific cell,
        assign an entire row from a Python list or numpy array, or
        assign an entire row from another QUA array variable.
        Note that the row index is 0-based.
        This method overrides the default assign method to handle 2D arrays and does not
        propose the same handling of conditional assignment as the original QUA assign.

        :param row: Row index (0-based)
        :param col_or_vals: Either a column index (0-based) or a sequence of values
        :param val: Value to assign if col_or_vals is a column index
        """
        if self.var is None:
            raise RuntimeError(f"{self.name} must be declared first")

        # Case A: assign(row, col, value)
        if val is not None:
            col = col_or_vals
            qua_assign(self[row, col], val)
            return

        # Case B: assign(row, sequence)
        seq = col_or_vals
        # allow numpy arrays
        if isinstance(seq, np.ndarray):
            if seq.ndim != 1:
                raise ValueError(
                    f"Expected a 1D array for row assignment, got {seq.ndim}D array"
                )
            seq = seq.tolist()
        if isinstance(seq, List) and all(
            isinstance(item, (int, float, bool)) for item in seq
        ):
            # already a list of numbers, no conversion needed
            if len(seq) != self.n_cols:
                raise ValueError(
                    f"Length mismatch: trying to assign {len(seq)} values into a row of length {self.n_cols}"
                )
            # Check type of elements in the list and match with QUA type
            if not all(isinstance(item, type(seq[0])) for item in seq):
                raise TypeError(
                    f"All elements in the list must be of same type: {type(seq[0])}"
                )
            if self.type is fixed and not all(isinstance(item, float) for item in seq):
                raise TypeError(
                    f"All elements must be of type float for QUA fixed type, got {type(seq[0])}"
                )
            elif not all(isinstance(item, self.type) for item in seq):
                raise TypeError(
                    f"All elements must be of type {self.type} for QUA type, got {type(seq[0])}"
                )
            for j in range(self.n_cols):
                qua_assign(self[row, j], seq[j])
            return
        # Case C: assign(row, qua_array) → where qua_array is a QUA‐array
        elif isinstance(seq, QuaArrayVariable):
            with for_(self._ctr, 0, self._ctr < self.n_cols, self._ctr + 1):
                qua_assign(self[row, self._ctr], seq[self._ctr])
            return

    def stream_processing(
        self,
        mode: Literal["save", "save_all"] = "save_all",
        buffer: Union[Tuple[int], int] = None,
    ):
        """
        Stream processing for the 2D array.
        - mode: "save" to save only the last row, "save_all" to save all rows.
        - buffer: size of the buffer to use for streaming.
        """
        if mode not in ["save", "save_all"]:
            raise ValueError("mode must be either 'save' or 'save_all'")

        if buffer is None:
            buffer = (self.n_rows, self.n_cols)
        elif isinstance(buffer, int):
            buffer = (buffer, self.n_cols)
        if self.stream is not None:
            if buffer is not None:
                stream = self.stream.buffer(*buffer)
            else:
                stream = self.stream
            getattr(stream, mode)(self.name)
        else:
            raise ValueError("Output stream is not declared for this QUA2DArray")

    def push_to_opx(
        self,
        value: Union[np.ndarray, List[List[Number]]],
        job: RunningQmJob,
        verbosity: int = 1,
        time_out: int = 30,
    ):
        """
        Push the 2D array to the OPX.
        - value: single value or a sequence of values to push.
        - job: RunningQmJob instance to use for pushing.
        - verbosity: level of verbosity for the operation.
        - time_out: time out in seconds for the operation.
        """

        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a numpy array or a list of lists")

        if isinstance(value, np.ndarray):
            if value.ndim != 2 or value.shape != (self.n_rows, self.n_cols):
                raise ValueError(
                    f"Value must be a 2D array of shape ({self.n_rows}, {self.n_cols})"
                )
            value = value.flatten().tolist()

        if isinstance(value, list) and all(
            isinstance(row, (list, tuple)) for row in value
        ):
            if len(value) != self.n_rows or any(
                len(row) != self.n_cols for row in value
            ):
                raise ValueError(
                    f"Value must be a 2D list of shape ({self.n_rows}, {self.n_cols})"
                )
            value = [item for row in value for item in row]

        # Push the flattened list to the OPX
        super().push_to_opx(
            value=value,
            job=job,
            verbosity=verbosity,
            time_out=time_out,
        )


# auxiliary class so arr[i][j] works
class _QUA2DRow:
    def __init__(self, parent: QUA2DArray, row: int):
        self._parent = parent
        self._row = row

    def __getitem__(self, col: Union[ScalarInt, slice]):
        if isinstance(col, slice):
            # Support row[:] syntax
            col_indices = range(*col.indices(self._parent.n_cols))
            return [self._parent[self._row, c] for c in col_indices]
        return self._parent[self._row, col]

    def assign(
        self,
        col_or_vals: Union[ScalarInt, Sequence, QuaArrayVariable],
        val: Scalar = None,
    ):
        """Delegate to the parent’s assign."""
        self._parent.assign(self._row, col_or_vals, val)

    def __len__(self):
        return self._parent.n_cols
