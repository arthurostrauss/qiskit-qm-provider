from numbers import Number

from qm.qua import declare, assign as qua_assign, fixed, for_
from quam.utils.qua_types import QuaVariableInt, Scalar, ScalarInt

from .parameter import Parameter
from typing import Tuple, Union, List, Sequence, Literal
import numpy as np
from qm.qua._dsl import QuaArrayVariable


class QUA2DArray(Parameter):
    """
    A 2D view over one big 1D QUA array.
    QUA array of length n_rows * n_cols and gives 2D indexing.
    """

    def __init__(
        self,
        name: str,
        n_rows: int,
        n_cols: int,
        value: np.ndarray | List[List[Number]] = None,
        qua_type: Union[str, type] = None,
        input_type=None,
        direction=None,
        units: str = "",
    ):
        # prepare an "initial" 1D list of zeros so that Parameter can infer length/type
        length = n_rows * n_cols
        if value is not None:
            if isinstance(value, np.ndarray):
                if value.ndim != 2 or value.shape != (n_rows, n_cols):
                    raise ValueError(f"Value must be a 2D array of shape ({n_rows}, {n_cols})")
                init_list = value.flatten().tolist()
            elif isinstance(value, list) and all(isinstance(row, list) for row in value):
                if len(value) != n_rows or any(len(row) != n_cols for row in value):
                    raise ValueError(f"Value must be a 2D list of shape ({n_rows}, {n_cols})")
                init_list = [item for row in value for item in row]
            else:
                raise TypeError("Value must be a 2D numpy array or a list of lists")
        else:
            # if no value is given, initialize with zeros
            if not isinstance(n_rows, int) or not isinstance(n_cols, int):
                raise TypeError("n_rows and n_cols must be integers")
            if n_rows < 1 or n_cols < 1:
                raise ValueError("n_rows and n_cols must be strictly positive integers")
            if n_rows * n_cols > 1000000:
                raise ValueError("2D array size exceeds maximum limit of 1,000,000 elements")

            init_list = [0] * length

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

        # placeholders; will be set when we declare our QUA variable
        self._counter_var = None

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

    def __getitem__(self, key: Union[ScalarInt, Tuple[ScalarInt, ScalarInt]]):
        """
        2D indexing:
         • arr[i, j] → returns the QUA‐VarRef for slot (i,j)
         • arr[i] → returns a RowView so you can do arr[i][j]
        """
        if self.var is None:
            raise RuntimeError(f"{self.name} not declared yet")
        if isinstance(key, tuple):
            i, j = key
            return self.var[self._flat_index(i, j)]

        else:
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
                raise ValueError(f"Expected a 1D array for row assignment, got {seq.ndim}D array")
            seq = seq.tolist()
        if isinstance(seq, List) and all(isinstance(item, (int, float, bool)) for item in seq):
            # already a list of numbers, no conversion needed
            if len(seq) != self.n_cols:
                raise ValueError(
                    f"Length mismatch: trying to assign {len(seq)} values into a row of length {self.n_cols}"
                )
            # Check type of elements in the list and match with QUA type
            if not all(isinstance(item, type(seq[0])) for item in seq):
                raise TypeError(f"All elements in the list must be of same type: {type(seq[0])}")
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
            with for_(self._counter_var, 0, self._counter_var < self.n_cols, self._counter_var + 1):
                qua_assign(self[row, self._counter_var], seq[self._counter_var])
            return

    def stream_processing(
        self, mode: Literal["save", "save_all"] = "save_all", buffer: Union[Tuple[int], int] = None
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


# auxiliary class so arr[i][j] works
class _QUA2DRow:
    def __init__(self, parent: QUA2DArray, row: int):
        self._parent = parent
        self._row = row

    def __getitem__(self, col: int):
        return self._parent[self._row, col]

    def assign(self, col_or_vals: Union[ScalarInt, Sequence, QuaArrayVariable], val: Scalar = None):
        """Delegate to the parent’s assign."""
        self._parent.assign(self._row, col_or_vals, val)
