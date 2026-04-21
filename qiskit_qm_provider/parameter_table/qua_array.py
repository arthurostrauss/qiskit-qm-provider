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

    The underlying QUA allocation is always 1D (QUA only supports 1D arrays natively).
    Multi-dimensional indexing is emulated via row-major (C-order) stride arithmetic,
    i.e. element [i, j, k] maps to flat index  i*s0 + j*s1 + k*s2  where strides are
    computed from the shape at construction time.

    Indices may be a mix of plain Python ``int`` (resolved at compile time) and QUA
    integer variables (resolved at runtime).  The flat-index expression produced by
    ``_flat_index`` is therefore either a Python ``int`` or a QUA arithmetic expression,
    both of which are valid as QUA array subscripts.

    Supports:
        - Full indexing:   ``arr[i, j, k]``  → single QUA variable
        - Partial indexing: ``arr[i]``         → ``_QUAArrayView`` proxy
        - Slice expansion:  ``arr[i, :]``      → Python list of variables / views
        - Whole-array assign: ``arr.assign(value)``
        - Element assign:    ``arr.assign((i, j), val)``  (delegates to view for partial)
    """

    def __init__(
        self,
        name: str,
        shape: Optional[Tuple[int, ...]] = None,
        value: Optional[Union[List[Any], np.ndarray]] = None,
        qua_type: Optional[Union[str, type]] = None,
        input_type=None,
        direction=None,
        units: str = "",
    ):
        """
        Exactly one of ``shape`` or ``value`` must be provided.

        Args:
            name:       Name of the parameter.
            shape:      Tuple of ints describing the array dimensions, e.g. ``(3, 4)``.
                        The underlying 1D QUA array is zero-initialised.
            value:      A Python list (possibly nested) or numpy array whose shape is
                        used as the array dimensions and whose flattened contents are
                        used as initial values.
            qua_type:   QUA type of the elements (e.g. ``fixed``, ``int``, ``bool``).
            input_type: Input type forwarded to ``Parameter``.
            direction:  Direction forwarded to ``Parameter`` (OPNIC only).
            units:      Units string forwarded to ``Parameter``.

        Raises:
            TypeError:  If neither or both of ``shape``/``value`` are supplied, or if
                        their types are wrong.
        """
        if shape is None and value is None:
            raise TypeError("Exactly one of 'shape' or 'value' must be provided.")
        if shape is not None and value is not None:
            raise TypeError(
                "Provide either 'shape' or 'value', not both. "
                "If you want to initialise with data, pass the array as 'value'; "
                "its shape is inferred automatically."
            )

        if shape is not None:
            if not (
                isinstance(shape, tuple) and all(isinstance(d, int) for d in shape)
            ):
                raise TypeError(
                    f"'shape' must be a tuple of ints, got {type(shape)}: {shape!r}. "
                    "To initialise from data use the 'value' kwarg instead."
                )
            self.shape: Tuple[int, ...] = shape
            init_list = [0] * int(np.prod(shape))

        else:  # value is not None
            arr = np.asarray(value)
            self.shape = arr.shape
            init_list = arr.flatten().tolist()

        # Row-major (C-order) strides: stride[i] = product(shape[i+1:])
        # Stored as a tuple so it is immutable after construction.
        strides: List[int] = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        self.strides: Tuple[int, ...] = tuple(strides)

        super().__init__(
            name=name,
            value=init_list,
            qua_type=qua_type,
            input_type=input_type,
            direction=direction,
            units=units,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_index(self, *indices: ScalarInt) -> ScalarInt:
        """
        Convert an N-D index tuple into a flat 1-D index using pre-computed strides.

        Works for any mix of Python ``int`` and QUA integer variables:
        - Pure Python ints  → returns a plain Python ``int`` (zero overhead).
        - Any QUA variable  → returns a QUA arithmetic expression that the QUA
          compiler will evaluate at runtime on the QPU.

        Bounds-checking is performed only for Python ints (QUA variables have no
        compile-time value to check).

        Raises:
            IndexError: Wrong number of indices, or a Python int is out of bounds.
        """
        if len(indices) != len(self.shape):
            raise IndexError(
                f"Expected {len(self.shape)} indices, got {len(indices)}. "
                f"Array shape is {self.shape}."
            )

        flat_idx = 0
        for i, (ind, stride, dim_size) in enumerate(
            zip(indices, self.strides, self.shape)
        ):
            if isinstance(ind, int):
                if not (0 <= ind < dim_size):
                    raise IndexError(
                        f"Index {ind} out of bounds for dimension {i} "
                        f"(size {dim_size})."
                    )
                if ind == 0:
                    continue  # contributes nothing; skip to avoid spurious QUA ops

            term = ind if stride == 1 else ind * stride

            flat_idx = (
                term
                if (isinstance(flat_idx, int) and flat_idx == 0)
                else flat_idx + term
            )

        return flat_idx

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Multi-dimensional element / sub-array access.

        Behaviour by key type:
        - ``arr[i, j, k]``  (full rank, no slices) → single QUA variable.
        - ``arr[i]``         (partial rank)         → ``_QUAArrayView`` proxy.
        - ``arr[i, :]``      (contains a slice)     → Python list of variables or
                                                       ``_QUAArrayView`` objects,
                                                       obtained by expanding the slice
                                                       and recursing.

        Notes:
            Slice expansion is done purely in Python (compile time).  It is therefore
            only valid when the slice bounds resolve to Python ints, which is always
            the case for ``:``, ``::2``, etc.  Slicing with QUA-variable start/stop
            is not supported and will raise ``TypeError`` inside ``slice.indices()``.
        """
        if self.var is None:
            raise RuntimeError(
                f"QUAArray '{self.name}' has not been declared yet. "
                "Call declare() before accessing elements."
            )

        if not isinstance(key, tuple):
            key = (key,)

        # Expand the leftmost slice found, recurse for the rest.
        for i, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, step = k.indices(self.shape[i])
                return [
                    self[key[:i] + (v,) + key[i + 1 :]]
                    for v in range(start, stop, step)
                ]

        # No slices remain — key contains only scalar indices (Python int or QUA var).
        if len(key) == len(self.shape):
            # Full rank → return the QUA variable directly.
            return self.var[self._flat_index(*key)]
        elif len(key) < len(self.shape):
            # Partial rank → return a view proxy for further indexing / assignment.
            return _QUAArrayView(self, key)
        else:
            raise IndexError(
                f"Too many indices: got {len(key)}, array has {len(self.shape)} dimensions."
            )

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(self, indices_or_val, val=None):
        """
        Flexible element / whole-array assignment.

        Calling conventions
        -------------------
        ``arr.assign(value)``
            Whole-array assign.  ``value`` must be list/ndarray/QUA array accepted
            by ``Parameter.assign``.  Delegates directly to the parent implementation.

        ``arr.assign((i,), row_value)``
        ``arr.assign((i, j), scalar_val)``
            Partial or full index assign.  ``indices_or_val`` must be a tuple of
            indices (Python ints or QUA variables).  For partial indices a
            ``_QUAArrayView`` is constructed and its ``assign`` is called, which
            handles the flat ``for_`` loop internally.  For a complete index tuple a
            direct ``qua_assign`` is emitted.

        Note: single-element shorthand ``arr.assign(i, val)`` is intentionally *not*
        supported — always wrap indices in a tuple to avoid ambiguity with the
        whole-array form.
        """
        if val is None:
            # Whole-array assign — delegate to Parameter.
            super().assign(indices_or_val)
            return

        # Index-based assign.
        if not isinstance(indices_or_val, tuple):
            raise TypeError(
                "Indices must be passed as a tuple, e.g. arr.assign((i, j), val). "
                "For whole-array assignment omit the second argument."
            )

        indices = indices_or_val

        if len(indices) == len(self.shape):
            # Full index → single element assign, no view needed.
            qua_assign(self[indices], val)
        elif len(indices) < len(self.shape):
            # Partial index → build a view and delegate.
            # _QUAArrayView.assign handles the flat for_ loop for arbitrary sub-shapes.
            _QUAArrayView(self, indices).assign(val)
        else:
            raise IndexError(
                f"Too many indices in assign: got {len(indices)}, "
                f"array has {len(self.shape)} dimensions."
            )

    # ------------------------------------------------------------------
    # OPX I/O
    # ------------------------------------------------------------------

    def push_to_opx(
        self,
        value: Union[np.ndarray, List],
        job: RunningQmJob,
        verbosity: int = 1,
        time_out: int = 30,
    ):
        """
        Push an N-D array of values into the OPX at runtime.

        ``value`` is validated against ``self.shape``, flattened to a 1-D list in
        row-major order, and forwarded to ``Parameter.push_to_opx``.  This mirrors
        exactly what ``QUA2DArray.push_to_opx`` does for the 2-D case, generalised
        to arbitrary rank.

        Args:
            value:     A numpy array or (possibly nested) Python list whose shape
                       must equal ``self.shape``.
            job:       The ``RunningQmJob`` instance returned by ``qm.execute()``.
            verbosity: Verbosity level forwarded to ``Parameter.push_to_opx``.
            time_out:  Timeout in seconds forwarded to ``Parameter.push_to_opx``.

        Raises:
            TypeError:  ``value`` is not a numpy array or list.
            ValueError: ``value`` shape does not match ``self.shape``.
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError(
                f"'value' must be a numpy array or a (nested) list, "
                f"got {type(value).__name__}."
            )

        arr = np.asarray(value)

        if arr.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: QUAArray '{self.name}' has shape {self.shape} "
                f"but 'value' has shape {arr.shape}."
            )

        # Flatten to 1-D in row-major (C) order — matches the flat QUA allocation.
        super().push_to_opx(
            value=arr.flatten().tolist(),
            job=job,
            verbosity=verbosity,
            time_out=time_out,
        )

    def stream_processing(
        self,
        mode: Literal["save", "save_all"] = "save_all",
        buffer: Union[Tuple[int, ...], int, None] = None,
    ):
        """
        Declare stream-processing for this N-D array.

        The QUA stream API expects a flat stream to be buffered into a shape before
        saving.  This method builds the correct buffer tuple and calls
        ``stream.buffer(*buffer).save[_all](name)``.

        Buffer resolution
        -----------------
        ``None`` (default)
            Use ``self.shape`` as the buffer — the stream is reshaped back to the
            array's natural N-D shape.  Equivalent to ``QUA2DArray``'s default of
            ``(n_rows, n_cols)``.

        ``int`` (leading-dimension shorthand)
            Prepend the integer to ``self.shape``, yielding a buffer of shape
            ``(int, *self.shape)``.  This is useful when the array is streamed once
            per repetition and you want to accumulate ``int`` repetitions before
            saving.  Equivalent to ``QUA2DArray``'s ``buffer=int`` shorthand which
            produced ``(int, n_cols)``.

        ``tuple``
            Used as-is.  Must be a tuple of positive ints; no further validation of
            total size is performed here (the QUA compiler will catch mismatches).

        Args:
            mode:   ``"save"`` to keep only the last buffer, ``"save_all"`` to
                    accumulate all buffers (default).
            buffer: Buffer shape — see above.

        Raises:
            ValueError: ``mode`` is invalid or the stream has not been declared.
            TypeError:  ``buffer`` is not ``None``, an ``int``, or a ``tuple``.
        """
        if mode not in ("save", "save_all"):
            raise ValueError(f"mode must be 'save' or 'save_all', got {mode!r}.")

        if buffer is None:
            # Default: reshape stream back to the array's own N-D shape.
            buffer_tuple: Tuple[int, ...] = self.shape
        elif isinstance(buffer, int):
            # Leading-dimension shorthand: accumulate ``buffer`` repetitions of the
            # full array, producing a (buffer, *self.shape) result.
            buffer_tuple = (buffer,) + self.shape
        elif isinstance(buffer, tuple):
            buffer_tuple = buffer
        else:
            raise TypeError(
                f"'buffer' must be None, an int, or a tuple of ints, "
                f"got {type(buffer).__name__}."
            )

        if self.stream is None:
            raise ValueError(
                f"Output stream is not declared for QUAArray '{self.name}'. "
                "Ensure the stream variable is set before calling stream_processing()."
            )

        getattr(self.stream.buffer(*buffer_tuple), mode)(self.name)


class _QUAArrayView:
    """
    Lightweight proxy representing a sub-array slice of a ``QUAArray``.

    Instances are returned by ``QUAArray.__getitem__`` on partial indexing and by
    ``QUAArray.assign`` for partial-index writes.  Users should not normally
    construct these directly.

    The view's shape is ``parent.shape[len(indices):]``.  Its flat base offset in
    the underlying 1D QUA array is ``parent._flat_index(*indices, 0, 0, ..., 0)``
    (i.e. the flat index of the first element of this sub-array).
    """

    def __init__(self, parent: QUAArray, indices: Tuple):
        self._parent = parent
        self._indices = (
            indices  # already-resolved prefix indices (Python int or QUA var)
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Append ``key`` to the already-resolved prefix indices and delegate back to
        the parent ``QUAArray``.  Supports scalars, tuples, and slices — the full
        ``QUAArray.__getitem__`` machinery handles all cases.
        """
        if not isinstance(key, tuple):
            key = (key,)
        return self._parent[self._indices + key]

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(self, val):
        """
        Assign ``val`` to every element covered by this view.

        Strategy — flatten both sides, emit a single QUA ``for_`` loop
        ---------------------------------------------------------------
        Because the underlying storage is already 1D, assigning to a sub-array
        is equivalent to writing a contiguous (or strided, but here always
        contiguous in our row-major layout) range of the flat buffer.

        The flat offset of this view's first element is::

            base = parent._flat_index(*self._indices, 0, 0, ..., 0)

        The view then occupies ``view_size`` consecutive positions starting at
        ``base``.  A single ``for_`` loop from ``0`` to ``view_size`` is therefore
        sufficient regardless of the view's dimensionality.

        ``val`` must be one of:
        - A Python list / numpy array of length ``view_size`` (compile-time unroll
          if len ≤ a small threshold, otherwise a ``for_`` loop over a QUA array).
        - A 1D QUA array of length ``view_size`` (runtime ``for_`` loop).

        For a Python list the loop is unrolled at compile time (one ``qua_assign``
        per element), which is efficient for small fixed shapes.  For a QUA array
        source a single ``for_`` is generated.

        Raises:
            ValueError: Shape / length mismatch between view and ``val``.
            NotImplementedError: ``val`` is neither a list, ndarray, nor QUA array.
        """
        view_shape = self._parent.shape[len(self._indices) :]
        view_size = int(np.prod(view_shape))

        # Compute the flat index of this view's first element.
        # Append zeros for all remaining dimensions so _flat_index gets the right arity.
        trailing_zeros = (0,) * len(view_shape)
        base: ScalarInt = self._parent._flat_index(*self._indices, *trailing_zeros)

        if isinstance(val, (list, np.ndarray)):
            flat_val = np.asarray(val).flatten().tolist()
            if len(flat_val) != view_size:
                raise ValueError(
                    f"Length mismatch: view covers {view_size} elements but "
                    f"'val' has {len(flat_val)}."
                )
            # Compile-time unroll — no QUA for_ needed.
            # Each assignment emits a single QUA assign instruction.
            for offset, item in enumerate(flat_val):
                flat_pos = base + offset if offset != 0 else base
                qua_assign(self._parent.var[flat_pos], item)

        else:
            # Assume val is a 1D QUA array (or anything subscriptable with a QUA index).
            # Emit a single for_ loop; the loop variable acts as the flat offset.
            ctr = self._parent._ctr
            if ctr is None:
                ctr = declare(int)

            with for_(ctr, 0, ctr < view_size, ctr + 1):
                flat_pos = (
                    base + ctr if (isinstance(base, int) and base == 0) else base + ctr
                )
                qua_assign(self._parent.var[flat_pos], val[ctr])
