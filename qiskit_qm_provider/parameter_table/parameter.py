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

"""Parameter class: mapping of a parameter to a QUA variable for runtime updates.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

import copy
import warnings
from typing import Optional, List, Union, Tuple, Literal, Sequence, TYPE_CHECKING, Dict, Any, Type

import numpy as np
from qm.qua import (
    fixed,
    assign,
    declare,
    pause,
    declare_input_stream,
    save,
    for_,
    advance_input_stream as qua_advance_input_stream,
    declare_stream as qua_declare_stream,
    IO1,
    IO2,
    if_,
    Util,
)
from qm.qua._expressions import QuaArrayVariable
from qm.jobs.running_qm_job import RunningQmJob
from qm import QuantumMachine
from qm.api.v2.job_api import JobApi
from qualang_tools.results import wait_until_job_is_paused

from .parameter_pool import ParameterPool
from .input_type import Direction, InputType

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from qm.qua.type_hints import (
        Scalar,
        Vector,
        VectorOfAnyType,
        ScalarOfAnyType,
        QuaScalar,
        ResultStreamSource,
    )


def set_type(qua_type: str | Type[int | float | bool]) -> Type[int | float | bool]:
    """
    Set the type of the QUA variable to be declared.
    Args: qua_type: Type of the QUA variable to be declared (int, fixed, bool).
    """

    if qua_type == "fixed" or qua_type == fixed:
        return fixed
    elif qua_type == "bool" or qua_type == bool:
        return bool
    elif qua_type == "int" or qua_type == int:
        return int
    else:
        raise ValueError("Invalid QUA type. Please use 'fixed', 'int' or 'bool'.")


def infer_type(value: Optional[Union[int, float, bool, List, np.ndarray]] = None) -> Type[int | float | bool]:
    """
    Infer automatically the type of the QUA variable to be declared from the type of the initial parameter value.
    """
    if value is None:
        raise ValueError("Initial value must be provided to infer type.")

    # Handle scalar values
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return fixed if not value.is_integer() or value <= 8 else int

    # Handle array values
    if isinstance(value, (List, np.ndarray)):
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("Array must be 1D")
            value = value.tolist()

        if not value:
            raise ValueError("Array cannot be empty")

        first_type = type(value[0])
        if not all(isinstance(x, first_type) for x in value):
            raise ValueError("All array elements must be of same type")

        if first_type == bool:
            return bool
        if first_type == int:
            return int
        if first_type == float:
            return fixed

        raise ValueError("Array elements must be bool, int or float")

    raise ValueError("Value must be bool, int, float or array")


def reset_var(var: QuaScalar, type: Type[int | float | bool]):
    """
    Reset the QUA variable to a 0 value (in the appropriate QUA type).
    """
    if type == int:
        assign(var, 0)
    elif type == fixed:
        assign(var, 0.0)
    elif type == bool:
        assign(var, False)
    else:
        raise ValueError("Invalid QUA type. Please use 'int', 'fixed' or 'bool'.")


#: Mapping from a Parameter-level method name (the ``context`` string passed into
#: :meth:`Parameter._require_standalone_opnic_table`) to the equivalent ParameterTable
#: method that callers should use when the parameter is field-managed.
_OPNIC_OWNER_METHOD_MAP: Dict[str, str] = {
    "declare_variable": "declare_variables",
    "push_to_opx": "push_to_opx",
    "fetch_from_opx": "fetch_from_opx",
    "load_input_value": "load_input_values",
    "stream_back": "stream_back",
}


def _resolve_struct_stream_ids(handle: Any) -> tuple[Optional[int], Optional[int]]:
    """Extract (incoming_id, outgoing_id) from a Quarc handle-like object."""
    struct_spec = getattr(handle, "_struct_spec", None)
    if struct_spec is None:
        return None, None
    incoming = getattr(struct_spec, "incoming_stream_spec", None)
    outgoing = getattr(struct_spec, "outgoing_stream_spec", None)
    return getattr(incoming, "id", None), getattr(outgoing, "id", None)


def _resolve_stream_id_for_direction(
    handle: Any, direction: Optional[Direction]
) -> Optional[int]:
    """Resolve the stream id for qiskit-qm-provider direction semantics."""
    incoming_id, outgoing_id = _resolve_struct_stream_ids(handle)
    if direction == Direction.OUTGOING:
        # OPNIC -> OPX maps to Quarc incoming stream.
        return incoming_id
    if direction == Direction.INCOMING:
        # OPX -> OPNIC maps to Quarc outgoing stream.
        return outgoing_id
    if direction == Direction.BOTH:
        return incoming_id if incoming_id is not None else outgoing_id
    return None


def _assert_compatible_existing_parameter(
    existing: "Parameter",
    *,
    requested_value: Any,
    requested_qua_type: Any,
    requested_input_type: Optional[InputType],
    requested_direction: Optional[Direction],
    requested_units: str,
) -> None:
    """Validate that re-declaring ``Parameter(name=existing.name, ...)`` would not change
    its semantics. If the requested args genuinely differ from ``existing``'s current
    state, raise :class:`ValueError` with a precise diff. Used by :meth:`Parameter.__new__`.

    The caller has already asserted that ``existing.input_type`` is *not* ``OPNIC`` and
    that the requested ``input_type`` is *not* ``OPNIC`` either — OPNIC collisions are
    rejected unconditionally before we get here.
    """
    diffs: List[str] = []

    # qua_type: either explicitly requested, or omitted (None), in which case we accept.
    if requested_qua_type is not None:
        normalized_requested = set_type(requested_qua_type)
        if existing.type is not normalized_requested:
            diffs.append(
                f"qua_type: existing={existing.type!r}, requested={normalized_requested!r}"
            )

    # input_type: only flag if the user passed a non-None value that disagrees.
    if requested_input_type is not None and existing.input_type != requested_input_type:
        diffs.append(
            f"input_type: existing={existing.input_type!r}, requested={requested_input_type!r}"
        )

    # direction: only flag if the user passed a non-None value that disagrees.
    if requested_direction is not None and existing._direction != requested_direction:
        diffs.append(
            f"direction: existing={existing._direction!r}, requested={requested_direction!r}"
        )

    # length / array shape: derived from `value`. Only check if the user actually passed one.
    if requested_value is not None:
        existing_length = existing.length
        if isinstance(requested_value, (list, np.ndarray)):
            requested_length = len(requested_value)
        else:
            requested_length = 0
        if existing_length != requested_length:
            diffs.append(
                f"length: existing={existing_length}, requested={requested_length}"
            )

    # units: only flag a non-empty mismatch.
    if requested_units and existing.units != requested_units:
        diffs.append(f"units: existing={existing.units!r}, requested={requested_units!r}")

    if diffs:
        raise ValueError(
            f"Parameter named {existing.name!r} already exists in the pool with "
            f"different attributes. Diffs:\n  - "
            + "\n  - ".join(diffs)
            + "\nIf you intended to reuse the existing parameter, drop the conflicting "
            "constructor arguments. If you intended a fresh one, choose a different name."
        )


class Parameter:
    """
    Class enabling the mapping of a parameter to a QUA variable to be updated. The type of the QUA variable to be
    adjusted can be declared explicitly or either be automatically inferred from the type of provided initial value.
    """

    def __new__(
        cls,
        name: str,
        value: Optional[Union[int, float, bool, List, np.ndarray]] = None,
        qua_type: Optional[Union[str, type]] = None,
        input_type: Optional[Union[Literal["OPNIC", "INPUT_STREAM", "IO1", "IO2"], InputType]] = None,
        direction: Optional[Union[Literal["INCOMING", "OUTGOING", "BOTH"], Direction]] = None,
        units: str = "",
    ):
        """Construct, or look up, the :class:`Parameter` named ``name``.

        Lookup semantics (Option 1 — validating dedup, OPNIC-strict):

        - If no parameter named ``name`` exists in the pool (registry + pending
          standalone OPNIC list), a fresh instance is constructed and returned.
        - If a parameter named ``name`` already exists, **and either the existing one
          or the requested one is OPNIC**, the call is rejected with a
          :class:`ValueError`: OPNIC parameters are single-owner and cannot be
          re-declared or shared by re-construction.
        - Otherwise (both non-OPNIC), the existing instance is returned **only after**
          validating that all requested constructor arguments match its current state.
          Mismatches (e.g. different ``qua_type``, ``input_type``, ``direction``,
          ``length``, ``units``) raise :class:`ValueError` with a per-field diff so the
          user catches accidental aliasing instead of silently dropping the new args.

        This replaces the historical "warn-and-return-existing" behavior, which silently
        clobbered the new arguments.
        """
        # Normalize for the lookup-side comparison.
        if isinstance(input_type, str):
            input_type_norm: Optional[InputType] = InputType(input_type)
        else:
            input_type_norm = input_type
        if isinstance(direction, str):
            direction_norm: Optional[Direction] = Direction(direction)
        else:
            direction_norm = direction

        existing = ParameterPool._lookup_parameter_by_name(name)
        if existing is None:
            return super().__new__(cls)

        # OPNIC-strict: any collision involving an OPNIC parameter raises.
        if existing.input_type == InputType.OPNIC or input_type_norm == InputType.OPNIC:
            raise ValueError(
                f"Parameter named {name!r} already exists in the pool with "
                f"input_type={existing.input_type!r}; OPNIC parameters cannot be "
                f"re-declared or shared. Use a different name, or fetch the existing "
                f"parameter via ParameterPool._lookup_parameter_by_name(name)."
            )

        # Non-OPNIC: validate compatibility, then return the existing instance.
        _assert_compatible_existing_parameter(
            existing,
            requested_value=value,
            requested_qua_type=qua_type,
            requested_input_type=input_type_norm,
            requested_direction=direction_norm,
            requested_units=units,
        )
        return existing

    def __init__(
        self,
        name: str,
        value: Optional[Union[int, float, List, np.ndarray]] = None,
        qua_type: Optional[Union[str, type]] = None,
        input_type: Optional[
            Union[Literal["OPNIC", "INPUT_STREAM", "IO1", "IO2"], InputType]
        ] = None,
        direction: Optional[
            Union[Literal["INCOMING", "OUTGOING", "BOTH"], Direction]
        ] = None,
        units: str = "",
    ):
        """

        Args:
            name: Name of the parameter.
            value: Initial value of the parameter.
            qua_type: Type of the QUA variable to be declared (int, fixed, bool). If none is provided, the type is inferred from initial value.
            input_type: Input type of the parameter (OPNIC, INPUT_STREAM, IO1, IO2). Default is None.
            direction: Direction of the parameter stream (INCOMING, OUTGOING, BOTH).
                For OPNIC, direction describes data flow vs OPX:
                OPNIC -> OPX: OUTGOING
                OPX -> OPNIC: INCOMING
                OPNIC <-> OPX: BOTH
                Default is None. Relevant only if input_type is OPNIC.
            units: Units of the parameter. Default is "".

        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._name = name
        self.units = units
        self.value = value
        self._index = -1  # Default value for parameters not part of a parameter table
        self._var = None
        self._is_declared = False
        self._stream = None
        self._type = set_type(qua_type) if qua_type is not None else infer_type(value)
        self._length = 0 if not isinstance(value, (List, np.ndarray)) else len(value)
        self._ctr: Optional[QuaScalar[int]] = None  # Counter for QUA array variables

        self._qua_external_stream_in = None
        self._qua_external_stream_out = None

        if input_type is not None:
            input_type = (
                InputType(input_type) if isinstance(input_type, str) else input_type
            )
        self._input_type: Optional[InputType] = input_type
        if direction is not None:
            direction = (
                Direction(direction) if isinstance(direction, str) else direction
            )
        self._direction = direction
        self._table_indices: Dict[str, int] = {}
        #: The owning :class:`ParameterTable`, or ``None`` while still pending. For a
        #: promoted standalone OPNIC parameter, this points to the synthetic
        #: single-field ``ParameterTable`` (which has ``_is_synthetic_standalone=True``).
        self._main_table: Optional["ParameterTable"] = None

        if self._input_type == InputType.OPNIC and self.direction is None:
            raise ValueError("Direction must be provided for OPNIC input type.")

        self._initialized = True
        # OPNIC parameters are tracked in :data:`ParameterPool._pending_standalone_opnic`
        # at construction so that (a) :meth:`Parameter.__new__` can de-dup by name across
        # the whole pool, and (b) the user can still attach the parameter to a regular
        # ``ParameterTable`` later. If the parameter is *not* attached to a table by the
        # time the user calls :meth:`declare_variable` (or any other transport method),
        # it is promoted into a synthetic single-field ``ParameterTable`` and this entry
        # is removed from the pending list.
        if self._input_type == InputType.OPNIC:
            ParameterPool._register_pending_standalone_opnic(self)

    def __repr__(self):
        """
        Returns:
            str: String representation of the parameter.
        """
        return f"Parameter({self.name}{self.units}[{self.type.__name__}])"

    def get_index(self, param_table: ParameterTable) -> int:
        """
        Get the index of the parameter in the parameter table.
        Args:
            param_table: ParameterTable object to get the index from.
        Returns:
            int: Index of the parameter in the parameter table.
        """
        if self._index == -1:
            raise ValueError(
                "This parameter is not part of a parameter table. "
                "Please use this method through the parameter table instead."
            )
        if param_table.name not in self._table_indices:
            raise ValueError(
                f"Parameter {self.name} is not part of the parameter table {param_table.name}."
            )
        return self._table_indices[param_table.name]

    def set_index(self, param_table: ParameterTable, index: int):
        """
        Set the index of the parameter in the parameter table.

        Args:
            param_table: ``ParameterTable`` object to attach to.
            index: Index of the parameter in the parameter table.

        For OPNIC parameters, attaching to a regular (non-synthetic) ``ParameterTable``
        also clears the parameter from :data:`ParameterPool._pending_standalone_opnic`,
        so the parameter is no longer a standalone candidate. Attempting to attach an
        already-promoted standalone parameter (whose ``main_table`` is a synthetic
        single-field table) to a *different* ``ParameterTable`` raises.
        """
        # If the parameter was already promoted to a synthetic standalone table, reject
        # any attempt to attach it elsewhere — the locking semantics require single
        # ownership for OPNIC fields.
        if (
            self._main_table is not None
            and getattr(self._main_table, "_is_synthetic_standalone", False)
            and self._main_table is not param_table
        ):
            raise ValueError(
                f"Parameter {self.name!r} is already finalized as a standalone OPNIC "
                f"field (synthetic table "
                f"{getattr(self._main_table, 'name', '<unnamed>')!r}); cannot attach to "
                f"another ParameterTable {param_table.name!r}."
            )
        if self._index == -1:
            self._index = -2

        if param_table.name not in self._table_indices:
            self._table_indices[param_table.name] = index
        else:
            raise ValueError(
                f"Parameter {self.name} is already part of the parameter table {param_table.name}."
            )
        if self._main_table is None:
            self._main_table = param_table

        # Remove from the pending standalone list when attached to a non-synthetic
        # multi-or-single-field table. Synthetic standalone tables also call set_index,
        # but they remove the entry themselves to keep this branch simple.
        if (
            self._input_type == InputType.OPNIC
            and not getattr(param_table, "_is_synthetic_standalone", False)
        ):
            ParameterPool._unregister_pending_standalone_opnic(self)

    def is_standalone(self) -> bool:
        """Backward-compat alias for :pyattr:`is_stand_alone`."""
        return self.is_stand_alone

    @property
    def is_stand_alone(self) -> bool:
        """Whether this parameter is treated as a standalone OPNIC field.

        A parameter is "standalone" if any of:

        - it has no owning ``ParameterTable`` (still pending — has not been attached
          and has not been promoted yet); or
        - it has been promoted into a synthetic single-field ``ParameterTable``
          (``_is_synthetic_standalone == True``).

        Once attached to a regular multi-or-single-field ``ParameterTable``, the
        parameter is *not* standalone anymore.
        """
        if self._main_table is None:
            return True
        if getattr(self._main_table, "_is_synthetic_standalone", False):
            return True
        return False

    @property
    def opnic_table(self):
        """The synthetic single-field ``ParameterTable`` wrapping this parameter, or
        ``None`` if the parameter has not been promoted yet (still pending) or is part
        of a regular multi-field table.
        """
        if self._main_table is not None and getattr(
            self._main_table, "_is_synthetic_standalone", False
        ):
            return self._main_table
        return None

    @property
    def main_table(self) -> Optional[ParameterTable]:
        """
        Returns:
            The ParameterTable object used to declare the parameter.
            Specifically, the one that should be used for communication if InputType is OPNIC.
        :returns: ParameterTable object or None if not found.

        """
        return self._main_table

    def assign(
        self,
        value: Union["Parameter", ScalarOfAnyType, VectorOfAnyType],
        condition=None,
        value_cond: Optional[
            Union["Parameter", ScalarOfAnyType, VectorOfAnyType]
        ] = None,
    ):
        """
        Assign value to the QUA variable corresponding to the parameter.

        Args:
            value: Value to be assigned to the QUA variable. If the ParameterValue corresponds to a QUA array,
                   the value should be a list or a QUA array of the same length.
            condition: Condition to be met for the value to be assigned to the QUA variable.
            value_cond: Optional value to be assigned to the QUA variable if provided condition is not met.

        Raises:
            ValueError: If the variable is not declared, or if the condition and value_cond are not provided together,
                        or if the value_cond is not of the same type as value, or if the value length does not match
                        the parameter length, or if the input is invalid.
        """
        if not self.is_declared:
            raise ValueError(
                "Variable not declared. Declare the variable first through declare_variable method."
            )
        if (condition is not None) != (value_cond is not None):
            raise ValueError("Both condition and value_cond must be provided.")

        def assign_with_condition(var, val, cond_val):
            if condition is not None:
                assign(var, Util.cond(condition, val, cond_val))
            else:
                assign(var, val)

        if isinstance(value, Parameter):
            if not value.is_declared:
                raise ValueError(
                    "Variable not declared. Declare the variable first through declare_variable method."
                )
            if value.length != self.length:
                raise ValueError(
                    f"Invalid input. Mismatch in length of {self.name} ({self.length}) and {value.name} ({value.length})."
                )
            if value_cond is not None and not isinstance(value_cond, Parameter):
                raise ValueError(
                    "Invalid input. value_cond should be of same type as value."
                )
            if self.is_array:
                with for_(self._ctr, 0, self._ctr < self.length, self._ctr + 1):
                    assign_with_condition(
                        self.var[self._ctr],
                        value.var[self._ctr],
                        value_cond.var[self._ctr] if value_cond else None,
                    )
            else:
                assign_with_condition(
                    self.var, value.var, value_cond.var if value_cond else None
                )
        else:
            if self.is_array:
                if isinstance(value, QuaArrayVariable):
                    with for_(self._ctr, 0, self._ctr < self.length, self._ctr + 1):
                        assign_with_condition(
                            self.var[self._ctr],
                            value[self._ctr],
                            value_cond[self._ctr] if value_cond is not None else None,
                        )
                else:
                    if len(value) != self.length:
                        raise ValueError(
                            f"Invalid input. {self.name} should be a list of length {self.length}."
                        )
                    for i in range(self.length):
                        assign_with_condition(
                            self.var[i], value[i], value_cond[i] if value_cond else None
                        )
            else:
                if isinstance(value, List):
                    raise ValueError(
                        f"Invalid input. {self.name} should be a single value, not a list."
                    )
                assign_with_condition(self.var, value, value_cond)

    def declare_variable(self, pause_program=False, declare_stream=True):
        """
        Declare the QUA variable associated with the parameter.
        Args: pause_program: Boolean indicating if the program should be paused after declaring the variable.
            Default is False.
        declare_stream: Boolean indicating if an output stream should be declared to save the QUA variable.
        """
        if self.input_type == InputType.OPNIC:
            opnic_table = self._require_standalone_opnic_table(context="declare_variable")
            opnic_table.declare_variables(
                pause_program=pause_program, declare_streams=declare_stream
            )
            return self._var
        if self.is_declared:
            raise ValueError("Variable already declared. Cannot declare again.")
        else:
            if self.input_type == InputType.INPUT_STREAM:
                # if self.is_array:
                    # self._var = declare_input_stream('client', self.name, self.type, size=self.length)
                # else:
                #     self._var = declare_input_stream('client', self.name, self.type)
                self._var = declare_input_stream(self.name, self.type, value=self.value)
            else:
                if self.value is not None:
                    self._var = declare(self.type, value=self.value)
                else:
                    self._var = declare(self.type, size=self.length if self.is_array else None)
        if self.is_array and self.length > 1:
            self._ctr = declare(int)
        if pause_program:
            pause()
        self._is_declared = True
        if declare_stream:
            self.declare_stream()
        return self._var

    def declare_stream(self):
        """
        Declare the output stream associated with the parameter.
        """
        if self._stream is None:
            # self._stream = declare_output_stream(target='client', stream_id=self.name, dtype=self.type)
            self._stream = qua_declare_stream()
        else:
            warnings.warn(f"Stream {self.name} already declared, skipping declaration.")
        return self._stream

    @property
    def is_declared(self):
        """Boolean indicating if the QUA variable has been declared."""
        return self._is_declared

    @property
    def name(self):
        """Name of the parameter."""
        return self._name

    @property
    def direction(self):
        """
        Direction of the parameter stream (INCOMING, OUTGOING, BOTH).
        For OPNIC, direction describes data flow vs OPX:
        OPNIC -> OPX: OUTGOING
        OPX -> OPNIC: INCOMING
        OPNIC <-> OPX: BOTH
        Default is None. Relevant only if input_type is OPNIC.
        """
        if self.input_type != InputType.OPNIC:
            warnings.warn("This parameter is not associated with an OPNIC stream.")
            raise ValueError("This parameter is not associated with an OPNIC stream.")
        return self._direction

    @property
    def opnic_struct(self):
        """The Quarc :class:`QuaStructHandle` (or pybind runtime endpoint) bound to the
        ``ParameterTable`` that owns this OPNIC parameter.

        For a parameter inside a regular multi-field ``ParameterTable``, returns
        ``main_table._var``. For a promoted standalone parameter, returns
        ``opnic_table._var`` (which is the synthetic single-field table's handle).
        Raises :class:`RuntimeError` if the parameter is OPNIC but still pending (no
        owning table yet — call :meth:`declare_variable` to promote, or attach it to a
        ``ParameterTable``).
        """
        if self.input_type != InputType.OPNIC:
            raise ValueError(
                "Invalid input type for calling opnic_struct property. "
                "Must be set to InputType.OPNIC."
            )
        if self._main_table is None:
            raise RuntimeError(
                f"Parameter {self.name!r} is OPNIC but has not been bound to a "
                f"ParameterTable yet. Either attach it to a ParameterTable or call "
                f"declare_variable() / to_quarc_module() to promote it as a standalone."
            )
        return self._main_table._var

    @property
    def stream_id(self) -> int:
        """Stream id of the owning OPNIC table (delegates to the table's id)."""
        if self._input_type != InputType.OPNIC:
            raise ValueError("Stream ID is only defined for OPNIC parameters.")
        if self._main_table is None:
            raise RuntimeError(
                f"Parameter {self.name!r} is OPNIC and not bound to a ParameterTable."
            )
        return self._main_table.stream_id

    @property
    def incoming_stream_id(self) -> int:
        """Incoming stream id exposed by the owning OPNIC table."""
        if self._input_type != InputType.OPNIC:
            raise ValueError("Incoming stream ID is only defined for OPNIC parameters.")
        if self._main_table is None:
            raise RuntimeError(
                f"Parameter {self.name!r} is OPNIC and not bound to a ParameterTable."
            )
        owner = self._main_table
        if not hasattr(owner, "incoming_stream_id"):
            return owner.stream_id
        return owner.incoming_stream_id

    @property
    def outgoing_stream_id(self) -> int:
        """Outgoing stream id exposed by the owning OPNIC table."""
        if self._input_type != InputType.OPNIC:
            raise ValueError("Outgoing stream ID is only defined for OPNIC parameters.")
        if self._main_table is None:
            raise RuntimeError(
                f"Parameter {self.name!r} is OPNIC and not bound to a ParameterTable."
            )
        owner = self._main_table
        if not hasattr(owner, "outgoing_stream_id"):
            return owner.stream_id
        return owner.outgoing_stream_id

    @property
    def var(self) -> QuaArrayVariable | QuaScalar:
        """
        Returns:
            QUA variable associated with the parameter.
        """
        if not self.is_declared:
            raise ValueError(
                "Variable not declared. Declare the variable first through declare_variable method."
            )
        return self._var

    def _require_standalone_opnic_table(self, *, context: str) -> "ParameterTable":
        """Return the synthetic single-field ``ParameterTable`` that owns this parameter.

        Promotes the parameter on first access (P2 model — promote on use): if the
        parameter is still pending (no owning table), construct a synthetic single-field
        :class:`ParameterTable` named after the parameter, which will eagerly emit its
        struct via the pool's Quarc module (lazily creating a default
        :class:`quarc.BaseModule` if none is bound yet).

        If the parameter is attached to a regular multi-or-single-field
        ``ParameterTable``, OPNIC transport is *table-managed* and this method raises
        :class:`RuntimeError` directing the caller to use the owning table's API.

        ``context`` names the calling Parameter-level method (``"declare_variable"``,
        ``"push_to_opx"``, …) and is used to suggest the equivalent table-level method
        in the error message.
        """
        if not self.is_stand_alone:
            owning_table_name = (
                self.main_table.name if self.main_table is not None else "<unknown>"
            )
            owner_method = _OPNIC_OWNER_METHOD_MAP.get(context, context)
            raise RuntimeError(
                f"Parameter.{context}() is table-managed for OPNIC fields "
                f"(parameter={self.name!r}, table={owning_table_name!r}). "
                f"Use ParameterTable.{owner_method}() on the owning table instead."
            )

        # Promote on use if still pending.
        if self.opnic_table is None:
            self._promote_to_synthetic_standalone_table()

        synthetic = self.opnic_table
        if synthetic is None:
            raise RuntimeError(
                f"Standalone OPNIC parameter {self.name!r} could not be promoted to a "
                f"synthetic ParameterTable during {context}."
            )
        return synthetic

    def _promote_to_synthetic_standalone_table(self) -> "ParameterTable":
        """Build the synthetic single-field ``ParameterTable`` that wraps this parameter.

        The synthetic table is constructed with ``_is_synthetic_standalone=True`` so
        that the hybrid emission rule force-emits its Quarc struct immediately
        (creating a default :class:`quarc.BaseModule` via
        :meth:`ParameterPool.quarc_module` if no module is bound yet). After
        construction this parameter's :attr:`main_table` points at the synthetic table,
        and the entry is removed from :data:`ParameterPool._pending_standalone_opnic`.

        We remove the parameter from the pending list *before* constructing the
        synthetic table: pool-level name-uniqueness consults both the registry and the
        pending list, so leaving the parameter pending while registering a synthetic
        table named after it would (rightly) trigger a duplicate-name guard.
        """
        if self.input_type != InputType.OPNIC:
            raise RuntimeError(
                f"Cannot promote non-OPNIC parameter {self.name!r} to a synthetic "
                f"standalone ParameterTable."
            )
        from .parameter_table import ParameterTable

        ParameterPool._unregister_pending_standalone_opnic(self)
        synthetic = ParameterTable(
            [self], name=self.name, _is_synthetic_standalone=True
        )
        return synthetic

    @property
    def tables(self) -> List[ParameterTable]:
        """
        Returns:
            List of ParameterTable objects associated with the parameter.
        """
        from .parameter_table import ParameterTable

        tables = []
        for table in ParameterPool.get_all_objs():
            if isinstance(table, ParameterTable) and table.has_parameter(self):
                tables.append(table)
        return tables

    @property
    def type(self):
        """Type of the associated QUA variable."""
        return self._type

    @type.setter
    def type(self, value: Union[str, type]):
        if self.is_declared:
            raise ValueError("Variable already declared. Cannot change type.")
        self._type = set_type(value)

    @property
    def length(self):
        """Length of the parameter if it refers to a QUA array (
        returns 0 if single value)."""
        return self._length

    @property
    def input_type(self) -> Optional[InputType]:
        """
        Type of input stream associated with the parameter.
        """
        return self._input_type

    @property
    def is_array(self):
        """Boolean indicating if the parameter refers to a QUA array."""
        return self.length > 0

    @property
    def stream(self) -> ResultStreamSource:
        """Output stream associated with the parameter."""
        return self._stream

    def save_to_stream(self, reset: bool = False):
        """
        Save the QUA variable to the output stream.

        Args:
            reset: Whether to reset the parameter to a 0 value (in the appropriate QUA type) after saving it to the stream.
        Raises:
            ValueError: If the output stream is not declared, or if the variable is not declared.
        """
        if self.is_declared and self.stream is not None:
            if self.is_array:
                with for_(self._ctr, 0, self._ctr < self.length, self._ctr + 1):
                    save(self.var[self._ctr], self.stream)
                    if reset:
                        reset_var(self.var[self._ctr], self.type)
            else:
                save(self.var, self.stream)
                if reset:
                    reset_var(self.var, self.type)
        else:
            raise ValueError("Output stream or variable itself not declared.")

    def stream_processing(
        self,
        mode: Literal["save", "save_all"] = "save_all",
        buffer: Optional[Union[Tuple[int, ...], int, Literal["default"]]] = "default",
    ):
        """
        Process the output stream associated with the parameter.
        Args:
            mode: Mode of processing the stream. Can be "save" or "save_all". Default is "save_all".
            buffer: Buffer size for the stream. If "default", the default buffer size is used (no buffer for a single variable
                and buffer of array size for an array). Can also be set to None for no buffer.
        """
        if mode not in ["save", "save_all"]:
            raise ValueError("Invalid mode. Must be 'save' or 'save_all'.")
        if buffer == "default":
            if self.is_array:
                buffer = (self.length,)
            else:
                buffer = None
        elif isinstance(buffer, int):
            buffer = (buffer,)
        if self.stream is not None:
            if buffer is not None:
                stream = self.stream.buffer(*buffer)
            else:
                stream = self.stream
            getattr(stream, mode)(self.name)
        else:
            raise ValueError("Output stream not declared.")

    def clip(
        self,
        min_val: Optional[
            Scalar[int], Scalar[float], Vector[int], Vector[float]
        ] = None,
        max_val: Optional[
            Scalar[int], Scalar[float], Vector[int], Vector[float]
        ] = None,
        is_qua_array: bool = False,
    ):
        """
        Clip the QUA variable to a given range.
        Args: min_val: Minimum value of the range.
            max_val: Maximum value of the range.
            is_array: Boolean indicating if the bounds are QUA arrays.
        """
        if not self.is_declared:
            raise ValueError(
                "Variable not declared. Declare the variable first through declare_variable method."
            )
        if not self.is_array and is_qua_array:
            raise ValueError(
                "Invalid input. Single value cannot be clipped with array bounds."
            )
        elif (
            isinstance(min_val, (int, float))
            and isinstance(max_val, (int, float))
            and min_val > max_val
        ):
            raise ValueError(
                "Invalid range. Minimum value must be less than maximum value."
            )

        elif min_val is None and max_val is None:
            warnings.warn("No range specified. No clipping performed.")
            return

        if self.is_array:
            i = self._ctr
            with for_(i, 0, i < self.length, i + 1):
                if is_qua_array:
                    if min_val is not None:
                        with if_(self.var[i] < min_val[i]):
                            assign(self.var[i], min_val[i])
                    if max_val is not None:
                        with if_(self.var[i] > max_val[i]):
                            assign(self.var[i], max_val[i])
                else:
                    if min_val is not None:
                        with if_(self.var[i] < min_val):
                            assign(self.var[i], min_val)
                    if max_val is not None:
                        with if_(self.var[i] > max_val):
                            assign(self.var[i], max_val)
        else:
            if min_val is not None:
                with if_(self.var < min_val):
                    assign(self.var, min_val)
            if max_val is not None:
                with if_(self.var > max_val):
                    assign(self.var, max_val)

    def load_input_value(self):
        """
        QUA Macro: Load a value from the input mechanism associated with the parameter.
        This should be corresponding to one call of the `fetch_from_opx` method on the client side.
        For input streams, the stream is advanced.
        For IO1 and IO2, the value is assigned to the QUA variable.
        For OPNIC, the value is polled.
        """
        if self.input_type is None:
            raise ValueError("No input type specified")
        elif self.input_type == InputType.INPUT_STREAM:
            qua_advance_input_stream(self.var)

        elif self.input_type == InputType.OPNIC:
            self._require_standalone_opnic_table(context="load_input_value").load_input_values()

        elif self.input_type in [InputType.IO1, InputType.IO2]:
            io = IO1 if self.input_type == InputType.IO1 else IO2
            if self.is_array:
                if self.length == 1:
                    pause()
                    assign(self.var[0], io)
                    return
                with for_(self._ctr, 0, self._ctr < self.length, self._ctr + 1):
                    pause()
                    assign(self.var[self._ctr], io)
            else:
                pause()
                assign(self.var, io)

        else:
            raise ValueError("Invalid input stream type.")

    def push_to_opx(
        self,
        value: Union[int, float, bool, Sequence[Union[int, float, bool]]],
        job: Optional[RunningQmJob | JobApi] = None,
        qm: Optional[QuantumMachine] = None,
        verbosity: int = 1,
        time_out: int = 30,
    ):
        """
        Client function: pass an input value to the OPX from client/server side.
        This should be corresponding to one call of the `load_input_value` method on the QUA side.
        Args:
            value: Value to be passed to the OPX.
            job: RunningQmJob object (required if input_type is IO1 or IO2 or input_stream).
            qm: QuantumMachine object (required if input_type is IO1 or IO2).
            verbosity: Verbosity level. Default is 1.
            time_out: Time out for waiting for the job to be paused. Default is 90 seconds.
        """

        if self.is_array:
            if not isinstance(value, (List, np.ndarray)):
                raise ValueError(
                    f"Invalid input. {self.name} should be a list of {self.type}."
                )
            if len(value) != self.length:
                raise ValueError(
                    f"Invalid input. {self.name} should be a list of length {self.length}."
                )
        param_type = self.type if self.type != fixed else float
        if self.is_array and not all(isinstance(x, param_type) for x in value):
            try:
                value = [param_type(x) for x in value]
            except ValueError:
                raise ValueError(
                    f"Invalid input. {self.name} should be a list of {param_type}."
                )
        elif not self.is_array and not isinstance(value, param_type):
            try:
                value = param_type(value)
            except ValueError:
                raise ValueError(
                    f"Invalid input. {self.name} should be a single value of type {param_type}."
                )

        if self.input_type in [InputType.IO1, InputType.IO2]:
            if job is None:
                raise ValueError("Job object is required to set IO values.")
            io = "io1" if self.input_type == InputType.IO1 else "io2"
            if self.is_array:
                for i in range(self.length):
                    if verbosity > 1:
                        print(f"Setting {self.name} to {value[i]} through {io}")
                    wait_until_job_is_paused(job, time_out)
                    if isinstance(job, JobApi):
                        job.set_io_values(**{io: value[i]})
                    else:
                        if qm is None:
                            raise ValueError(
                                "QuantumMachine object is required to set IO values."
                            )
                        qm.set_io_values(**{io: value[i]})
                    job.resume()
            else:
                if verbosity > 1:
                    print(f"Setting {self.name} to {value} through {io}")
                wait_until_job_is_paused(job, time_out)
                if isinstance(job, JobApi):
                    # For JobApi, we need to use the set_io_values method
                    job.set_io_values(**{io: value})
                else:
                    if qm is None:
                        raise ValueError(
                            "QuantumMachine object is required to set IO values."
                        )
                    qm.set_io_values(**{io: value})
                job.resume()

        elif self.input_type == InputType.INPUT_STREAM:
            if verbosity > 1:
                print(f"Pushing value {value} to {self.name} through input stream.")
            if job is None:
                raise ValueError(
                    "Job object is required to push values to the input stream."
                )
            job.push_to_input_stream(self.name, value)

        elif self.input_type == InputType.OPNIC:
            self._require_standalone_opnic_table(context="push_to_opx").push_to_opx(
                {self.name: value}, job=job, qm=qm, verbosity=verbosity
            )

    def stream_back(self, reset: bool = False):
        """
        QUA Macro: Save/stream the value of the parameter to the client/server side.
        This method uses stream_processing to save the value to the stream if input_type is not OPNIC.
        If input_type is OPNIC, the value is sent to the external stream.

        Args:
            reset: Whether to reset the parameter to a 0 value (in the appropriate QUA type) after sending it to the client/server side.
        """
        if (
            self.input_type in [InputType.INPUT_STREAM, None]
            and self.stream is not None
        ):
            self.save_to_stream(reset)
        elif self.input_type == InputType.OPNIC:
            self._require_standalone_opnic_table(context="stream_back").stream_back(reset=reset)
        elif self.input_type in [InputType.IO1, InputType.IO2]:
            io = IO1 if self.input_type == InputType.IO1 else IO2
            if self.is_array:
                i = self._ctr
                with for_(i, 0, i < self.length, i + 1):
                    assign(io, self.var[i])
                    if reset:
                        reset_var(self.var[i], self.type)
                    pause()

            else:
                assign(io, self.var)
                if reset:
                    reset_var(self.var, self.type)
                pause()

    def fetch_from_opx(
        self,
        job: Optional[RunningQmJob | JobApi] = None,
        fetching_index: int = 0,
        fetching_size: int = 1,
        verbosity: int = 1,
        time_out=30,
    ):
        """
        Client function: Fetches data based on the specified input type and returns the fetched value.

        This method handles various input types defined by the `InputType` enumeration
        (IO1, IO2, INPUT_STREAM, OPNIC). It manages the fetching logic, including waiting
        for paused jobs, accessing specified result streams, and interacting with
        external modules when necessary. For OPNIC it also checks configurations
        and fetches data based on parameters related to outgoing or incoming streams.

        :param job: The job instance of the RunningQmJob for which data is being fetched.
        :type job: RunningQmJob
        :param qm: The QuantumMachine instance utilized for fetching the data. Defaults to None.
        :param fetching_index: The starting index for fetching data when required. Defaults to 0.
        :type fetching_index: int
        :param fetching_size: Number of items to fetch from the source, if applicable. Defaults to 1.
        :type fetching_size: int
        :param verbosity: Level of output verbosity for log printing. A verbosity > 1 enables detailed logging.
        :type verbosity: int
        :param time_out: Time in seconds to wait for the job to be paused before fetching data. Defaults to 30 seconds.
        :return: The fetched value depending upon the input type and fetching logic.
        """
        if self.input_type == InputType.INPUT_STREAM or self.input_type is None:
            if job is None:
                raise ValueError(
                    "Job object is required to fetch values from the result handles."
                )
            if verbosity > 1:
                print(
                    f"Fetching value from {self.name} with input type {self.input_type}"
                )
            result_handle = job.result_handles
            if self.name not in result_handle:
                raise ValueError(
                    f"Parameter {self.name} not found in the result handles. "
                    "Make sure to save the parameter to the stream first."
                )
            result = result_handle.get(self.name)
            result.wait_for_values(fetching_index + fetching_size, time_out)
            value = result.fetch(slice(fetching_index, fetching_index + fetching_size))[
                "value"
            ]

        elif self.input_type == InputType.OPNIC:
            res = self._require_standalone_opnic_table(context="fetch_from_opx").fetch_from_opx(
                job=job,
                fetching_index=fetching_index,
                fetching_size=fetching_size,
                verbosity=verbosity,
                time_out=time_out,
            )
            return res[self.name]
        elif self.input_type in [InputType.IO1, InputType.IO2]:
            io_method = (
                "get_io1_value" if self.input_type == InputType.IO1 else "get_io2_value"
            )
            if self.is_array:
                value = []
                for i in range(self.length):
                    wait_until_job_is_paused(job, time_out)
                    value.append(getattr(job, io_method)(self.type))
                    job.resume()
            else:
                wait_until_job_is_paused(job, time_out)
                value = getattr(job, io_method)(self.type)
                job.resume()
        else:
            raise ValueError("Invalid input type.")
        if verbosity > 1:
            print(f"Fetched value: {value}")
        return value

    def reset(self):
        """Client function: reset transient QUA-side declaration state.

        Clears ``_is_declared``, ``_var``, ``_stream``, ``_ctr``. For OPNIC parameters,
        deliberately preserves ``_main_table`` (the owning ``ParameterTable``, real or
        synthetic) so the binding to module/runtime transport survives across resets.
        """
        self._is_declared = False
        self._var = None
        self._stream = None
        self._ctr = None


    def reset_var(self):
        """
        QUA Macro: Assign the QUA variable to 0 (in the appropriate QUA type).
        """
        if self.is_array:
            with for_(self._ctr, 0, self._ctr < self.length, self._ctr + 1):
                reset_var(self.var[self._ctr], self.type)
        else:
            reset_var(self.var, self.type)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        if id(self) in memodict:
            return memodict[id(self)]
        cls = self.__class__
        new_param = object.__new__(cls)
        memodict[id(self)] = new_param

        # Now, manually populate new_param with copied attributes:
        new_param._name = self._name
        new_param.units = self.units
        new_param._type = self._type
        new_param._input_type = self._input_type
        new_param._direction = self._direction
        new_param.value = copy.deepcopy(self.value, memodict)  # Deepcopy mutable values
        new_param._length = self._length

        # Reset QUA-specific and context-dependent state
        new_param._var = None
        new_param._is_declared = False
        new_param._stream = None
        new_param._ctr = None
        new_param._qua_external_stream_in = None
        new_param._qua_external_stream_out = None

        # Reset table/OPNIC-specific attributes that will be set by the new table or properties
        new_param._index = -1  # Default for a parameter not (yet) in a table
        new_param._table_indices = {}  # New dictionary for table associations
        new_param._main_table = None

        # If your __init__ sets an _initialized flag, set it here too
        if hasattr(self, "_initialized"):
            new_param._initialized = True

        return new_param
