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

"""Compiler-owned measurement output fields wired from qm-qasm compilation results."""

from __future__ import annotations

import weakref
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

from qm.qua import assign, declare, declare_stream
from qm.qua.type_hints import QuaScalar

from ..parameter_table._scope import require_qua_program, requires_qua_program
from .backend_utils import pack_register_to_int, _measurement_var_is_array

if TYPE_CHECKING:
    from qm_qasm import CompilationResult


class MeasurementRegisterField:
    """One classical register (or loose bit) wired from a qm-qasm compilation result.

    Measurement fields live in a **separate namespace** from runtime
    :class:`~qiskit_qm_provider.parameter_table.Parameter` objects registered in
    :class:`~qiskit_qm_provider.parameter_table.ParameterPool`. The same string name
    (e.g. matching a creg and an input struct field) may refer to both a runtime knob
    and a measurement output; they are distinct Python objects with distinct QUA variables.
    """

    def __init__(
        self,
        name: str,
        register_size: int,
        *,
        compute_state_int: bool = True,
        parent: Optional[Any] = None,
    ):
        """Create a measurement field handle before wiring from a compilation result.

        Args:
            name: Output key in ``result_program`` (creg name, ``_bitN``, or other
                compiler output key).
            register_size: Number of classical bits represented by this field when
                packed into :attr:`state_int`.
            compute_state_int: If ``True``, :attr:`state_int` lazily packs bits into an
                ``int`` QUA variable.
            parent: Optional owning :class:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation` (stored as weakref).
        """
        self.name = name
        self._register_size = register_size
        self._compute_state_int = compute_state_int
        self._parent = weakref.ref(parent) if parent is not None else None
        self._var = None
        self._is_declared = False
        self._state_int_var: Optional[QuaScalar[int]] = None
        self._should_reset_state_int = False
        self._stream = None
        self._wired_program_key: Optional[str] = None
        self._wired_compilation_uuid: Optional[str] = None
        self._var_is_array: bool = False

    @property
    def is_measurement_output(self) -> bool:
        """``True`` — sentinel used to reject attachment to runtime ``ParameterTable``\\ s."""
        return True

    @property
    def is_declared(self) -> bool:
        """Whether :meth:`_wire_from_result` has bound a compiler-owned QUA variable."""
        return self._is_declared

    @property
    def size(self) -> int:
        """Number of classical bits in this register (``1`` for loose clbits)."""
        return self._register_size

    @property
    def var_is_array(self) -> bool:
        """Whether :attr:`var` is a QUA array (``True``) or scalar (``False``).

        Set when the field is wired from ``result_program``; packing in
        :attr:`state_int` follows this shape, not circuit metadata.
        """
        return self._var_is_array

    @property
    def var(self):
        """Compiler-owned QUA bool scalar or array for this measurement output.

        Must be accessed inside ``with program():``. Use
        :meth:`~qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.get_parameter`
        for the Python handle.
        """
        require_qua_program("MeasurementRegisterField.var")
        if not self._is_declared:
            raise ValueError(
                f"Measurement field {self.name!r} is not wired yet. "
                f"Compile the circuit with quantum_circuit_to_qua first."
            )
        return self._var

    def _wire_from_result(
        self,
        result: "CompilationResult",
        program_key: str,
        *,
        register_size: Optional[int] = None,
    ) -> None:
        """Bind compiler-owned QUA variables from ``result.result_program``.

        Called during compilation wiring, not by user code. Invalidates cached
        :attr:`state_int` and :attr:`stream` handles when register size or compilation
        identity changes.

        Args:
            result: qm-qasm compilation result exposing ``result_program[key]``.
            program_key: Key in ``result.result_program`` (creg name or ``_bitN``).
            register_size: Override bit width; defaults to the value from ``__init__``.
        """
        new_size = register_size if register_size is not None else self._register_size
        compilation_uuid = getattr(result, "uuid", None)

        if new_size != self._register_size:
            self._register_size = new_size
            self._state_int_var = None
            self._stream = None
        elif self._wired_program_key != program_key or self._wired_compilation_uuid != compilation_uuid:
            self._state_int_var = None
            self._stream = None

        self._var = result.result_program[program_key]
        self._var_is_array = _measurement_var_is_array(self._var)
        if not self._var_is_array and self._register_size > 1:
            raise ValueError(
                f"Measurement field {self.name!r} expects a {self._register_size}-bit "
                f"packed output (QUA array) but wired variable is a scalar "
                f"({type(self._var).__name__})."
            )
        self._is_declared = True
        self._wired_program_key = program_key
        self._wired_compilation_uuid = compilation_uuid
        if self._state_int_var is not None:
            self._should_reset_state_int = True

    @property
    def state_int(self) -> QuaScalar[int]:
        """Lazy-packed ``int`` QUA variable (LSB = bit index 0).

        Must be accessed inside ``with program():``. Prefer bulk access via
        :attr:`~qiskit_qm_provider.backend.qua_circuit_compilation.MeasurementOutcomeTable.state_ints`
        when wiring multiple registers.

        Raises:
            AttributeError: If ``compute_state_int=False`` was passed at construction.
        """
        require_qua_program("MeasurementRegisterField.state_int")
        if not self._compute_state_int:
            raise AttributeError(
                f"state_int is disabled for {self.name!r}; use .var instead " f"(compute_state_int=False)."
            )
        return self._materialize_state_int()

    @property
    def stream(self):
        """Output stream for ``save(..., stream)`` / ``stream_processing()`` on the host.

        Declared lazily on first access inside ``with program():``.
        """
        require_qua_program("MeasurementRegisterField.stream")
        if self._stream is None:
            self._stream = declare_stream()
        return self._stream

    def _materialize_state_int(self) -> QuaScalar[int]:
        """Declare (once) and assign the packed integer from :attr:`var`."""
        packed = pack_register_to_int(self._var, self._register_size)
        if self._state_int_var is None:
            self._state_int_var = declare(int)
            assign(self._state_int_var, packed)
        elif self._should_reset_state_int:
            assign(self._state_int_var, packed)
            self._should_reset_state_int = False
        return self._state_int_var

    @requires_qua_program
    def declare_stream(self):
        """Declare the output stream if not already created; return it."""
        if self._stream is None:
            self._stream = declare_stream()
        return self._stream

    def __deepcopy__(self, memo):
        """Refuse duplication — handles are tied to a single compilation wiring."""
        raise TypeError("MeasurementRegisterField handles are compilation-owned and cannot be deepcopied.")
