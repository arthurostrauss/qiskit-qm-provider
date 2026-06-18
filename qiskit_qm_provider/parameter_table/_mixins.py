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

"""Shared QUA field-table accessor mixin for runtime and measurement tables."""

from __future__ import annotations

from typing import Any, Protocol, Union, runtime_checkable

from qm.qua._expressions import QuaArrayVariable
from quam.utils.qua_types import QuaVariable

from ._scope import require_qua_program


@runtime_checkable
class TableFieldProtocol(Protocol):
    """Minimal interface shared by runtime :class:`Parameter` and measurement fields.

    Implemented by :class:`~qiskit_qm_provider.parameter_table.Parameter` and
    :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField`.
    Used by :class:`QuaFieldTable` accessors.
    """

    name: str

    @property
    def is_declared(self) -> bool:
        """Whether a QUA variable is bound for this field."""

    @property
    def var(self) -> QuaVariable | QuaArrayVariable:
        """QUA variable for this field (requires program scope when read)."""

    @property
    def is_array(self) -> bool:
        """Whether this field holds a QUA array (``True``) or a scalar (``False``)."""

    @property
    def length(self) -> int:
        """Number of elements: ``0`` for a scalar, the array length otherwise."""


class QuaFieldTable:
    """Mixin providing ParameterTable-aligned QUA variable accessors.

    Subclasses must define ``self.table: dict[str, TableFieldProtocol]`` and may
    override :meth:`_resolve_index` for index lookup semantics.
    """

    table: dict[str, TableFieldProtocol]

    def _resolve_index(self, index: int) -> TableFieldProtocol:
        """Return the field at positional ``index`` (default: insertion order).

        Override in :class:`~.parameter_table.ParameterTable` to use
        :meth:`~.parameter_table.Parameter.get_index`.
        """
        for i, field in enumerate(self.table.values()):
            if i == index:
                return field
        raise IndexError(f"No parameter with index {index} in the parameter table.")

    def get_parameter(self, parameter: Union[str, int, TableFieldProtocol]) -> TableFieldProtocol:
        """Return the field handle (not the QUA variable).

        Args:
            parameter: Field name, positional index, or handle instance.

        Returns:
            :class:`~qiskit_qm_provider.parameter_table.Parameter` or
            :class:`~qiskit_qm_provider.backend.measurement_field.MeasurementRegisterField`
            depending on table type.
        """
        if isinstance(parameter, str):
            if parameter not in self.table:
                raise KeyError(f"No parameter named {parameter} in the parameter table.")
            return self.table[parameter]
        if isinstance(parameter, int):
            return self._resolve_index(parameter)
        if parameter in self.parameters:
            return parameter
        raise KeyError("Provided field not in this parameter table.")

    def get_variable(self, parameter: Union[str, int, TableFieldProtocol]) -> QuaVariable | QuaArrayVariable:
        """Return the QUA variable for a field name, index, or handle.

        Must be called inside ``with program():`` when resolving by index or handle.
        String names delegate to :meth:`__getitem__`.
        """
        if isinstance(parameter, str):
            return self[parameter]
        if isinstance(parameter, int):
            field = self._resolve_index(parameter)
            require_qua_program(f"{type(self).__name__}.get_variable")
            if not field.is_declared:
                raise ValueError(
                    f"No QUA variable found for parameter with index {parameter}. "
                    f"Please declare variables within a QUA program first."
                )
            return field.var
        if parameter in self.parameters:
            require_qua_program(f"{type(self).__name__}.get_variable")
            if not parameter.is_declared:
                raise ValueError(
                    f"No QUA variable found for parameter {parameter.name!r}. "
                    f"Please declare variables within a QUA program first."
                )
            return parameter.var
        raise ValueError("Invalid parameter name. Please use a string, int, or field handle.")

    def __getitem__(self, item: Union[str, int]) -> QuaVariable | QuaArrayVariable:
        """Return the QUA variable for a field name or index.

        Must be called inside ``with program():``. For the Python handle, use
        :meth:`get_parameter`.
        """
        require_qua_program(f"{type(self).__name__}.__getitem__")
        if isinstance(item, str):
            if item not in self.table:
                raise KeyError(f"No parameter named {item} in the parameter table.")
            field = self.table[item]
            if field.is_declared:
                return field.var
            raise ValueError(
                f"No QUA variable found for parameter {item}. " f"Please declare variables within a QUA program first."
            )
        if isinstance(item, int):
            field = self._resolve_index(item)
            if field.is_declared:
                return field.var
            raise ValueError(
                f"No QUA variable found for parameter with index {item}. "
                f"Please declare variables within a QUA program first."
            )
        raise ValueError("Invalid parameter name. Please use a string or an int.")

    def __getattr__(self, item: str) -> Any:
        """Attribute access by field name — returns the QUA variable (not the handle)."""
        if item in self.table:
            require_qua_program(f"{type(self).__name__}.__getattr__")
            return self.table[item].var
        raise AttributeError(f"No attribute named {item} in the parameter table.")

    def __len__(self) -> int:
        """Number of fields in the table."""
        return len(self.table)

    @property
    def variables(self) -> list[QuaVariable | QuaArrayVariable]:
        """List of QUA variables for all fields (requires program scope)."""
        return [self[item] for item in self.table]
