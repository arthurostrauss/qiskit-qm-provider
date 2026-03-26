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

"""ParameterPool: management of unique IDs and registration for parameters.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING, Union
import sys

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from .parameter import Parameter


class ParameterPool:
    """
    A class to manage unique IDs for parameters.
    """

    _counter = itertools.count(1)
    _parameters_dict: Dict[int, ParameterTable | Parameter] = {}
    _patched = False
    _configured = False

    @classmethod
    def get_id(cls, obj: Any = None) -> int:
        """
        Get the next unique ID.

        Returns:
            int: The next unique ID.
            obj: The object associated with the ID.
        """
        next_id = next(cls._counter)
        if obj is not None:
            cls._parameters_dict[next_id] = obj
        return next_id

    @classmethod
    def get_obj(cls, id: int) -> ParameterTable | Parameter:
        """
        Get the object associated with the given ID.

        Args:
            id (int): The ID of the object.

        Returns:
            obj: The object associated with the ID.
        """
        return cls._parameters_dict[id]

    @classmethod
    def reset(cls):
        """
        Reset the counter and the dictionary.
        """
        cls._counter = itertools.count(1)
        for value in cls._parameters_dict.values():
            if hasattr(value, "_clear_quarc_stream_specs"):
                value._clear_quarc_stream_specs()
            if hasattr(value, "reset"):
                value.reset()
        cls._parameters_dict.clear()
        cls._configured = False
        cls._patched = False

    @classmethod
    def iter_dgx_quarc_packet_targets(cls) -> List[Union["ParameterTable", "Parameter"]]:
        """
        Every :class:`~qiskit_qm_provider.parameter_table.ParameterTable` and standalone
        :class:`~qiskit_qm_provider.parameter_table.Parameter` in the pool with ``input_type == DGX_Q``,
        sorted by :func:`~qiskit_qm_provider.parameter_table.quarc_naming.quarc_packet_sort_key`
        (same order as Quarc ``add_struct`` when the pool drives synthesis).
        """
        from .input_type import InputType
        from .parameter import Parameter
        from .parameter_table import ParameterTable
        from .quarc_naming import quarc_packet_sort_key

        tables = cls.iter_dgx_parameter_tables()
        standalones: List[Parameter] = []
        for obj in cls.get_all_objs():
            if isinstance(obj, Parameter) and obj.input_type == InputType.DGX_Q and obj.is_standalone():
                standalones.append(obj)
        targets: List[Union[ParameterTable, Parameter]] = [*tables, *standalones]
        # Ensure the private Quarc struct naming hook is consistent across both
        # standalone Parameters and ParameterTables.
        for obj in targets:
            obj._materialize_dgx_stream_id_for_quarc()
        targets.sort(key=quarc_packet_sort_key)
        return targets

    @classmethod
    def prepare_dgx_quarc_hybrid_packets(
        cls,
        *,
        quarc_module_class_name: str | None = None,
    ) -> List[Union["ParameterTable", "Parameter"]]:
        """
        Build an in-memory Quarc ``BaseModule`` (``add_struct`` per target), then copy stream ids onto pool objects.

        Requires the ``quarc`` package. Uses isolated Quarc stream counters so this does not perturb other Quarc
        usage in-process. Call after the pool snapshot is final and before QUA uses ``declare_external_stream``.

        Args:
            quarc_module_class_name: Name of the generated Quarc ``BaseModule`` subclass (must be a valid Python identifier).
                If omitted, uses :data:`quarc_live_module.DEFAULT_QUARC_MODULE_CLASS_NAME`.
        """
        targets = cls.iter_dgx_quarc_packet_targets()
        if not targets:
            return []
        try:
            from .quarc_live_module import (
                attach_pool_targets_from_quarc_module,
                build_quarc_base_module_from_specs,
                isolated_quarc_stream_id_counters,
                specs_from_quarc_packet_targets,
            )
        except ImportError as exc:
            raise ImportError(
                "ParameterPool.prepare_dgx_quarc_hybrid_packets requires the `quarc` package."
            ) from exc
        with isolated_quarc_stream_id_counters():
            specs = specs_from_quarc_packet_targets(
                targets,
                module_class_name=quarc_module_class_name,
            )
            module = build_quarc_base_module_from_specs(specs)
            attach_pool_targets_from_quarc_module(module, targets)
        return targets

    @classmethod
    def get_all_ids(cls) -> List[int]:
        """
        Get all the IDs.

        Returns:
            List[int]: A list of all the IDs.
        """
        return list(cls._parameters_dict.keys())

    @classmethod
    def get_all_objs(cls) -> List[ParameterTable | Parameter]:
        """
        Get all the objects.

        Returns:
            List[Any]: A list of all the objects.
        """
        return list(cls._parameters_dict.values())

    @classmethod
    def iter_dgx_parameter_tables(cls) -> List["ParameterTable"]:
        """
        Return every registered :class:`~qiskit_qm_provider.parameter_table.ParameterTable`
        with ``input_type == DGX_Q``, sorted by pool ``_id`` (stable order for Quarc codegen).

        Used to drive Quarc module synthesis from the full pool snapshot (not only policy/reward).
        """
        from .input_type import InputType
        from .parameter_table import ParameterTable

        tables: List[ParameterTable] = []
        for obj in cls.get_all_objs():
            if isinstance(obj, ParameterTable) and obj.input_type == InputType.DGX_Q:
                tables.append(obj)
        tables.sort(key=lambda t: t._id)
        return tables

    @classmethod
    def get_all(cls) -> Dict[int, ParameterTable | Parameter]:
        """
        Get all the IDs and the associated objects.

        Returns:
            Dict[int, Any]: A dictionary containing the IDs and the associated objects.
        """
        return cls._parameters_dict

    def __getitem__(self, id: int) -> ParameterTable | Parameter:
        """
        Get the object associated with the given ID.

        Args:
            id (int): The ID of the object.

        Returns:
            obj: The object associated with the ID.
        """
        return self.get_obj(id)

    def __setitem__(self, id: int, obj: ParameterTable | Parameter):
        """
        Set the object associated with the given ID.

        Args:
            id (int): The ID of the object.
            obj: The object to be associated with the ID.
        """
        if id in self._parameters_dict:
            raise ValueError(f"Parameter with ID {id} already exists.")
        self._parameters_dict[id] = obj

    def __delitem__(self, id: int):
        """
        Delete the object associated with the given ID.

        Args:
            id (int): The ID of the object.
        """
        if id in self._parameters_dict:
            del self._parameters_dict[id]
        else:
            raise KeyError(f"Parameter with ID {id} does not exist.")

    def __contains__(self, id: int) -> bool:
        """
        Check if the object associated with the given ID exists.

        Args:
            id (int): The ID of the object.

        Returns:
            bool: True if the object exists, False otherwise.
        """
        return id in self._parameters_dict

    def __len__(self) -> int:
        """
        Get the number of objects in the pool.

        Returns:
            int: The number of objects in the pool.
        """
        return len(self._parameters_dict)

    def __iter__(self):
        """
        Iterate over the objects in the pool.

        Returns:
            Iterator: An iterator over the objects in the pool.
        """
        return iter(self._parameters_dict.values())

    def __str__(self) -> str:
        """
        Get a string representation of the pool.

        Returns:
            str: A string representation of the pool.
        """
        return str(self._parameters_dict)

    def __repr__(self) -> str:
        """
        Get a string representation of the pool.

        Returns:
            str: A string representation of the pool.
        """
        return f"ParameterPool({self._parameters_dict})"