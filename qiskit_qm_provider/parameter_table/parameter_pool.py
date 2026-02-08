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
            if hasattr(value, "reset"):
                value.reset()
        cls._parameters_dict.clear()
        cls._configured = False
        cls._patched = False

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

    @classmethod
    def patch_opnic_wrapper(
        cls, path_to_python_wrapper: Union[str, Path], force_recompile_python_wrapper: bool = False
    ):
        """
        Patch the OPNIC wrapper.

        Args:
            path_to_opnic_dev (Optional[str]): The path to the OPNIC development directory
        """
        from .opnic_utils import patch_opnic_wrapper
        from .parameter_table import ParameterTable

        def check_function(param_table: ParameterTable | Parameter) -> bool:
            from .input_type import InputType
            return (
                (param_table._usable_for_dgx_communication
                if isinstance(param_table, ParameterTable)
                else param_table.is_standalone()) and param_table.input_type == InputType.DGX_Q
            )

        param_tables = list(filter(check_function, cls.get_all_objs()))
        patch_opnic_wrapper(param_tables, path_to_python_wrapper, force_recompile_python_wrapper)
        cls._patched = True

    @classmethod
    def configure_stream(
        cls,
        path_to_python_wrapper: Union[str, Path],
    ):
        # from opnic_python.opnic_wrapper import configure_stream
        # from opnic_python.opnic_wrapper import Direction_INCOMING, Direction_OUTGOING
        if "opnic_wrapper" not in sys.modules:
            sys.path.append(str(path_to_python_wrapper))
        from opnic_wrapper import (
            Direction_INCOMING,
            Direction_OUTGOING,
            configure_stream,
        )
        from .input_type import Direction

        for obj in cls.get_all_objs():
            direction = Direction_INCOMING if obj.direction == Direction.INCOMING else Direction_OUTGOING
            configure_stream(obj.stream_id, direction)
        cls._configured = True

    @classmethod
    def initialize_streams(cls, path_to_python_wrapper: Union[str, Path]):
        """
        Initialize the OPNIC and the necessary streams for the current stage of the ParameterPool.
        Args:
            path_to_python_wrapper: The path to the Python wrapper.

        Returns:

        """
        cls.patch_opnic_wrapper(path_to_python_wrapper)
        cls.configure_stream(path_to_python_wrapper)

    @classmethod
    def opx_handshake(cls):
        """
        Perform the OPNIC handshake.
        """
        from opnic_wrapper import opx_handshake

        opx_handshake()

    @classmethod
    def patched(cls) -> bool:
        return cls._patched

    @classmethod
    def configured(cls) -> bool:
        return cls._configured

    @classmethod
    def close_streams(cls):
        """
        Close the streams.
        """
        if cls._configured and cls._patched:
            from opnic_wrapper import close_stream

            for obj in cls.get_all_objs():
                close_stream(obj.stream_id)
            cls._configured = False
            cls._patched = False
        else:
            raise ValueError("The streams are not configured or patched.")
