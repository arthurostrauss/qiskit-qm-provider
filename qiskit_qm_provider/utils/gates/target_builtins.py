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

"""Operations every QM backend Target exposes, and OpenQASM 3 keyword names.

Author: Arthur Strauss
Date: 2026-10-29
"""

from __future__ import annotations

import warnings
from typing import Dict, Type

from qiskit.circuit import Barrier, Store
from qiskit.circuit.controlflow import (
    ControlFlowOp,
    ForLoopOp,
    IfElseOp,
    SwitchCaseOp,
    WhileLoopOp,
)

try:
    from qiskit.circuit.controlflow import get_control_flow_name_mapping

    control_flow_name_mapping = get_control_flow_name_mapping()
except ImportError:
    warnings.warn(
        "get_control_flow_name_mapping is not available in this version of Qiskit, skipping it from control flow mapping."
    )
    control_flow_name_mapping: Dict[str, Type[ControlFlowOp]] = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }

# Variable-width instructions every QM backend Target exposes as global operations, the
# way control flow is exposed. They are lowered by qm-qasm itself, not through the QUA
# operations mapping: ``barrier`` becomes a QUA ``align`` over the elements of the barrier
# qubits, and ``store`` becomes a QUA assignment.
variadic_builtin_instructions: Dict[str, type] = {
    "barrier": Barrier,
    "store": Store,
}

qasm3_keyword_instructions = (
    "barrier",
    "store",
    "measure",
    "reset",
    "delay",
    "nop",
    "box",
    "for_loop",
    "while_loop",
    "if_else",
    "switch_case",
)
_QASM3_DUMP_LOOSE_BIT_PREFIX = "_bit"
