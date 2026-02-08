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

"""Parameter transformer: QASM3-safe identifiers and parameter naming for QUA.

Author: Arthur Strauss
Date: 2026-02-08
"""

import re
from qiskit.circuit import Parameter

# Regular expression for a valid QASM3 identifier (letters, digits, and underscores, not starting with a digit)
_VALID_DECLARABLE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_]")

# Reserved QASM3 keywords that cannot be used as variable names
QASM3_RESERVED_KEYWORDS = {
    "if",
    "else",
    "let",
    "input",
    "output",
    "reset",
    "gate",
    "measure",
    "barrier",
    "creg",
    "qreg",
    "opaque",
    "cal",
    "def",
    "bit",
    "qubit",
}


class ParameterNameTransformer:
    def __init__(self):
        # Track used names to ensure uniqueness
        self.existing_names = set()
        self.name_counter = {}

    def transform(self, param: Parameter) -> str:
        """Transform a Qiskit Parameter object's name to a valid and unique QASM3-compatible name."""
        original_name = param.name

        # Step 1: Ensure valid identifier
        if not _VALID_DECLARABLE_IDENTIFIER.fullmatch(original_name):
            transformed_name = "_" + _INVALID_CHARS.sub("_", original_name)
        else:
            transformed_name = original_name

        # Step 2: Avoid reserved keywords
        if transformed_name in QASM3_RESERVED_KEYWORDS:
            transformed_name = f"{transformed_name}_var"

        # Step 3: Ensure uniqueness in the current scope
        if transformed_name in self.existing_names:
            count = self.name_counter.get(transformed_name, 0) + 1
            self.name_counter[transformed_name] = count
            transformed_name = f"{transformed_name}_{count}"

        # Step 4: Register the name to prevent future conflicts
        self.existing_names.add(transformed_name)
        return transformed_name
