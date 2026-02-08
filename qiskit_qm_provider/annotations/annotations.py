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

"""QMSaveAnnotation and OpenQASM3 serialization for save/stream semantics in circuits.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from qiskit.circuit.annotation import Annotation, OpenQASM3Serializer


@dataclass(eq=True)
class QMSaveAnnotation(Annotation):
    """
    Annotation indicating that classical outcomes within a Box should be saved
    for offline decoding, without constraining real-time reuse of the same
    classical bits within the circuit.

    Semantics
    ---------
    - When attached to a `BoxOp`, this signals the backend/QPU compiler to
      materialize stream-processing saves for the classical entities used in the
      box.
    - If `mode == "register"`, the intent is to aggregate per `ClassicalRegister`
      into an integer (bitstring-as-integer) and save one integer per involved
      register.
    - If `mode == "bit"`, the intent is to save each `Clbit` individually.

    The annotation itself does not carry references to Qiskit objects (to keep
    it serializable and backend-agnostic). Instead, it optionally declares
    symbolic names that the compiler may bind to concrete classical registers or
    bits discovered within the Box scope.
    """

    namespace: str = "qm.save"

    # Either "register" (packed integer per ClassicalRegister) or "bit".
    mode: str = "register"

    # Optional symbolic labels to scope which classical entities to save. If
    # empty/None, the intent applies to all classical entities used within the Box.
    # For mode=="register", entries represent register names; for mode=="bit",
    # entries represent bit names (e.g. "c[3]").
    labels: Optional[Tuple[str, ...]] = None

    # Optional user label for downstream identification of the saved stream(s).
    tag: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in ("register", "bit"):
            raise ValueError("QMSaveAnnotation.mode must be 'register' or 'bit'")
        if self.labels is not None and not isinstance(self.labels, tuple):
            # Store immutably for safe equality and serialization
            self.labels = tuple(self.labels)  # type: ignore[assignment]


class QMSaveOpenQASM3Serializer(OpenQASM3Serializer):
    """
    OpenQASM 3 serializer for `QMSaveAnnotation`.

    Payload format (single line JSON-like without spaces for compactness):
        qm.save:{"m":"register"|"bit","ls":[<str>...],"t":<str|null>}

    Unknown namespaces or payloads must return NotImplemented to allow other
    handlers to take over.
    """

    def dump(self, annotation: Annotation):
        if not isinstance(annotation, QMSaveAnnotation):
            return NotImplemented
        # Minimal, stable serialization
        # Avoid importing json to keep payload short and deterministic
        mode = annotation.mode
        labels = annotation.labels or ()
        tag = annotation.tag
        # Build a compact representation
        # example: qm.save:{"m":"register","ls":["creg"],"t":"shot"}
        esc = lambda s: s.replace("\\", "\\\\").replace("\"", "\\\"")
        labels_part = ",".join(f"\"{esc(x)}\"" for x in labels)
        tag_part = f"\"{esc(tag)}\"" if tag is not None else "null"
        payload = (
            f'{annotation.namespace}:{{"m":"{esc(mode)}","ls":[{labels_part}],"t":{tag_part}}}'
        )
        return payload

    def load(self, namespace: str, payload: str):
        if namespace != "qm.save":
            return NotImplemented
        # Expect the dumped format; tolerate surrounding namespace prefix.
        # The incoming payload may either be the full string returned by dump()
        # or only the right-hand JSON object depending on upstream emitter.
        try:
            right = payload
            if right.startswith("qm.save:"):
                right = right[len("qm.save:"):]
            # Simple hand-rolled parse for the limited schema
            # Format: {"m":"<mode>","ls":["a","b"],"t":null|"..."}
            # We'll use json if available; fall back to eval-safe literal.
            import json  # local import

            data = json.loads(right)
            mode = data.get("m", "register")
            labels = tuple(data.get("ls") or [])
            tag = data.get("t", None)
            if tag is None:
                tag_val = None
            else:
                tag_val = str(tag)
            return QMSaveAnnotation(mode=mode, labels=labels or None, tag=tag_val)
        except Exception:
            return NotImplemented
