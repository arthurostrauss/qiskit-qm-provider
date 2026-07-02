.. _qm-pulse:

Pulse integration (Qiskit 1.x)
==============================

See the :doc:`Backend & Utilities guide </backend>` for Pulse scope (gate schedules only).

Pulse measurement caveat
------------------------

Qiskit Pulse ``Measure`` / measurement instructions are **not** supported. For hybrid
readout, compile circuit-level ``measure`` gates with
:meth:`~qiskit_qm_provider.backend.QMBackend.quantum_circuit_to_qua` and access
classical outcomes via ``comp.outputs`` on the returned
:class:`~qiskit_qm_provider.backend.qua_circuit_compilation.QuaCircuitCompilation`.
See :doc:`Measurement outputs </measurement_outputs>`.

.. currentmodule:: qiskit_qm_provider.pulse

.. rubric:: Functions

.. autosummary::
   :toctree: stubs/

   schedule_to_qua_macro
   validate_schedule

.. automodule:: qiskit_qm_provider.pulse
   :no-members:
   :no-inherited-members:
   :no-special-members:
