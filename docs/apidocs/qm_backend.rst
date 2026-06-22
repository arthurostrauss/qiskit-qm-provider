.. _qm-backend:

Backend and utilities
=====================

See the :doc:`Backend & Utilities guide </backend>` for execution modes and hybrid embedding.
For the measurement-output locality model, see :doc:`Measurement outputs </measurement_outputs>`.

.. currentmodule:: qiskit_qm_provider.backend

.. rubric:: Backends

.. autosummary::
   :toctree: stubs/

   QMBackend
   FluxTunableTransmonBackend
   QMInstructionProperties

.. rubric:: Circuit compilation and measurement outputs

.. currentmodule:: qiskit_qm_provider.backend.qua_circuit_compilation

.. autosummary::
   :toctree: stubs/

   QuaCircuitCompilation
   MeasurementOutcomeTable

.. currentmodule:: qiskit_qm_provider.backend.measurement_field

.. autosummary::
   :toctree: stubs/

   MeasurementRegisterField

.. currentmodule:: qiskit_qm_provider.backend.backend_utils

.. rubric:: Functions

.. autosummary::
   :toctree: stubs/

   add_basic_macros
   assign_struct_with_table
   get_measurement_outcomes
   get_qua_script
   dump_qua_script
   pack_register_to_int

.. automodule:: qiskit_qm_provider.backend
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit_qm_provider.backend.qua_circuit_compilation
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit_qm_provider.backend.measurement_field
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit_qm_provider.backend.backend_utils
   :no-members:
   :no-inherited-members:
   :no-special-members:
