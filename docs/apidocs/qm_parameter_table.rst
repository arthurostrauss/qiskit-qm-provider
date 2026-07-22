.. _qm-parameter-table:

Parameter table
===============

See the :doc:`Parameter Table guide </parameter_table>` for hybrid classical–quantum data flow.
Measurement outputs (``comp.outputs``) use a separate namespace — see :doc:`Measurement outputs </measurement_outputs>`.

.. currentmodule:: qiskit_qm_provider.parameter_table

.. rubric:: Classes

.. autosummary::
   :toctree: stubs/

   ParameterTable
   Parameter
   ParameterPool
   InputType
   Direction
   QUAArray
   QUA2DArray

.. currentmodule:: qiskit_qm_provider.parameter_table._mixins

.. rubric:: Field-table mixins

.. autosummary::
   :toctree: stubs/

   QuaFieldTable
   TableFieldProtocol

.. currentmodule:: qiskit_qm_provider.parameter_table._scope

.. rubric:: QUA program scope guards

.. autosummary::
   :toctree: stubs/

   is_inside_scope
   require_qua_program
   requires_qua_program

.. automodule:: qiskit_qm_provider.parameter_table
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit_qm_provider.parameter_table._mixins
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. automodule:: qiskit_qm_provider.parameter_table._scope
   :no-members:
   :no-inherited-members:
   :no-special-members:
