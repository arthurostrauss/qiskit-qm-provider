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

"""QM estimator job: runs estimator PUBs as QUA programs and returns expectation values.

Author: Arthur Strauss
Date: 2026-02-08
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.backend_estimator_v2 import _pauli_expval_with_variance
from qiskit.primitives.base.base_primitive_job import ResultT
from qiskit.primitives.containers import DataBin, BitArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers import PubResult
from qiskit.result import Counts
from qm.jobs.running_qm_job import RunningQmJob
from typing import Optional, Union, List, Dict, Tuple, TYPE_CHECKING
from ..backend import QMBackend
from ..parameter_table import InputType, ParameterPool
from .iqcc_job_mixin import IQCCJobMixin
from .primitive_plans import EstimatorPlan
from .qm_execution_options import (
    await_running_jobs,
    join_job_ids,
    submit_qua_programs,
)
from .qua_programs import plan_estimator_programs, compute_locator
from .qm_primitive_job import QMPrimitiveJob
from ..primitives.qm_estimator import QMEstimatorOptions
import os
import inspect
import tempfile

if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob


def counts_from_estimator_stream(plan: EstimatorPlan, raw) -> List[Counts]:
    """Build per-metadata :class:`~qiskit.result.Counts` from the ``__c`` stream."""
    data = np.asarray(raw, dtype=int)
    shots = plan.shots
    num_qubits = plan.num_qubits
    unit = shots * num_qubits
    if data.size % unit != 0:
        raise ValueError(
            f"Estimator stream size {data.size} is not a multiple of "
            f"shots×num_qubits ({shots}×{num_qubits}={unit})."
        )
    stream_count = data.size // unit
    if stream_count != len(plan.metadata):
        raise ValueError(f"Estimator stream has {stream_count} buffers but expected {len(plan.metadata)}.")

    # Buffer k is metadata task k. Column j is __c[j], i.e. clbit j: little-endian, as Pauli
    # labels and counts expect.
    bit_array = BitArray.from_bool_array(data.reshape(stream_count, shots, num_qubits).astype(bool), order="little")
    return [Counts(bit_array.get_counts(loc=(k,))) for k in range(stream_count)]


def estimator_pub_result(plan: EstimatorPlan, raw) -> PubResult:
    """Turn one plan's ``__c`` stream into its :class:`~qiskit.primitives.PubResult`.

    As in Qiskit's ``BackendEstimatorV2``, both the parameter indices and the observables are
    broadcast to the pub shape before post-processing. A plan without tasks has no stream
    (``raw`` is ignored): its identity terms are evaluated exactly.
    """
    counts_list = counts_from_estimator_stream(plan, raw) if plan.has_tasks else []
    expval_map = QMEstimatorJob._calc_expval_map(counts_list, plan.metadata)
    bc_param_ind, bc_obs = np.broadcast_arrays(plan.param_indices, plan.pub.observables)
    return QMEstimatorJob._postprocess_pub(plan.pub, expval_map, bc_param_ind, bc_obs, plan.shots)


class QMEstimatorJob(QMPrimitiveJob):
    """Job handle for :class:`~qiskit_qm_provider.primitives.QMEstimatorV2` execution.

    Builds a QUA estimator program from pubs and returns expectation values via
    :meth:`result`. See :attr:`programs` for the compiled QUA source and
    :attr:`result_handles` for raw QM stream access after submit.
    """

    def result(self) -> ResultT:
        """Build and return primitive estimator results from QM streaming data.

        Returns:
            :class:`~qiskit.primitives.PrimitiveResult` with per-pub expectation
            values and standard errors.
        """
        if self._qm_jobs is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._result_function(self._qm_jobs)

    @property
    def runtime_pubs(self) -> List[EstimatorPlan]:
        """Execution plans for this estimator job.

        One :class:`~qiskit_qm_provider.job.primitive_plans.EstimatorPlan` per input PUB. Each
        plan holds the measurement grouping metadata, the parameter tables and the packed
        observable rows that drive the per-qubit basis switches. Inspect it to see how
        observables were grouped or what is streamed per binding.
        """
        return self._plans

    def __init__(
        self,
        backend: QMBackend,
        pubs: List[EstimatorPub],
        input_type: Optional[InputType],
        switch_obs_circuit: QuantumCircuit,
        **kwargs,
    ):
        """Create an estimator job.

        Args:
            backend: Backend that compiled the circuits.
            pubs: Coerced estimator pubs to execute.
            input_type: How circuit parameters are streamed to the OPX.
            switch_obs_circuit: Pre-transpiled circuit used to rotate observables.
            **kwargs: Estimator options forwarded to execution planning.
        """
        super().__init__(backend, pubs, input_type, **kwargs)
        ParameterPool.reset()
        options = QMEstimatorOptions(input_type=input_type, **kwargs)
        self._plans: List[EstimatorPlan] = [
            EstimatorPlan.from_pub(pub, options=options, plan_index=i) for i, pub in enumerate(pubs)
        ]
        self._switch_obs_circuit: QuantumCircuit = switch_obs_circuit
        programs, self._chunk_layout = plan_estimator_programs(backend, self._plans)
        self._programs = programs
        self._locator = compute_locator(self._chunk_layout)

    @property
    def _run_options(self) -> Dict:
        """Backend run options (``simulate``, ``compiler_options``, ...), which the estimator
        nests under ``run_options``, flattened with the top-level metadata."""
        nested = self.metadata.get("run_options") or {}
        return {**nested, **{k: v for k, v in self.metadata.items() if k != "run_options"}}

    def submit(self):
        """Submit the job to the backend.

        Programs are submitted via :func:`~.submit_qua_programs` (``qm.execute``
        for simulation; OPX1000 ``add_to_queue`` / OPX+ ``queue.add`` for real
        hardware). Each chunk is waited on until running, then plan data is
        streamed. IQCC sync-hook submission is handled by
        :class:`IQCCEstimatorJob`.
        """
        if self._qm_jobs is not None:
            raise RuntimeError("Job has already been submitted.")

        pending_jobs = submit_qua_programs(
            self._backend.qm,
            self._backend.qmm,
            self._programs,
            self._run_options,
        )
        self._job_id = join_job_ids(pending_jobs)
        self._qm_jobs = await_running_jobs(
            pending_jobs, self._chunk_layout, entity="PUB indices"
        )
        for job, chunk in zip(self._qm_jobs, self._chunk_layout):
            for global_idx in chunk:
                self._plans[global_idx].push(job, self._backend.qm)

    @staticmethod
    def _calc_expval_map(
        counts: List[Counts],
        metadata: List[Dict],
    ) -> Dict[Tuple[Tuple[int, ...], str], Tuple[float, float]]:
        """Computes the map of expectation values from Counts objects.

        Args:
            counts: List of Counts objects, one per task
            metadata: List of metadata dicts, one per task

        Returns:
            The map of expectation values takes a pair of an index of the bindings array and
            a pauli string as a key and returns the expectation value and variance of the pauli string.
        """
        expval_map: Dict[Tuple[Tuple[int, ...], str], Tuple[float, float]] = {}

        for task_idx, meta in enumerate(metadata):
            orig_paulis = meta["orig_paulis"]
            meas_paulis = meta["meas_paulis"]
            param_index = meta["param_index"]

            # Get counts for this task
            count = counts[task_idx]

            # Compute expectation values using the same method as backend_estimator_v2
            expvals, variances = _pauli_expval_with_variance(count, meas_paulis)

            # Map back to original Paulis
            # orig_paulis can be a SparsePauliOp (with coefficients) or PauliList
            # meas_paulis and orig_paulis should be in the same order
            if isinstance(orig_paulis, SparsePauliOp):
                # orig_paulis is a SparsePauliOp, iterate over its paulis
                orig_paulis_list = orig_paulis.paulis
            elif isinstance(orig_paulis, PauliList):
                # orig_paulis is a PauliList
                orig_paulis_list = orig_paulis
            else:
                # Fallback: treat as single Pauli
                if isinstance(orig_paulis, Pauli):
                    orig_paulis_list = PauliList([orig_paulis])
                else:
                    # Try to convert to PauliList
                    orig_paulis_list = PauliList([Pauli(str(orig_paulis))])

            # Map expectation values: meas_paulis and orig_paulis_list should be in same order
            if len(meas_paulis) == len(orig_paulis_list):
                # Direct 1-to-1 mapping
                for orig_pauli, expval, variance in zip(orig_paulis_list, expvals, variances):
                    expval_map[param_index, orig_pauli.to_label()] = (
                        float(expval),
                        float(variance),
                    )
            else:
                # Length mismatch - this shouldn't happen normally, but handle it
                # Use first expval for all orig_paulis (grouped measurement case)
                for orig_pauli in orig_paulis_list:
                    expval_map[param_index, orig_pauli.to_label()] = (
                        float(expvals[0]),
                        float(variances[0]),
                    )

        return expval_map

    @staticmethod
    def _postprocess_pub(
        pub: EstimatorPub,
        expval_map: Dict,
        param_indices: np.ndarray,
        observables: np.ndarray,
        shots: int,
    ) -> PubResult:
        """Computes expectation values (evs) and standard errors (stds).

        The values are stored in arrays broadcast to the shape of the pub.

        Args:
            pub: The pub to postprocess.
            expval_map: The map of expectation values.
            param_indices: The parameter indices array.
            observables: The observables array broadcast to pub shape.
            shots: The number of shots.

        Returns:
            The pub result.
        """
        bc_param_ind = param_indices
        bc_obs = observables
        evs = np.zeros_like(bc_param_ind, dtype=float)
        variances = np.zeros_like(bc_param_ind, dtype=float)

        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            for pauli, coeff in bc_obs[index].items():
                if not pauli.strip("I"):  # <I> = 1 exactly, with no variance: never measured
                    evs[index] += coeff
                    continue
                expval, variance = expval_map[param_index, pauli]
                evs[index] += expval * coeff
                variances[index] += np.abs(coeff) * variance**0.5

        stds = variances / np.sqrt(shots)
        data_bin = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data_bin,
            metadata={
                "target_precision": pub.precision,
                "shots": shots,
                "circuit_metadata": pub.circuit.metadata,
            },
        )

    def _result_function(self, qm_jobs: List[RunningQmJob]) -> PrimitiveResult[PubResult]:
        results_handles = [job.result_handles for job in qm_jobs]
        for handle in results_handles:
            handle.wait_for_all_values()

        pub_results = []
        for i, plan in enumerate(self._plans):
            raw = None
            if plan.has_tasks:  # plans without tasks were never run (see plan_estimator_programs)
                chunk_idx, local_idx = self._locator[i]
                raw = results_handles[chunk_idx].get(f"__c_{local_idx}").fetch_all()
            pub_results.append(estimator_pub_result(plan, raw))

        return PrimitiveResult(pub_results, metadata={"version": 2})

    def _run(self):
        pass

    def _parse_result(self):
        pass


class IQCCEstimatorJob(IQCCJobMixin, QMEstimatorJob):
    """IQCC cloud variant of :class:`QMEstimatorJob`.

    Execution is **synchronous**: each :meth:`submit` call blocks until the
    remote OPX program completes.  ``CloudJob.status`` is always
    ``"completed"``; real failure information lives in ``_run_data["stderr"]``
    (see :class:`~.IQCCJobMixin` and :meth:`~.IQCCJobMixin.status`).

    ``result()`` raises :class:`~.IQCCCloudExecutionError` before attempting to
    fetch streams when any chunk job's stderr contains a Python traceback.
    """

    def submit(self):
        """Submit all QUA programs to the IQCC cloud backend.

        For each chunk produced by :func:`~.plan_estimator_programs`, a separate
        cloud job is executed synchronously.  When any of the chunk's plans streams
        inputs (parameters or observables), a sync-hook script is written to a
        temporary file and passed to ``execute()``; the file is unlinked
        immediately after the call regardless of outcome.

        Results from all chunks are stitched back by the inherited
        :meth:`_result_function` using the locator built at construction time.
        """
        from .sync_hook import generate_sync_hook

        if self._qm_jobs is not None:
            raise RuntimeError("IQCC QM job has already been submitted")

        programs = self._programs
        timeout = self._run_options.get("timeout")
        self._qm_jobs = []

        for prog, chunk in zip(programs, self._chunk_layout):
            chunk_plans = [self._plans[g] for g in chunk]

            sync_hook_path = None
            if any(p.streams_inputs for p in chunk_plans):
                sync_hook_code = generate_sync_hook(chunk_plans)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(sync_hook_code)
                    sync_hook_path = f.name
                options = {"sync_hook": sync_hook_path}
            else:
                options = {}

            if timeout is not None:
                options["timeout"] = timeout

            try:
                self._qm_jobs.append(self._backend.qm.execute(prog, options=options))  # type: ignore
            finally:
                if sync_hook_path is not None:
                    os.unlink(sync_hook_path)

        self._job_id = join_job_ids(self._qm_jobs)
