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

"""QMJob: Qiskit JobV1 implementation for Quantum Machines program execution.

Author: Arthur Strauss
Date: 2026-02-08
"""

from __future__ import annotations

from typing import Optional, List, Callable, Dict, Union, TYPE_CHECKING

from copy import deepcopy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BitArray, SamplerPubResult, DataBin
from qiskit.providers.job import JobV1, JobStatus
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData, MeasLevel, MeasReturnType
from qm import QuantumMachine, Program, SimulationConfig, StreamingResultFetcher, QuantumMachinesManager
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.pending_job import QmPendingJob

from qiskit_qm_provider.backend.qm_backend import QMBackend
from qiskit_qm_provider.backend.backend_utils import validate_circuits, _QASM3_DUMP_LOOSE_BIT_PREFIX

if TYPE_CHECKING:
    from iqcc_cloud_client.qmm_cloud import CloudJob
    from iqcc_cloud_client import IQCC_Cloud


try:
    # Optional Qiskit Pulse import – mirrors backend behaviour but kept local
    from qiskit.pulse import DriveChannel  # type: ignore[unused-import]

    _QISKIT_PULSE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without Pulse
    _QISKIT_PULSE_AVAILABLE = False


class QMJob(JobV1):
    """QMJob class for Quantum Machines."""

    def __init__(
        self,
        backend: QMBackend,
        job_id: str,
        qm: Union[QuantumMachine, IQCC_Cloud],
        program: Program,
        result_function: Callable[[RunningQmJob], Result],
        **kwargs,
    ):
        JobV1.__init__(self, backend, job_id, **kwargs)
        self.qm = qm
        self._qm_job: Optional[RunningQmJob | QmPendingJob | List[QmPendingJob]] = None
        self.program = program
        self._result_function = result_function

    # ------------------------------------------------------------------
    # High-level constructors used by QMBackend.run
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result_function(
        backend: QMBackend,
        num_circuits: int,
        num_shots: int,
        circuits: List[QuantumCircuit],
        cregs_dicts: List[Dict[str, int]],
        meas_level: MeasLevel,
        meas_return: MeasReturnType,
        memory: bool,
    ) -> Callable[[RunningQmJob | List[RunningQmJob] | "CloudJob" | Dict], Result]:
        """Create a Result-building callback for standard circuit execution.

        This function encapsulates the data plumbing that was previously
        implemented as an inner closure in ``QMBackend.run``.
        """

        def result_function(
            qm_job: RunningQmJob | List[RunningQmJob] | "CloudJob" | Dict,
        ) -> Result:
            from iqcc_cloud_client.qmm_cloud import CloudJob, CloudResultHandles  # type: ignore[import]

            is_job_list = isinstance(qm_job, list)
            if is_job_list:
                results_handle = [job.result_handles for job in qm_job]  # type: ignore[attr-defined]
                for handle in results_handle:
                    handle.wait_for_all_values()
            elif isinstance(qm_job, (RunningQmJob, CloudJob)):
                results_handle = qm_job.result_handles
                results_handle.wait_for_all_values()
            else:
                # Fallback path for IQCC-style dictionary response
                results_handle = qm_job["result"]

            all_data: List[SamplerPubResult] = []
            for i in range(num_circuits):
                qc_meas_data = {}
                for creg, creg_size in cregs_dicts[i].items():
                    if is_job_list:
                        data = (
                            np.array(results_handle[i].get(f"{creg}_{i}").fetch_all())  # type: ignore[index]
                            .flatten()
                            .tolist()
                        )
                    elif isinstance(results_handle, (StreamingResultFetcher, CloudResultHandles)):
                        data = (
                            np.array(results_handle.get(f"{creg}_{i}").fetch_all())  # type: ignore[index]
                            .flatten()
                            .tolist()
                        )
                    else:
                        data = (
                            np.array(results_handle.get(f"{creg}_{i}"))  # type: ignore[index]
                            .flatten()
                            .tolist()
                        )

                    if meas_level == MeasLevel.CLASSIFIED:
                        bit_array = BitArray.from_samples(data, creg_size)
                        qc_meas_data[creg] = bit_array
                    elif meas_level == MeasLevel.KERNELED:
                        if meas_return == MeasReturnType.SINGLE:
                            qc_meas_data[creg] = np.array(
                                [d[0] + 1j * d[1] for d in data],
                                dtype=complex,
                            )
                        elif meas_return == MeasReturnType.AVERAGE:
                            qc_meas_data[creg] = np.mean(
                                [d[0] + 1j * d[1] for d in data],
                            )

                sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
                all_data.append(sampler_data.join_data())

            experiment_data = []
            for i, data in enumerate(all_data):
                # Attach circuit-level metadata to the result header so that
                # qiskit-experiments can recover `datum["metadata"]` for curve analysis.
                # In particular, experiments such as T1 expect
                #     circuit.metadata = {"xval": ...}
                # and look for this via ExperimentResult.header.metadata.
                circuit_metadata = getattr(circuits[i], "metadata", {}) or {}
                experiment_result = ExperimentResult(
                    shots=num_shots,
                    success=True,
                    data=ExperimentResultData(
                        data.get_counts(),
                        memory=data.get_bitstrings() if memory else None,
                    ),
                    meas_level=meas_level,
                    meas_return=meas_return,
                    header={"metadata": circuit_metadata},
                    status=getattr(qm_job, "status", "done"),
                )
                experiment_data.append(experiment_result)

            result = Result(
                results=experiment_data if num_circuits > 1 else experiment_data[0],
                backend_name=backend.name,
                job_id=getattr(qm_job, "id", "unknown"),
                backend_version=2,
                qobj_id=None,
                success=True,
            )
            return result

        return result_function

    @classmethod
    def from_circuits(
        cls,
        backend: QMBackend,
        run_input: QuantumCircuit | List[QuantumCircuit],
        **options,
    ) -> "QMJob":
        """Factory that mirrors the original ``QMBackend.run`` logic.

        This method performs:
        - circuit validation and optional reset insertion,
        - target / calibration updates,
        - QUA program generation via ``get_run_program``,
        - result object construction from streamed data,
        - and submission of either a local ``QMJob`` or cloud ``IQCCJob``.
        """
        from iqcc_cloud_client.qmm_cloud import CloudQuantumMachinesManager  # type: ignore[import]
        from .qua_programs import get_run_program

        # Merge explicit options into backend defaults (preserving current behaviour)
        options_ = deepcopy(backend.options.__dict__)
        options_.update(options)

        num_shots = options.get("shots", backend.options.shots)
        skip_reset = options.get("skip_reset", backend.options.skip_reset)
        memory = options.get("memory", backend.options.memory)
        meas_level = options.get("meas_level", backend.options.meas_level)
        meas_return = options.get("meas_return", backend.options.meas_return)

        if not isinstance(run_input, list):
            run_input = [run_input]

        new_circuits = validate_circuits(run_input, should_reset=not skip_reset, check_for_params=True)
        num_circuits = len(new_circuits)

        # Synchronize backend target and (optionally) pulse calibrations
        backend.update_target()
        if _QISKIT_PULSE_AVAILABLE:
            for qc in new_circuits:
                backend.update_calibrations(qc)

        # Build the QUA program the QM will execute
        run_program = get_run_program(backend, num_shots, new_circuits)
        qm = backend.qm

        job_id = "pending"
        cregs_dicts: List[Dict[str, int]] = [{creg.name: creg.size for creg in qc.cregs} for qc in new_circuits]
        for i, qc in enumerate(new_circuits):
            solo_bits = [bit for bit in qc.clbits if len(qc.find_bit(bit).registers) == 0]
            if len(solo_bits) > 0:
                cregs_dicts[i][_QASM3_DUMP_LOOSE_BIT_PREFIX] = len(solo_bits)

        result_function = cls._build_result_function(
            backend=backend,
            num_circuits=num_circuits,
            num_shots=num_shots,
            circuits=new_circuits,
            cregs_dicts=cregs_dicts,
            meas_level=meas_level,
            meas_return=meas_return,
            memory=memory,
        )

        # Decide between local QM job and IQCCCloud job
        if isinstance(backend.qmm, (QuantumMachinesManager, CloudQuantumMachinesManager)):
            job_cls: type[QMJob] = QMJob
        else:
            job_cls = IQCCJob
        print("job_cls: ", job_cls)
        job = job_cls(
            backend,
            job_id,
            qm,
            run_program,
            result_function=result_function,
            **options_,
        )

        job.submit()
        # Reset the calibration mapping as in the original QMBackend.run implementation
        backend._calibration_operation_mapping_QUA = backend._operation_mapping_QUA.copy()  # type: ignore[attr-defined]
        return job

    def status(self) -> JobStatus:
        """Return the job status."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        status = self._qm_job.status if hasattr(self._qm_job, "status") else "unknown"
        mapping = {
            "unknown": JobStatus.ERROR,
            "pending": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "completed": JobStatus.DONE,
            "canceled": JobStatus.CANCELLED,
            "loading": JobStatus.VALIDATING,
            "error": JobStatus.ERROR,
        }
        return mapping.get(status, JobStatus.ERROR)

    def submit(self):
        """Submit the job to the backend."""
        compiler_options = self.metadata.get("compiler_options", None)
        simulate = self.metadata.get("simulate", None)
        if isinstance(self.qm, QuantumMachine):
            kwargs = {
                "simulate": simulate,
                "compiler_options": compiler_options,
            }
        else: # CloudQuantumMachine
            kwargs = {
                "terminal_output": True,
            }
        if isinstance(simulate, SimulationConfig):
            self._qm_job = self.qm.simulate(self.program, simulate=simulate, compiler_options=compiler_options)
        else:
            if isinstance(self.program, list):
                self._job_id = ""
                self._qm_job = []
                for prog in self.program:
                    self._qm_job.append(self.qm.queue.add(prog, **kwargs))
                self._job_id += ",".join([job.id for job in self._qm_job])
            else:
                self._qm_job = self.qm.execute(self.program, **kwargs)
                self._job_id = self._qm_job.id if hasattr(self._qm_job, "id") else ""

    def cancel(self):
        """Cancel the job."""
        if self._qm_job is None:
            raise RuntimeError("QM job is not running")
        if isinstance(self._qm_job, list):
            for job in self._qm_job:
                job.cancel()
            return
        return self._qm_job.cancel()

    def result(self):
        """Get the job result."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")

        return self._result_function(self._qm_job)

    @property
    def qm_job(self) -> Optional[RunningQmJob | List[QmPendingJob | RunningQmJob]]:
        """Get the QM job."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        return self._qm_job


class IQCCJob(QMJob):
    """IQCC Job class for Quantum Machines."""

    def __init__(
        self,
        backend: QMBackend,
        job_id: str,
        qm: IQCC_Cloud,
        program: Program,
        result_function: Callable[[RunningQmJob], Result],
        **kwargs,
    ):
        super().__init__(backend, job_id, qm, program, result_function, **kwargs)
        self._qm_job = None

    def status(self) -> JobStatus:
        raise NotImplementedError("IQCCJob does not support status method. Use IQCC_Cloud methods to check job status.")

    def submit(self):
        """Submit the job to the IQCC backend."""
        if self._qm_job is not None:
            raise RuntimeError("IQCC job has already been submitted")
        try:
            config = self.metadata["config"]
        except KeyError:
            raise ValueError("Job metadata must contain 'config' key for IQCC job submission")

        qm: IQCC_Cloud = self.qm
        timeout = self.metadata.get("timeout", None)

        self._qm_job = qm.execute(self.program, config, options={"timeout": timeout} if timeout is not None else {})
