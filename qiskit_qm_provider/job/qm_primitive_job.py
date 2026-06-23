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

"""Base class for QM primitive jobs (sampler/estimator) executing QUA programs from PUBs.

Author: Arthur Strauss
Date: 2026-02-08
"""

from abc import ABC
from typing import Optional, Union, List, Dict, Any
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import JobStatus
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from qm import Program
from ..backend import QMBackend
from ..parameter_table import InputType
from .iqcc_job_mixin import result_handles_from_qm_job

Pub = Union[SamplerPub, EstimatorPub]


class QMPrimitiveJob(BasePrimitiveJob, ABC):
    """Base class for :class:`QMSamplerJob` and :class:`QMEstimatorJob`.

    Primitive jobs compile PUBs into QUA program(s) at construction time.
    Inspect them with ``qm.generate_qua_script(job.programs[0])`` or iterate
    ``job.programs`` for chunked execution.

    Attributes:
        programs: List of compiled :class:`qm.Program` objects; length 1 when
            no chunking occurred.
        pubs: PUB list passed to ``run()``.
        inputs: Dict snapshot of pubs, ``input_type``, and ``metadata``.
        backend: :class:`~qiskit_qm_provider.backend.qm_backend.QMBackend` used for
            compilation and execution.
        job_id: QM SDK job id (set on ``submit()``).
        metadata: Options forwarded from the primitive / ``run()`` call.

    Use :attr:`qm_job` and :attr:`result_handles` for the live QM SDK handle and
    measurement streams after submission.
    """

    def __init__(self, backend: QMBackend, pubs: List[Pub], input_type: InputType, **kwargs):
        super().__init__(job_id="pending", **kwargs)
        self._backend = backend
        self._pubs = pubs
        self._input_type = input_type
        self._qm_job: Optional[Union[RunningQmJob, QmPendingJob, List[QmPendingJob]]] = None
        self._programs: Optional[List[Program]] = None

    def status(self) -> JobStatus:
        """Return the job status.

        When the job was split into multiple QUA programs (chunked execution),
        returns the least-advanced status across all chunk jobs.  The aggregate
        is ``DONE`` only when every chunk has completed.
        """
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        mapping = {
            "unknown": JobStatus.ERROR,
            "pending": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "completed": JobStatus.DONE,
            "canceled": JobStatus.CANCELLED,
            "loading": JobStatus.VALIDATING,
            "error": JobStatus.ERROR,
        }
        if isinstance(self._qm_job, list):
            statuses = [mapping.get(getattr(j, "status", "unknown"), JobStatus.ERROR) for j in self._qm_job]
            for state in (JobStatus.ERROR, JobStatus.CANCELLED, JobStatus.VALIDATING, JobStatus.QUEUED, JobStatus.RUNNING):
                if state in statuses:
                    return state
            return JobStatus.DONE
        return mapping.get(self._qm_job.status, JobStatus.ERROR)

    def done(self) -> bool:
        """Return whether the job has successfully run."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Return whether the job is actively running."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Return whether the job has been cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
        return self.status() in [JobStatus.DONE, JobStatus.ERROR]

    def cancel(self):
        """Attempt to cancel the job.  Cancels all chunk jobs for chunked execution."""
        if self._qm_job is None:
            raise RuntimeError("QM job is not running")
        if isinstance(self._qm_job, list):
            return all(j.cancel() for j in self._qm_job)
        return self._qm_job.cancel()

    @property
    def qm_job(self) -> Optional[Union[RunningQmJob, List[QmPendingJob]]]:
        """Underlying QM SDK job after :meth:`submit`.

        Exposes ``result_handles``, ``cancel``, ``push_to_input_stream``, and other
        runtime APIs. Raises if the job has not been submitted yet.
        """
        return self._qm_job

    @property
    def result_handles(self) -> Any:
        """QM SDK result stream handles after :meth:`submit`.

        Returns ``qm_job.result_handles``. Raises if the job has not been
        submitted yet.
        """
        return result_handles_from_qm_job(self._qm_job)

    @property
    def inputs(self) -> Dict:
        """Job input parameters.

        Returns:
            Input parameters used in this job.
        """

        return {
            "pubs": [(pub.circuit, pub.parameter_values, pub.shots) for pub in self._pubs],
            "input_type": self._input_type,
            "metadata": self.metadata,
        }

    @property
    def programs(self) -> Optional[List[Program]]:
        """Compiled QUA programs for this job.

        Always a list; length 1 when no chunking occurred.  Available
        immediately after job construction.  Print with::

            from qm import generate_qua_script
            for prog in job.programs:
                print(generate_qua_script(prog))
        """
        return self._programs

    @property
    def pubs(self) -> List[Pub]:
        """Return the PUBs used in this job."""
        return self._pubs
