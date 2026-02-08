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
from typing import Optional, Union, List, Dict
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import JobStatus
from qm.jobs.pending_job import QmPendingJob
from qm.jobs.running_qm_job import RunningQmJob
from qm import Program
from ..backend import QMBackend
from ..parameter_table import InputType

Pub = Union[SamplerPub, EstimatorPub]


class QMPrimitiveJob(BasePrimitiveJob, ABC):
    """QM Primitive Job class for executing QUA programs from PUBs."""

    def __init__(self, backend: QMBackend, pubs: List[Pub], input_type: InputType, **kwargs):
        super().__init__(job_id="pending", **kwargs)
        self._backend = backend
        self._pubs = pubs
        self._input_type = input_type
        self._qm_job: Optional[Union[RunningQmJob, QmPendingJob, List[QmPendingJob]]] = None
        self._program = None

    def status(self) -> JobStatus:
        """Return the job status."""
        if self._qm_job is None:
            raise RuntimeError("QM job has not submitted yet")
        status = self._qm_job.status
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
        """Attempt to cancel the job."""
        if self._qm_job is None:
            raise RuntimeError("QM job is not running")
        return self._qm_job.cancel()

    @property
    def qm_job(self) -> Optional[Union[RunningQmJob, List[QmPendingJob]]]:
        """Return the QM job."""
        return self._qm_job

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
    def program(self) -> Optional[Program]:
        """Return the QUA program."""
        return self._program

    @property
    def pubs(self) -> List[Pub]:
        """Return the PUBs used in this job."""
        return self._pubs