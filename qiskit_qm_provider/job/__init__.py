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

"""Job package: execution handles for ``QMBackend.run()`` and V2 primitives.

Returned job objects expose the compiled QUA programs on :attr:`~QMJob.programs` for
printing via ``qm.generate_qua_script``, bridge to the QM SDK through
:attr:`~QMJob.qm_jobs` / :meth:`~QMJob.get_qm_job`, and build Qiskit results in
:meth:`~QMJob.result`.

See the user guide: ``docs/jobs.md``.

Author: Arthur Strauss
Date: 2026-02-08
"""

from .qm_job import QMJob, IQCCJob
from .qm_sampler_job import QMSamplerJob
from .qm_estimator_job import QMEstimatorJob
from .iqcc_job_mixin import IQCCCloudExecutionError, IQCCJobMixin

__all__ = [
    "QMJob",
    "IQCCJob",
    "QMSamplerJob",
    "QMEstimatorJob",
    "IQCCCloudExecutionError",
    "IQCCJobMixin",
]
