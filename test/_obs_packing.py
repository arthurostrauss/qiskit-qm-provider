"""Host-side decoding of packed estimator observable words, for tests only."""

from qiskit_qm_provider.job.pub_parameters import OBS_QUBITS_PER_WORD


def unpack_obs_codes(words, num_qubits: int) -> list[int]:
    """Inverse of :func:`~qiskit_qm_provider.job.pub_parameters.pack_obs_codes` (what QUA's shifts and masks compute)."""
    return [
        (int(words[q // OBS_QUBITS_PER_WORD]) >> (2 * (q % OBS_QUBITS_PER_WORD))) & 3 for q in range(num_qubits)
    ]
