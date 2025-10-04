from qiskit.circuit.annotation import Annotation


class QMSaveAnnotation(Annotation):
    """
    Annotation to indicate that a save statement should be issued
    """

    namespace = "qm_save"

    def __init__(
        self,
    ):
        pass
