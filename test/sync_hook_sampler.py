from iqcc_cloud_client.runtime import get_qm_job
from qiskit.qasm3 import loads
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers import DataBin, BitArray
import numpy as np

from qiskit_qm_provider.parameter_table import ParameterTable, InputType

job = get_qm_job()
openqasm_code = [
    "OPENQASM 3.0;\ninput float[64] p;\nbit[1] meas;\nreset $0;\nrz(1.5707963267948966) $0;\nsx $0;\nrz(3.141592653589793 + p) $0;\nsx $0;\nrz(7.853981633974483) $0;\nbarrier $0;\nmeas[0] = measure $0;\n"
]
circuits = [loads(code) for code in openqasm_code]
parameter_values = [
    [
        [8.80449828e-01],
        [8.85857857e-01],
        [3.21387091e-02],
        [9.69636624e-01],
        [4.75666842e-01],
        [6.94624345e-01],
        [5.02078681e-01],
        [3.36981921e-01],
        [3.41244112e-01],
        [3.38450927e-01],
        [4.13718366e-01],
        [5.68535644e-03],
        [5.58906163e-01],
        [6.94013117e-01],
        [3.24476709e-02],
        [8.78344983e-01],
        [6.10282443e-01],
        [3.51650984e-01],
        [6.27137429e-01],
        [4.77821876e-01],
        [3.96218323e-01],
        [4.03641253e-01],
        [1.32268744e-01],
        [7.65892673e-01],
        [1.59785356e-01],
        [2.01762839e-01],
        [5.28526744e-01],
        [3.07598899e-01],
        [4.29355477e-01],
        [5.05378474e-01],
        [8.94279501e-01],
        [2.18693162e-01],
        [2.65806777e-01],
        [7.18582019e-01],
        [4.46032983e-01],
        [5.66306754e-01],
        [3.56935886e-01],
        [8.00284947e-02],
        [9.88827650e-01],
        [7.74230899e-01],
        [6.28294725e-01],
        [2.60714511e-02],
        [1.87857496e-01],
        [2.89743718e-01],
        [9.58539423e-01],
        [3.99683267e-02],
        [1.96482660e-01],
        [5.00351476e-02],
        [1.61829377e-01],
        [6.20372643e-01],
        [1.62468038e-01],
        [4.19856943e-01],
        [4.30512977e-01],
        [7.09750231e-01],
        [4.36507034e-01],
        [5.75919866e-01],
        [6.13248244e-01],
        [6.28757488e-02],
        [8.80350745e-01],
        [2.17141535e-01],
        [1.01952245e-01],
        [7.57148028e-01],
        [8.70093106e-01],
        [4.35740805e-01],
        [7.56033646e-01],
        [7.43005736e-01],
        [4.19695687e-01],
        [2.07139569e-01],
        [2.75424166e-01],
        [6.66653115e-01],
        [7.99255962e-02],
        [1.02700063e-01],
        [2.47400463e-01],
        [7.63177076e-01],
        [1.31947166e-01],
        [8.78215654e-02],
        [6.86364622e-01],
        [9.87875337e-01],
        [9.09697554e-01],
        [9.50485291e-01],
        [2.65921090e-01],
        [3.96094890e-01],
        [1.20189354e-01],
        [4.30790346e-01],
        [8.11966077e-01],
        [3.20439494e-01],
        [8.24905704e-01],
        [7.76570340e-01],
        [4.78298497e-03],
        [9.25981575e-01],
        [4.08510586e-01],
        [4.08189428e-01],
        [9.31275271e-01],
        [6.38147451e-01],
        [5.45201014e-01],
        [1.70516770e-01],
        [7.47468936e-01],
        [7.64625375e-01],
        [2.77581189e-01],
        [8.76158584e-01],
        [7.13715115e-01],
        [1.83243687e-01],
        [5.18094513e-01],
        [8.95557443e-01],
        [5.74721160e-01],
        [6.04678757e-01],
        [4.95108213e-01],
        [3.19716822e-01],
        [2.36348104e-01],
        [2.10554284e-01],
        [6.11628014e-01],
        [2.59422084e-03],
        [2.94149411e-01],
        [8.93691613e-01],
        [8.17866136e-02],
        [3.97155316e-01],
        [4.95033294e-02],
        [1.15367111e-01],
        [8.81041742e-02],
        [7.39065783e-01],
        [7.00952241e-03],
        [5.70860828e-01],
        [7.51882175e-01],
        [8.44676304e-01],
        [5.79741915e-01],
        [8.66460671e-01],
        [9.27304979e-01],
        [6.06498297e-01],
        [8.91038292e-01],
        [9.83166181e-01],
        [2.82058720e-01],
        [5.02586276e-01],
        [7.66934297e-01],
        [3.56048835e-01],
        [7.89272873e-01],
        [3.73923996e-01],
        [7.10299575e-01],
        [4.86513177e-01],
        [3.88454501e-01],
        [4.55662276e-01],
        [6.55497903e-01],
        [4.36274481e-01],
        [7.15864351e-01],
        [3.28857407e-01],
        [7.40744766e-01],
        [7.70409656e-02],
        [5.73277324e-01],
        [1.24334346e-01],
        [5.36104657e-01],
        [5.26795443e-01],
        [5.47073538e-01],
        [7.06504832e-01],
        [4.46028440e-01],
        [6.83343559e-01],
        [8.26998988e-01],
        [4.85352384e-01],
        [4.27812793e-01],
        [8.29762887e-01],
        [3.91363348e-01],
        [4.90608862e-01],
        [1.66155273e-01],
        [3.48372998e-01],
        [4.01531282e-01],
        [3.81766468e-02],
        [2.09185323e-01],
        [4.71490948e-01],
        [8.02312091e-01],
        [1.37634005e-01],
        [8.82832068e-01],
        [6.35661756e-01],
        [9.12674281e-01],
        [4.41585721e-01],
        [3.19042593e-01],
        [1.78991134e-01],
        [5.88088729e-01],
        [1.09213004e-01],
        [9.37089429e-01],
        [2.12278270e-01],
        [6.42506249e-01],
        [4.76817855e-01],
        [6.70626183e-01],
        [6.57063478e-01],
        [3.47379440e-01],
        [8.52964274e-01],
        [5.53330375e-02],
        [1.85318644e-01],
        [6.00524832e-01],
        [1.33820821e-01],
        [1.29138043e-01],
        [3.60873228e-02],
        [8.46537049e-04],
        [5.17173484e-01],
        [5.97088346e-01],
        [9.21918346e-01],
        [3.18167562e-01],
        [3.75650557e-01],
        [3.24892403e-01],
        [2.50350752e-02],
        [5.39218179e-01],
        [1.72955991e-01],
    ]
]
parameter_tables = [
    ParameterTable.from_qiskit(
        circuit,
        input_type="None",
        filter_function=lambda x: isinstance(x, Parameter),
        name=f"param_table_{i}",
    )
    for i, circuit in enumerate(circuits)
]
for parameter_value, parameter_table in zip(parameter_values, parameter_tables):
    if parameter_table is not None and parameter_table.input_type is not None:
        param_dict = {
            param.name: value
            for param, value in zip(parameter_table.parameters, parameter_value)
        }
        parameter_table.push_to_opx(param_dict, job)

results_handle = job.result_handles
results_handle.wait_for_all_values()

# all_data = []
# for i, circuit in enumerate(circuits):
#     qc_meas_data = {}
#     for creg in circuit.cregs:
#         data = results_handle.get("" + creg.name + "_" + str(i)).fetch_all()["value"]
#         meas_level = self.metadata.get("meas_level")
#         if meas_level == "classified":
#             bit_array = BitArray.from_samples(data.tolist(), creg.size).reshape(pub.shape)
#             qc_meas_data[creg.name] = bit_array
#         elif meas_level == "kerneled":
#             # TODO: Assume that buffering was done like (2, creg.size)
#             qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
#                 pub.shape + (pub.shots, creg.size)
#             )
#         else:
#             # TODO: Figure it out
#             qc_meas_data[creg.name] = np.array([d[0] + 1j * d[1] for d in data], dtype=complex).reshape(
#                 pub.shape + (pub.shots, creg.size)
#             )

#     sampler_data = SamplerPubResult(DataBin(**qc_meas_data))
#     all_data.append(sampler_data)

# result = PrimitiveResult(all_data)
# print(result)
