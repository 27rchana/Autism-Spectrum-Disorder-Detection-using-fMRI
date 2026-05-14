from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os

# Save once
token = os.getenv("QISKIT_IBM_TOKEN")
if token:
	QiskitRuntimeService.save_account(
		channel="ibm_quantum_platform",
		token=token,
		overwrite=True,
	)
else:
	raise RuntimeError(
		"Set QISKIT_IBM_TOKEN in your environment, then rerun this script."
	)

# Load account
service = QiskitRuntimeService(channel="ibm_quantum_platform")

backend = service.least_busy(operational=True, simulator=False)
print("Selected backend:", backend.name)

# Tiny end-to-end runtime submission on the selected backend.
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_qc = pm.run(qc)

sampler = SamplerV2(mode=backend)
job = sampler.run([isa_qc], shots=1024)
print("Submitted job ID:", job.job_id())
print("Current job status:", job.status())

job.wait_for_final_state(timeout=300)
print("Final job status:", job.status())

result = job.result()
first_pub = result[0]

# SamplerV2 stores shot data in register containers; extract counts if present.
counts = None
if hasattr(first_pub, "data"):
	data_obj = first_pub.data
	if hasattr(data_obj, "meas") and hasattr(data_obj.meas, "get_counts"):
		counts = data_obj.meas.get_counts()
	elif hasattr(data_obj, "c") and hasattr(data_obj.c, "get_counts"):
		counts = data_obj.c.get_counts()

if counts is not None:
	print("Counts:", counts)
else:
	print("Result:", result)
