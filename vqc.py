import numpy as np
import os
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.metrics import accuracy_score

# Load data
X_train=np.load("X_train_reduced.npy")
X_test=np.load("X_test_reduced.npy")
y_train=np.load("y_train.npy")
y_test=np.load("y_test.npy")

num_features=X_train.shape[1]

# IBM connection with token from environment
token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise RuntimeError("Set QISKIT_IBM_TOKEN environment variable")

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=token,
    overwrite=True,
)

service=QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.least_busy(operational=True, simulator=False)
print(f"Running on real hardware: {backend.name}")

sampler=SamplerV2(mode=backend)

# Pass manager for hardware transpilation
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

# Feature map (simplified for real hardware)
feature_map=ZZFeatureMap(feature_dimension=num_features, reps=1, entanglement='linear')

# Ansatz (simplified for real hardware)
ansatz=RealAmplitudes(num_qubits=num_features, reps=1)

# Optimizer - reduced iterations for hardware
optimizer=COBYLA(maxiter=30)

# VQC with transpilation support
vqc=VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
    callback=None,
    pass_manager=pm
)

# Train
vqc.fit(X_train,y_train)

# Predict
y_pred=vqc.predict(X_test)

print("VQC Accuracy:",accuracy_score(y_test,y_pred))