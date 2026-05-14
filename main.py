import numpy as np
import os
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
X_train=np.load("X_train_reduced.npy")
X_test=np.load("X_test_reduced.npy")
y_train=np.load("y_train.npy")
y_test=np.load("y_test.npy")

num_features=X_train.shape[1]

# IBM Quantum connection with token from environment
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

# Feature map
feature_map=ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')

# Quantum kernel (REAL quantum)
quantum_kernel=FidelityQuantumKernel(feature_map=feature_map,sampler=sampler)

# Train QSVM
qsvm=SVC(kernel=quantum_kernel.evaluate)
qsvm.fit(X_train,y_train)

# Predict
y_pred=qsvm.predict(X_test)

print("QSVM Accuracy:",accuracy_score(y_test,y_pred))