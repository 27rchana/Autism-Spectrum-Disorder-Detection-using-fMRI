import numpy as np
import os

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.metrics import accuracy_score

from utils import load_data, get_circuits

# =========================
# LOAD DATA
# =========================
X_train, X_test, y_train, y_test = load_data()

# =========================
# REDUCE SIZE (IMPORTANT)
# =========================
num_features = min(4, X_train.shape[1])   # 🔥 reduce qubits
X_train = X_train[:, :num_features]
X_test = X_test[:, :num_features]

X_train = X_train[:30]   # 🔥 reduce training samples
y_train = y_train[:30]

# =========================
# BACKEND (SIMULATOR)
# =========================
backend = AerSimulator()

# ✅ FIX: transpilation
pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1
)

sampler = SamplerV2(mode=backend)

# =========================
# CIRCUITS
# =========================
feature_map, ansatz = get_circuits(num_features)

# =========================
# OPTIMIZER
# =========================
optimizer = COBYLA(maxiter=15)   # 🔥 reduced iterations

# =========================
# MODEL (FIXED)
# =========================
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
    pass_manager=pm   # ✅ CRITICAL FIX
)

# =========================
# TRAIN
# =========================
print("Training on simulator...")
vqc.fit(X_train, y_train)

# =========================
# SAVE WEIGHTS
# =========================
os.makedirs("saved_model", exist_ok=True)
np.save("saved_model/weights.npy", vqc.weights)

# =========================
# TEST
# =========================
y_pred = vqc.predict(X_test[:10])
acc = accuracy_score(y_test[:10], y_pred)

print("Simulator Accuracy:", acc)