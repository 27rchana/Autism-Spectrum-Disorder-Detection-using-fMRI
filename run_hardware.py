import numpy as np
import os
import time
from collections import Counter
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.metrics import accuracy_score
from utils import load_data, get_circuits


def make_service(token=None):
    # Try common channel names across runtime versions.
    channels = ["ibm_quantum", "ibm_cloud", "ibm_quantum_platform"]
    last_error = None
    for channel in channels:
        try:
            if token:
                QiskitRuntimeService.save_account(
                    channel=channel,
                    token=token,
                    overwrite=True,
                )
            return QiskitRuntimeService(channel=channel), channel
        except Exception as exc:
            last_error = exc
    if token:
        raise RuntimeError(
            "Failed to initialize IBM Runtime service with provided token. "
            f"Last error: {last_error}"
        )

    raise RuntimeError(
        "QISKIT_IBM_TOKEN is not set in this process and no saved IBM Runtime "
        f"account could be loaded. Last error: {last_error}\n"
        "Set it for this terminal (PowerShell):\n"
        "$env:QISKIT_IBM_TOKEN='YOUR_TOKEN'\n"
        "Or persist for future terminals:\n"
        "setx QISKIT_IBM_TOKEN \"YOUR_TOKEN\""
    )


def pick_seed_subset(X, y, max_per_class=1):
    # Ensure fit sees at least one sample from each class when available.
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0][:max_per_class]
        indices.extend(cls_idx.tolist())

    if len(indices) < 2:
        indices = list(range(min(2, len(y))))

    return X[indices], y[indices]


def print_section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def format_counter(values):
    counts = Counter(np.asarray(values).tolist())
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def backend_qubit_count(backend):
    num_qubits = getattr(backend, "num_qubits", None)
    if num_qubits is not None:
        return num_qubits
    target = getattr(backend, "target", None)
    if target is not None and hasattr(target, "num_qubits"):
        return target.num_qubits
    return "unknown"


run_started_at = datetime.now()
print_section("RUN START")
print(f"Start time: {run_started_at.isoformat(timespec='seconds')}")

# =========================
# 1. LOAD DATA
# =========================
X_train, X_test, y_train, y_test = load_data()
print_section("DATA SUMMARY")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution: {format_counter(y_train)}")
print(f"y_test distribution: {format_counter(y_test)}")

#  Reduce features (fewer qubits → faster)
num_features = min(4, X_train.shape[1])
X_train = X_train[:, :num_features]
X_test = X_test[:, :num_features]
print(f"Using features: {num_features}")

# VERY SMALL test set (important for hardware)
X_test = X_test[:1]
y_test = y_test[:1]
print(f"Hardware test subset shape: X={X_test.shape}, y={y_test.shape}")

# =========================
# 2. LOAD TRAINED WEIGHTS
# =========================
weights = np.load("saved_model/weights.npy")
print_section("MODEL ARTIFACTS")
print(f"Loaded weights path: saved_model/weights.npy")
print(f"Weights length: {len(weights)}")

# =========================
# 3. IBM QUANTUM SETUP
# =========================
token = os.getenv("QISKIT_IBM_TOKEN")
service, channel_used = make_service(token)
print_section("IBM RUNTIME")
if token:
    print("Using QISKIT_IBM_TOKEN from current environment")
else:
    print("QISKIT_IBM_TOKEN not found in current environment; trying saved IBM Runtime account...")
print(f"Connected via channel: {channel_used}")

# =========================
# 4. SELECT BEST BACKEND
# =========================
print("Finding least busy backend...")

backend = service.least_busy(operational=True, simulator=False)

print(f"Running on: {backend.name}")
print(f"Backend version: {getattr(backend, 'backend_version', 'unknown')}")
print(f"Backend qubits: {backend_qubit_count(backend)}")

# =========================
# 5. SAMPLER (LOW SHOTS)
# =========================
sampler = SamplerV2(mode=backend)
print(f"Sampler mode backend: {backend.name}")

# =========================
# 6. PASS MANAGER
# =========================
pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=1
)

# =========================
# 7. CIRCUITS
# =========================
feature_map, ansatz = get_circuits(num_features)

# =========================
# 8. VQC MODEL
# =========================
min_valid_maxiter = len(weights) + 2
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=min_valid_maxiter),
    initial_point=weights,
    sampler=sampler,
    pass_manager=pm
)
print_section("VQC CONFIG")
print(f"COBYLA maxiter: {min_valid_maxiter}")
print(f"Feature map qubits: {feature_map.num_qubits}")
print(f"Ansatz qubits: {ansatz.num_qubits}")

# =========================
# 9. INITIALIZE MODEL STATE
# =========================
print("Initializing model state...")
X_seed, y_seed = pick_seed_subset(X_train, y_train, max_per_class=1)
print(f"Seed subset shape: X={X_seed.shape}, y={y_seed.shape}")
print(f"Seed class distribution: {format_counter(y_seed)}")
vqc.fit(X_seed, y_seed)

# =========================
# 10. RUN ON REAL HARDWARE
# =========================
print("Running on real quantum hardware...")
predict_start = time.perf_counter()
y_pred = vqc.predict(X_test)
y_pred = np.atleast_1d(y_pred)
predict_seconds = time.perf_counter() - predict_start

# =========================
# 11. RESULTS
# =========================
print_section("RESULTS")
print("Predictions:", y_pred)
print("Actual:", y_test)
for idx, (pred, actual) in enumerate(zip(y_pred, y_test)):
    status = "OK" if pred == actual else "MISS"
    print(f"Sample {idx}: pred={pred}, actual={actual} -> {status}")

acc = accuracy_score(y_test, y_pred)
print("Hardware Accuracy:", acc)
print(f"Prediction runtime: {predict_seconds:.2f}s")
print(f"Run finished at: {datetime.now().isoformat(timespec='seconds')}")