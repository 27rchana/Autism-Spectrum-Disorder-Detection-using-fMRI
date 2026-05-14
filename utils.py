import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

def load_data():
    # ✅ Load directly from current folder
    X_train = np.load("X_train_reduced.npy")
    X_test = np.load("X_test_reduced.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    return X_train, X_test, y_train, y_test

def get_circuits(num_features):
    feature_map = ZZFeatureMap(
        feature_dimension=num_features,
        reps=1,
        entanglement='linear'
    )

    ansatz = RealAmplitudes(
        num_qubits=num_features,
        reps=1
    )

    return feature_map, ansatz