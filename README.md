# Autism Disorder Detection using fMRI

This project explores autism spectrum disorder detection from fMRI-derived features using classical and quantum machine learning workflows. It includes preprocessing scripts, reduced feature datasets, a simulator-based VQC training flow, and a real IBM Quantum hardware run.

## Project Structure

- `datared.py` preprocesses the raw dataset, handles missing values, selects informative features, and saves reduced datasets.
- `train_simulator.py` trains a Variational Quantum Classifier on a simulator and saves model weights.
- `run_hardware.py` loads saved weights and runs inference on real IBM Quantum hardware.
- `main.py`, `vqc.py`, and `qu.py` are alternative quantum experiment scripts.
- `utils.py` contains shared data-loading and circuit-building helpers.
- `X_train_reduced.npy`, `X_test_reduced.npy`, `y_train.npy`, and `y_test.npy` are the prepared training and test arrays.
- `saved_model/weights.npy` stores trained weights for hardware execution.

## Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
- `qiskit`
- `qiskit-aer`
- `qiskit-ibm-runtime`
- `qiskit-machine-learning`
- `qiskit-algorithms`

## Setup

Install dependencies in your environment:

```bash
pip install numpy pandas scikit-learn qiskit qiskit-aer qiskit-ibm-runtime qiskit-machine-learning qiskit-algorithms
```

If you plan to run the hardware scripts, set your IBM Quantum token first:

```powershell
$env:QISKIT_IBM_TOKEN="YOUR_TOKEN"
```

## How To Run

### Train on Simulator

This trains the quantum model locally on a simulator and saves weights to `saved_model/weights.npy`.

```bash
python train_simulator.py
```

### Run on IBM Quantum Hardware

This loads the saved weights and performs inference on a least-busy real backend.

```bash
python run_hardware.py
```

### Other Scripts

- `python main.py` runs a quantum kernel SVM workflow on real hardware.
- `python vqc.py` runs another VQC hardware experiment.
- `python qu.py` submits a minimal IBM Runtime test circuit.

## Notes

- The hardware scripts require a valid IBM Quantum account and access to a real backend.
- The repository already includes reduced `.npy` datasets, so you can run the model scripts without rerunning preprocessing.
- `datared.py` expects raw inputs named `X.npy` and `y.npy` if you want to regenerate the reduced data files.

## Output

Typical outputs include model accuracy, selected backend information, prediction results, and saved weights for reuse.
