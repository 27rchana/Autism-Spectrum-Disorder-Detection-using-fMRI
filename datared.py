import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# ==============================
# 1. LOAD DATA
# ==============================
X = np.load("X.npy")
y = np.load("y.npy")

print("Original Shape:", X.shape)

# ==============================
# 2. HANDLE NaN VALUES (CRITICAL FIX)
# ==============================

# Convert to DataFrame for easier handling
X_df = pd.DataFrame(X)

# Remove columns that are completely NaN
X_df = X_df.dropna(axis=1, how='all')

print("After removing all-NaN columns:", X_df.shape)

# Impute remaining NaNs using mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X_df)

# ==============================
# 3. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. STANDARDIZATION
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 5. SelectKBest (40000 → 50)
# ==============================
kbest = SelectKBest(score_func=f_classif, k=50)
X_train_kbest = kbest.fit_transform(X_train, y_train)
X_test_kbest = kbest.transform(X_test)

print("After SelectKBest:", X_train_kbest.shape)

# ==============================
# 6. RFE (50 → 4)
# ==============================
svm_estimator = SVC(kernel='linear')

rfe = RFE(estimator=svm_estimator, n_features_to_select=4)
X_train_rfe = rfe.fit_transform(X_train_kbest, y_train)
X_test_rfe = rfe.transform(X_test_kbest)

print("After RFE:", X_train_rfe.shape)

# ==============================
# 7. FINAL MODEL
# ==============================
final_model = SVC(kernel='rbf')
final_model.fit(X_train_rfe, y_train)

# ==============================
# 8. EVALUATION
# ==============================
y_pred = final_model.predict(X_test_rfe)
accuracy = accuracy_score(y_test, y_pred)

print("Final Accuracy:", accuracy)

# ==============================
# 9. FEATURE INDEX MAPPING
# ==============================
kbest_indices = kbest.get_support(indices=True)
rfe_indices = rfe.get_support(indices=True)

final_selected_features = kbest_indices[rfe_indices]

print("Final 4 Selected Feature Indices:", final_selected_features)

# ==============================
# 10. FINAL REDUCED DATASET
# ==============================
X_reduced = X[:, final_selected_features]

print("Final Reduced Dataset Shape:", X_reduced.shape)

# ==============================
# 11. SAVE FILES
# ==============================

# Save npy
np.save("X_reduced.npy", X_reduced)
np.save("y.npy", y)

# Save csv
X_df = pd.DataFrame(X_reduced, columns=[f"Feature_{i}" for i in range(4)])
y_df = pd.DataFrame(y, columns=["Target"])

X_df.to_csv("X_reduced.csv", index=False)
y_df.to_csv("y.csv", index=False)

# Combined file
combined = np.column_stack((X_reduced, y))
combined_df = pd.DataFrame(combined, columns=[f"F{i}" for i in range(4)] + ["Target"])
combined_df.to_csv("final_dataset.csv", index=False)

print("All files saved successfully!")