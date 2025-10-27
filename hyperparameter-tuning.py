# ===========================================
# FAST Logistic Regression Hyperparameter Tuning
# ===========================================

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")  # Suppress convergence and deprecation warnings

# =============== STEP 1: Load Dataset ====================
# Replace with your dataset path
data = pd.read_csv("placement.csv")

print("✅ Data loaded successfully.")
print("Shape:", data.shape)
print("Columns:", list(data.columns))
print()

# =============== STEP 2: Define Features & Target ====================
# Replace 'target_column' with your actual target column name
target_col = 'placed'

X = data.drop(columns=[target_col])
y = data[target_col]

# Encode categorical target if needed
if y.dtype == 'object':
    y = y.astype('category').cat.codes

# =============== STEP 3: Split Train/Test ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============== STEP 4: Standardize ====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============== STEP 5: Optimized Hyperparameter Grid ====================
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'l1_ratio': [None, 0.5],
    'class_weight': [None, 'balanced'],
    'max_iter': [500],
    'multi_class': ['ovr']
}

# =============== STEP 6: Grid Search Setup ====================
log_reg = LogisticRegression()

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,            # Faster than 5-fold
    n_jobs=-1,       # Use all CPU cores
    verbose=2
)

# =============== STEP 7: Train & Time the Search ====================
print("🚀 Starting Grid Search...\n")
start_time = time.time()

grid_search.fit(X_train, y_train)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Grid Search completed in {elapsed_time:.2f} seconds")

# =============== STEP 8: Evaluate Best Model ====================
print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📊 Test Set Performance:")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =============== STEP 9: Save Full Results ====================
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(by="mean_test_score", ascending=False)
results.to_csv("logistic_regression_results_fast.csv", index=False)

print("\n💾 All results saved to 'logistic_regression_results_fast.csv'")
