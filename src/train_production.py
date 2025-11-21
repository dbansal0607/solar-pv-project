"""
Solar PV Power Prediction - Production Model Training Script

This script trains a RandomForest model inside a Pipeline with hyperparameter tuning,
saves the model with metadata, and generates performance metrics.

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime
import os

# ============================================================================
# STEP 1: LOAD AND INSPECT DATA
# ============================================================================
print("=" * 70)
print("STEP 1: Loading dataset...")
print("=" * 70)

# Load the CSV file into a pandas DataFrame
# This contains synthetic solar PV data with environmental parameters
df = pd.read_csv("data/Solar_PV_Synthetic_Dataset.csv")

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ First 5 rows preview:")
print(df.head())
print(f"\n✓ Column data types:")
print(df.dtypes)
print(f"\n✓ Missing values check:")
print(df.isnull().sum())

# ============================================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining features and target variable...")
print("=" * 70)

# These are the input features (X) that the model will learn from
# They represent environmental and panel configuration parameters
FEATURES = [
    "Solar_Irradiance_kWh_m2",  # Amount of sunlight hitting the panel
    "Temperature_C",  # Ambient air temperature
    "Wind_Speed_mps",  # Wind speed (affects cooling)
    "Relative_Humidity_%",  # Air moisture content
    "Panel_Tilt_deg",  # Angle of panel tilt
    "Panel_Azimuth_deg",  # Compass direction panel faces
    "Plane_of_Array_Irradiance",  # Direct irradiance on panel surface
    "Cell_Temperature_C",  # Temperature of solar cells themselves
]

# This is what we want to predict (y)
TARGET = "Power_Output_W"

# Extract features (X) and target (y) from the DataFrame
X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"✓ Features selected: {len(FEATURES)} variables")
print(f"  {FEATURES}")
print(f"✓ Target variable: {TARGET}")
print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target vector shape: {y.shape}")

# ============================================================================
# STEP 3: TRAIN-VALIDATION-TEST SPLIT (60/20/20)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Splitting data into train/validation/test sets (60/20/20)...")
print("=" * 70)

# First split: 80% for training+validation, 20% for final test
# random_state=42 ensures reproducibility (same split every time)
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,  # 20% held out for final testing
    random_state=42,
    shuffle=True,  # Randomly shuffle before splitting
)

# Second split: Split the 80% into 60% train and 20% validation
# 0.25 of 80% = 20% of total data
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 25% of 80% = 20% of total
)

print(
    f"✓ Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)"
)
print(f"✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(
    f"✓ Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)"
)

# Why 60/20/20?
# - 60% training: Used to fit the model (teach it patterns)
# - 20% validation: Used during GridSearchCV to tune hyperparameters
# - 20% test: Final unseen data to evaluate real-world performance

# ============================================================================
# STEP 4: BUILD THE MACHINE LEARNING PIPELINE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Creating ML Pipeline with StandardScaler + RandomForest...")
print("=" * 70)

# A Pipeline chains together preprocessing and model training
# This ensures the same transformations are applied during training and prediction
pipeline = Pipeline(
    [
        # Step 1: StandardScaler - Normalize features to have mean=0, std=1
        # Why? Different features have different scales (e.g., temperature vs irradiance)
        # Scaling prevents features with larger values from dominating the model
        ("scaler", StandardScaler()),
        # Step 2: RandomForestRegressor - The actual ML model
        # Random Forest creates many decision trees and averages their predictions
        # It handles non-linear relationships and feature interactions well
        ("model", RandomForestRegressor(random_state=42)),
    ]
)

print("✓ Pipeline created with 2 steps:")
print("  1. StandardScaler: Normalizes feature values")
print("  2. RandomForestRegressor: Ensemble of decision trees")

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Hyperparameter tuning with GridSearchCV...")
print("=" * 70)

# Define hyperparameters to test
# GridSearchCV will try all combinations and pick the best
param_grid = {
    # n_estimators: Number of trees in the forest
    # More trees = better performance but slower training
    "model__n_estimators": [100, 200, 300],
    # max_depth: Maximum depth of each tree
    # Deeper trees can learn complex patterns but may overfit
    # None means nodes expand until all leaves are pure
    "model__max_depth": [10, 20, None],
    # min_samples_split: Minimum samples required to split a node
    # Higher values prevent overfitting by avoiding tiny splits
    "model__min_samples_split": [2, 5, 10],
    # min_samples_leaf: Minimum samples required in a leaf node
    # Higher values create smoother decision boundaries
    "model__min_samples_leaf": [1, 2, 4],
}

print(f"✓ Hyperparameter grid defined:")
print(f"  - n_estimators: {param_grid['model__n_estimators']}")
print(f"  - max_depth: {param_grid['model__max_depth']}")
print(f"  - min_samples_split: {param_grid['model__min_samples_split']}")
print(f"  - min_samples_leaf: {param_grid['model__min_samples_leaf']}")
print(f"✓ Total combinations to test: {3 * 3 * 3 * 3} = 81")

# GridSearchCV automatically:
# 1. Tries all parameter combinations
# 2. Uses cross-validation (cv=3) to evaluate each combination
# 3. Picks the combination with the best validation score
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation on training data
    scoring="neg_mean_absolute_error",  # Minimize MAE (negative because sklearn maximizes)
    n_jobs=-1,  # Use all CPU cores for parallel processing
    verbose=2,  # Print progress updates
)

print("\n✓ Starting GridSearchCV (this may take 2-5 minutes)...")
print("  Cross-validating 81 combinations with 3 folds each = 243 model fits")

# Fit the grid search on training data
# This trains 243 models (81 combinations × 3 folds)
grid_search.fit(X_train, y_train)

print("\n✓ Grid search complete!")
print(f"✓ Best hyperparameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"✓ Best cross-validation MAE: {-grid_search.best_score_:.2f} W")

# Extract the best pipeline (with best hyperparameters)
best_pipeline = grid_search.best_estimator_

# ============================================================================
# STEP 6: EVALUATE ON VALIDATION SET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Evaluating on validation set...")
print("=" * 70)

# Make predictions on the validation set (data not used in training)
y_val_pred = best_pipeline.predict(X_val)

# Calculate error metrics
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print(f"✓ Validation Metrics:")
print(f"  - MAE:  {val_mae:.2f} W")
print(f"  - RMSE: {val_rmse:.2f} W")
print(f"  - R²:   {val_r2:.4f}")

# What do these metrics mean?
# - MAE (Mean Absolute Error): Average prediction error in Watts
#   Example: MAE=50W means predictions are off by 50W on average
# - RMSE (Root Mean Squared Error): Similar to MAE but penalizes large errors more
#   Always >= MAE. Large difference suggests some big outlier errors
# - R² (R-squared): Proportion of variance explained (0 to 1)
#   0.95 = model explains 95% of power output variation
#   Higher is better (1.0 = perfect predictions)

# ============================================================================
# STEP 7: FINAL EVALUATION ON TEST SET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Final evaluation on test set (unseen data)...")
print("=" * 70)

# Make predictions on completely unseen test data
y_test_pred = best_pipeline.predict(X_test)

# Calculate final test metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"✓ Test Set Metrics (FINAL PERFORMANCE):")
print(f"  - MAE:  {test_mae:.2f} W")
print(f"  - RMSE: {test_rmse:.2f} W")
print(f"  - R²:   {test_r2:.4f}")

# Compare test vs validation metrics
print(f"\n✓ Validation vs Test comparison:")
print(f"  - MAE:  Validation={val_mae:.2f}W  |  Test={test_mae:.2f}W")
print(f"  - RMSE: Validation={val_rmse:.2f}W  |  Test={test_rmse:.2f}W")
print(f"  - R²:   Validation={val_r2:.4f}  |  Test={test_r2:.4f}")

# If test metrics are similar to validation, the model generalizes well
# If test metrics are much worse, the model may be overfitting

# ============================================================================
# STEP 8: SAVE THE PRODUCTION MODEL WITH METADATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Saving production model artifact...")
print("=" * 70)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Generate version timestamp (YYYYMMDD_HHMMSS format)
version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create model artifact dictionary
# This bundles the pipeline with important metadata
model_artifact = {
    "pipeline": best_pipeline,  # The trained sklearn Pipeline
    "features": FEATURES,  # List of feature names (critical for prediction!)
    "version": version_timestamp,  # Version identifier
    "train_date": datetime.now().isoformat(),  # ISO format timestamp
    "training_samples": len(X_train),  # Number of samples used for training
    "best_params": grid_search.best_params_,  # Best hyperparameters found
    "sklearn_version": "1.3.0",  # scikit-learn version (for compatibility)
}

# Save the artifact using joblib (efficient for sklearn objects)
model_path = "models/pipeline_prod.joblib"
joblib.dump(model_artifact, model_path, compress=3)  # compress=3 for smaller file size

print(f"✓ Model artifact saved to: {model_path}")
print(f"✓ Version: {version_timestamp}")
print(f"✓ Artifact structure:")
print(f"  {{")
print(f"    'pipeline': <Pipeline object>,")
print(f"    'features': {FEATURES},")
print(f"    'version': '{version_timestamp}',")
print(f"    'train_date': '{datetime.now().isoformat()}',")
print(f"    'training_samples': {len(X_train)},")
print(f"    'best_params': {{...}},")
print(f"    'sklearn_version': '1.3.0'")
print(f"  }}")

# Why save metadata?
# - 'features': Ensures predictions use correct feature order
# - 'version': Track which model version is deployed
# - 'train_date': Audit trail for model updates
# - 'best_params': Reproduce model if needed
# - 'sklearn_version': Warn if loading with incompatible version

# ============================================================================
# STEP 9: SAVE METRICS TO JSON
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Saving metrics to JSON file...")
print("=" * 70)

# Create metrics dictionary for easy viewing
metrics = {
    "model_version": version_timestamp,
    "training_date": datetime.now().isoformat(),
    "data_split": {
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "train_percentage": 60.0,
        "validation_percentage": 20.0,
        "test_percentage": 20.0,
    },
    "best_hyperparameters": grid_search.best_params_,
    "validation_metrics": {
        "mae": round(val_mae, 2),
        "rmse": round(val_rmse, 2),
        "r2": round(val_r2, 4),
    },
    "test_metrics": {
        "mae": round(test_mae, 2),
        "rmse": round(test_rmse, 2),
        "r2": round(test_r2, 4),
    },
    "feature_importance": {
        feature: round(importance, 4)
        for feature, importance in zip(
            FEATURES, best_pipeline.named_steps["model"].feature_importances_
        )
    },
}

# Save to JSON file
metrics_path = "models/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Metrics saved to: {metrics_path}")
print(f"✓ Metrics summary:")
print(json.dumps(metrics, indent=2))

# ============================================================================
# TRAINING COMPLETE
# ============================================================================
print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"✓ Production model ready: {model_path}")
print(f"✓ Metrics file ready: {metrics_path}")
print(f"✓ Model version: {version_timestamp}")
print(f"✓ Test MAE: {test_mae:.2f} W  |  Test R²: {test_r2:.4f}")
print("\nNext steps:")
print("  1. Review metrics.json to verify performance")
print("  2. Start the FastAPI server: uvicorn src.server:app --reload")
print("  3. Launch the dashboard: streamlit run src/app.py")
print("=" * 70)
