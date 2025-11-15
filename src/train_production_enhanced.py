"""
Solar PV Power Prediction - Enhanced Training Script with EDA & Advanced Metrics

This script includes:
- Comprehensive EDA visualizations
- Advanced regression metrics
- Residual analysis
- Feature correlation analysis
- Prediction quality plots

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)
import joblib
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Loading dataset...")
print("="*70)

df = pd.read_csv('data/Solar_PV_Synthetic_Dataset.csv')

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n✓ Dataset Info:")
print(df.info())
print(f"\n✓ Statistical Summary:")
print(df.describe())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Performing Exploratory Data Analysis...")
print("="*70)

# Create visualizations directory
os.makedirs('models/visualizations', exist_ok=True)

# 2.1: Target Variable Distribution
print("\n[EDA 1/7] Creating target variable distribution plot...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Power_Output_W'], bins=50, color='#FFD700', edgecolor='black', alpha=0.7)
plt.xlabel('Power Output (W)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Power Output', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['Power_Output_W'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='#FFD700', alpha=0.7))
plt.ylabel('Power Output (W)', fontsize=12)
plt.title('Power Output Box Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/visualizations/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/01_target_distribution.png")

# 2.2: Feature Distributions
print("\n[EDA 2/7] Creating feature distribution plots...")
FEATURES = [
    'Solar_Irradiance_kWh_m2',
    'Temperature_C',
    'Wind_Speed_mps',
    'Relative_Humidity_%',
    'Panel_Tilt_deg',
    'Panel_Azimuth_deg',
    'Plane_of_Array_Irradiance',
    'Cell_Temperature_C'
]

fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, feature in enumerate(FEATURES):
    axes[idx].hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].set_title(f'Distribution: {feature}', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/visualizations/02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/02_feature_distributions.png")

# 2.3: Correlation Heatmap
print("\n[EDA 3/7] Creating correlation heatmap...")
plt.figure(figsize=(14, 10))
correlation_matrix = df[FEATURES + ['Power_Output_W']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='YlOrRd', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('models/visualizations/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/03_correlation_heatmap.png")

# Find top correlations with target
target_correlations = correlation_matrix['Power_Output_W'].drop('Power_Output_W').sort_values(ascending=False)
print("\n   Top 3 features correlated with Power Output:")
for i, (feat, corr) in enumerate(target_correlations.head(3).items(), 1):
    print(f"   {i}. {feat}: {corr:.4f}")

# 2.4: Pairplot of Top Features
print("\n[EDA 4/7] Creating pairplot of top correlated features...")
top_features = target_correlations.head(3).index.tolist() + ['Power_Output_W']
pairplot_data = df[top_features].sample(min(1000, len(df)), random_state=42)  # Sample for speed

pairplot = sns.pairplot(pairplot_data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'color': '#FFD700'},
                        diag_kws={'color': '#FFA500'})
pairplot.fig.suptitle('Pairplot of Top 3 Features vs Power Output', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('models/visualizations/04_pairplot_top_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/04_pairplot_top_features.png")

# 2.5: Missing Values Check
print("\n[EDA 5/7] Checking for missing values...")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("   ✓ No missing values found in dataset!")
else:
    print("   ⚠ Missing values detected:")
    print(missing_values[missing_values > 0])

# 2.6: Outlier Detection
print("\n[EDA 6/7] Detecting outliers using IQR method...")
outlier_counts = {}
for feature in FEATURES + ['Power_Output_W']:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))).sum()
    outlier_counts[feature] = outliers
    if outliers > 0:
        print(f"   {feature}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")

if sum(outlier_counts.values()) == 0:
    print("   ✓ No significant outliers detected!")

# 2.7: Feature vs Target Scatter Plots
print("\n[EDA 7/7] Creating feature vs target scatter plots...")
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, feature in enumerate(FEATURES):
    sample_data = df.sample(min(2000, len(df)), random_state=42)
    axes[idx].scatter(sample_data[feature], sample_data['Power_Output_W'], 
                     alpha=0.4, s=10, color='#FFD700', edgecolor='black', linewidth=0.5)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Power Output (W)', fontsize=10)
    axes[idx].set_title(f'{feature} vs Power Output', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df[feature].corr(df['Power_Output_W'])
    axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[idx].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('models/visualizations/05_feature_vs_target.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/05_feature_vs_target.png")

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Preparing data for training...")
print("="*70)

TARGET = 'Power_Output_W'
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Train-validation-test split (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"✓ Training set:   {X_train.shape[0]} samples")
print(f"✓ Validation set: {X_val.shape[0]} samples")
print(f"✓ Test set:       {X_test.shape[0]} samples")

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Training model with GridSearchCV...")
print("="*70)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

print("\n✓ Starting GridSearchCV...")
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
print(f"\n✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best CV MAE: {-grid_search.best_score_:.2f} W")

# ============================================================================
# STEP 5: COMPREHENSIVE EVALUATION
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Evaluating model with advanced metrics...")
print("="*70)

# Make predictions
y_train_pred = best_pipeline.predict(X_train)
y_val_pred = best_pipeline.predict(X_val)
y_test_pred = best_pipeline.predict(X_test)

# Calculate all metrics for each set
def calculate_metrics(y_true, y_pred, set_name):
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred)
    }
    
    print(f"\n{set_name} Metrics:")
    print(f"  MAE:                {metrics['mae']:.2f} W")
    print(f"  RMSE:               {metrics['rmse']:.2f} W")
    print(f"  R²:                 {metrics['r2']:.4f}")
    print(f"  MAPE:               {metrics['mape']:.2f} %")
    print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
    print(f"  Max Error:          {metrics['max_error']:.2f} W")
    
    return metrics

train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, "Validation")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test")

# ============================================================================
# STEP 6: RESIDUAL ANALYSIS & VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Creating residual analysis plots...")
print("="*70)

# 6.1: Residual Plot
print("\n[Plot 1/5] Residual plot...")
residuals_test = y_test - y_test_pred

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals_test, alpha=0.5, s=20, color='#FFD700', edgecolor='black', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Power (W)', fontsize=12)
plt.ylabel('Residuals (W)', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals_test, bins=50, color='#FFD700', edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (W)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Residual Distribution', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/visualizations/06_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/06_residual_analysis.png")

# 6.2: Predicted vs Actual
print("\n[Plot 2/5] Predicted vs Actual plot...")
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_test_pred, alpha=0.5, s=30, color='#FFD700', edgecolor='black', linewidth=0.5)

# Perfect prediction line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Power Output (W)', fontsize=14)
plt.ylabel('Predicted Power Output (W)', fontsize=14)
plt.title('Predicted vs Actual Power Output', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add R² score
plt.text(0.05, 0.95, f'R² = {test_metrics["r2"]:.4f}\nMAE = {test_metrics["mae"]:.2f} W',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('models/visualizations/07_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/07_predicted_vs_actual.png")

# 6.3: Error Distribution
print("\n[Plot 3/5] Error distribution plot...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
errors = np.abs(residuals_test)
plt.hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Error (W)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Absolute Error Distribution', fontsize=14, fontweight='bold')
plt.axvline(x=test_metrics['mae'], color='red', linestyle='--', linewidth=2, label=f'MAE = {test_metrics["mae"]:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
percentage_errors = np.abs(residuals_test / y_test) * 100
plt.hist(percentage_errors, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Percentage Error (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Percentage Error Distribution', fontsize=14, fontweight='bold')
plt.axvline(x=test_metrics['mape'], color='red', linestyle='--', linewidth=2, label=f'MAPE = {test_metrics["mape"]:.2f}%')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/visualizations/08_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/08_error_distribution.png")

# 6.4: Feature Importance
print("\n[Plot 4/5] Feature importance plot...")
feature_importance = best_pipeline.named_steps['model'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
colors = plt.cm.YlOrRd(np.linspace(0.4, 0.8, len(importance_df)))
plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance (RandomForest)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('models/visualizations/09_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/09_feature_importance.png")

# 6.5: Learning Curve
print("\n[Plot 5/5] Learning curve (sample sizes vs performance)...")
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
train_scores = []
val_scores = []

for size in train_sizes:
    n_samples = int(len(X_train) * size)
    X_subset = X_train[:n_samples]
    y_subset = y_train[:n_samples]
    
    temp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**{k.replace('model__', ''): v 
                                          for k, v in grid_search.best_params_.items()},
                                       random_state=42))
    ])
    temp_pipeline.fit(X_subset, y_subset)
    
    train_pred = temp_pipeline.predict(X_subset)
    val_pred = temp_pipeline.predict(X_val)
    
    train_scores.append(r2_score(y_subset, train_pred))
    val_scores.append(r2_score(y_val, val_pred))

plt.figure(figsize=(10, 6))
sample_counts = [int(len(X_train) * size) for size in train_sizes]
plt.plot(sample_counts, train_scores, 'o-', color='#FFD700', linewidth=2, markersize=8, label='Training Score')
plt.plot(sample_counts, val_scores, 'o-', color='#FFA500', linewidth=2, markersize=8, label='Validation Score')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Learning Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/visualizations/10_learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: models/visualizations/10_learning_curve.png")

# ============================================================================
# STEP 7: SAVE MODEL AND METRICS
# ============================================================================
print("\n" + "="*70)
print("STEP 7: Saving model and comprehensive metrics...")
print("="*70)

version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model artifact
model_artifact = {
    'pipeline': best_pipeline,
    'features': FEATURES,
    'version': version_timestamp,
    'train_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'best_params': grid_search.best_params_,
    'sklearn_version': sklearn.__version__
}

joblib.dump(model_artifact, 'models/pipeline_prod.joblib', compress=3)
print(f"✓ Model saved: models/pipeline_prod.joblib")

# Save comprehensive metrics
metrics_data = {
    'model_version': version_timestamp,
    'training_date': datetime.now().isoformat(),
    'data_split': {
        'train_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'train_percentage': 60.0,
        'validation_percentage': 20.0,
        'test_percentage': 20.0
    },
    'best_hyperparameters': grid_search.best_params_,
    'training_metrics': {
        'mae': round(train_metrics['mae'], 2),
        'rmse': round(train_metrics['rmse'], 2),
        'r2': round(train_metrics['r2'], 4),
        'mape': round(train_metrics['mape'], 2),
        'explained_variance': round(train_metrics['explained_variance'], 4),
        'max_error': round(train_metrics['max_error'], 2)
    },
    'validation_metrics': {
        'mae': round(val_metrics['mae'], 2),
        'rmse': round(val_metrics['rmse'], 2),
        'r2': round(val_metrics['r2'], 4),
        'mape': round(val_metrics['mape'], 2),
        'explained_variance': round(val_metrics['explained_variance'], 4),
        'max_error': round(val_metrics['max_error'], 2)
    },
    'test_metrics': {
        'mae': round(test_metrics['mae'], 2),
        'rmse': round(test_metrics['rmse'], 2),
        'r2': round(test_metrics['r2'], 4),
        'mape': round(test_metrics['mape'], 2),
        'explained_variance': round(test_metrics['explained_variance'], 4),
        'max_error': round(test_metrics['max_error'], 2)
    },
    'feature_importance': {
        feature: round(importance, 4)
        for feature, importance in zip(FEATURES, feature_importance)
    },
    'correlation_with_target': {
        feature: round(target_correlations[feature], 4)
        for feature in FEATURES
    },
    'outlier_counts': outlier_counts,
    'visualizations_generated': [
        '01_target_distribution.png',
        '02_feature_distributions.png',
        '03_correlation_heatmap.png',
        '04_pairplot_top_features.png',
        '05_feature_vs_target.png',
        '06_residual_analysis.png',
        '07_predicted_vs_actual.png',
        '08_error_distribution.png',
        '09_feature_importance.png',
        '10_learning_curve.png'
    ]
}

# Convert numpy types to Python types for JSON serialization
def convert_to_serializable(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Convert metrics_data to serializable format
metrics_data_serializable = convert_to_serializable(metrics_data)

with open('models/metrics.json', 'w') as f:
    json.dump(metrics_data_serializable, f, indent=2)
    print(f"✓ Metrics saved: models/metrics.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE WITH EDA & ADVANCED METRICS!")
print("="*70)
print(f"✓ Model Version: {version_timestamp}")
print(f"✓ Test MAE: {test_metrics['mae']:.2f} W")
print(f"✓ Test MAPE: {test_metrics['mape']:.2f} %")
print(f"✓ Test R²: {test_metrics['r2']:.4f}")
print(f"✓ Max Error: {test_metrics['max_error']:.2f} W")
print(f"\n✓ Generated 10 visualization plots in: models/visualizations/")
print(f"✓ Comprehensive metrics saved in: models/metrics.json")
print("\nNext steps:")
print("  1. Review visualizations in models/visualizations/")
print("  2. Start API server: uvicorn src.server:app --reload")
print("  3. Launch enhanced dashboard: streamlit run src/app_enhanced.py")
print("="*70 + "\n")