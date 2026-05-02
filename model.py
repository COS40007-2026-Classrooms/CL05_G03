"""
Zone 1 Power Consumption Prediction — GitHub Actions Pipeline
COS40007 AI Engineering - Group CL05_G03 - Task 02

Trains a Random Forest regressor on engineered features and produces the
artefacts required by the GitHub Actions workflow:
  train_set.csv        — training split (features + target)
  test_set.csv         — testing split (features + target)
  metrics.txt          — performance report
  model_results.png    — performance visualisation
  best_model.joblib    — deployment bundle (model + schema + thresholds)

Random Forest is selected because the team's separate model comparison
(all_model_test.py) confirmed it as the best-performing model
for this regression task.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================================
# 2. SCHEMA & CONSTANTS
# ============================================================================
SCHEMA_VERSION = "1.0"

# Map raw CSV column names to clean internal names.
# Fixes the double-space bug in 'Zone 2  Power Consumption' / 'Zone 3  Power Consumption'.
RAW_COLUMN_MAP = {
    'DateTime':                   'datetime',
    'Temperature':                'temperature',
    'Humidity':                   'humidity',
    'Wind Speed':                 'wind_speed',
    'general diffuse flows':      'general_diffuse',
    'diffuse flows':              'diffuse',
    'Zone 1 Power Consumption':   'zone1_power',
    'Zone 2  Power Consumption':  'zone2_power',
    'Zone 3  Power Consumption':  'zone3_power',
}

# Engineered feature set (in the order the API will expect)
FEATURE_COLUMNS = [
    'zone2_power',
    'zone3_power',
    'power_sum_23',
    'hour_sin',
    'hour_cos',
    'temperature',
    'humidity',
    'hdd',
    'cdd',
]
TARGET_COLUMN = 'zone1_power'


# ============================================================================
# 3. PURE FUNCTIONS
# ============================================================================
def load_and_normalize(csv_path):
    """Load the raw CSV and clean column names + datetime parsing."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RAW_COLUMN_MAP)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
    return df


def engineer_features(df):
    """Add 5 engineered features. Pure function — no side effects."""
    df = df.copy()
    hour = df['datetime'].dt.hour

    # Cyclical hour encoding so 23:00 and 00:00 are correctly close
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Aggregated related-zone demand
    df['power_sum_23'] = df['zone2_power'] + df['zone3_power']

    # Energy-domain heating/cooling degree-day proxies
    df['hdd'] = (18 - df['temperature']).clip(lower=0)
    df['cdd'] = (df['temperature'] - 24).clip(lower=0)

    return df


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================
print("=" * 70)
print("Zone 1 Power Consumption — Random Forest Training Pipeline")
print("Group CL05_G03 — COS40007 AI Engineering")
print("=" * 70)

# ----- STEP 1: LOAD DATA -----
print("\n[STEP 1/6] Loading dataset...")
df = load_and_normalize('power_consumption.csv')
print(f"  >> Dataset loaded successfully: {len(df)} rows, {df.shape[1]} columns")

# ----- STEP 2: FEATURE ENGINEERING -----
print("\n[STEP 2/6] Engineering features...")
df = engineer_features(df)
print("  >> Features engineered successfully:")
print("     - hour_sin, hour_cos    (cyclical time encoding)")
print("     - power_sum_23          (Zone 2 + Zone 3 aggregation)")
print("     - hdd                   (Heating Degree Day proxy)")
print("     - cdd                   (Cooling Degree Day proxy)")

# Sample for fast pipeline runs on the GitHub Actions free runner
df_sample = df.sample(n=5000, random_state=42).reset_index(drop=True)
X = df_sample[FEATURE_COLUMNS]
y = df_sample[TARGET_COLUMN]
print(f"  >> Sampled {len(df_sample)} rows for training")

# ----- STEP 3: TRAIN/TEST SPLIT -----
print("\n[STEP 3/6] Splitting data into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  >> Split successful: {len(X_train)} train rows | {len(X_test)} test rows")

# Save splits to CSV so the GitHub Actions run is reproducible and auditable.
# Anyone downloading the artefacts can see the exact data the model trained on.
train_set = X_train.copy()
train_set[TARGET_COLUMN] = y_train.values
train_set.to_csv('train_set.csv', index=False)
print(f"  >> train_set.csv saved successfully ({len(train_set)} rows)")

test_set = X_test.copy()
test_set[TARGET_COLUMN] = y_test.values
test_set.to_csv('test_set.csv', index=False)
print(f"  >> test_set.csv saved successfully ({len(test_set)} rows)")

# ----- STEP 4: MODEL TRAINING -----
print("\n[STEP 4/6] Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
print("  >> Model trained successfully (Random Forest, 100 estimators)")

# ----- STEP 5: PREDICTIONS & EVALUATION -----
print("\n[STEP 5/6] Evaluating model on test set...")
y_pred = model.predict(X_test)

mae  = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))
print("  >> Predictions generated successfully")
print(f"     MAE  = {mae:.2f} W")
print(f"     RMSE = {rmse:.2f} W")
print(f"     R2   = {r2:.4f}")

# ----- STEP 6: BUILD DEPLOYMENT BUNDLE -----
print("\n[STEP 6/6] Building deployment bundle...")
quantiles = np.quantile(y_train, [0.25, 0.5, 0.75])
category_thresholds = {
    'low':    float(quantiles[0]),
    'medium': float(quantiles[1]),
    'high':   float(quantiles[2]),
}

feature_ranges = {
    col: {
        'min':  float(X_train[col].min()),
        'max':  float(X_train[col].max()),
        'mean': float(X_train[col].mean()),
    }
    for col in FEATURE_COLUMNS
}

deployment_bundle = {
    'schema_version':       SCHEMA_VERSION,
    'trained_at':           datetime.now().isoformat(timespec='seconds'),
    'model':                model,
    'model_name':           'Random Forest',
    'feature_columns':      FEATURE_COLUMNS,
    'feature_ranges':       feature_ranges,
    'category_thresholds':  category_thresholds,
    'training_metadata': {
        'n_train':    len(X_train),
        'n_test':     len(X_test),
        'n_features': len(FEATURE_COLUMNS),
        'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2},
    },
}
print("  >> Deployment bundle assembled successfully")


# ============================================================================
# 5. CONSOLE SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"Model: Random Forest Regressor ({100} estimators)")
print(f"  MAE  (Mean Absolute Error):     {mae:.2f} W")
print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} W")
print(f"  R2   (Coefficient of Determ.):  {r2:.4f}")
print()
print("Demand category thresholds (equal-frequency binning):")
print(f"  Low       : <= {quantiles[0]:.0f} W")
print(f"  Medium    : <= {quantiles[1]:.0f} W")
print(f"  High      : <= {quantiles[2]:.0f} W")
print(f"  Very High : >  {quantiles[2]:.0f} W")
print("=" * 70)


# ============================================================================
# 6. SAVE ARTEFACTS
# ============================================================================
print("\nSaving artefacts for GitHub Actions upload...")

# 6a. metrics.txt
with open('metrics.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ZONE 1 POWER CONSUMPTION — RANDOM FOREST MODEL\n")
    f.write("Group CL05_G03 — COS40007 AI Engineering\n")
    f.write(f"Trained at: {deployment_bundle['trained_at']}\n")
    f.write(f"Schema version: {SCHEMA_VERSION}\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples:  {len(X_test)}\n")
    f.write(f"Number of features: {len(FEATURE_COLUMNS)}\n\n")

    f.write("Engineered feature set:\n")
    for fc in FEATURE_COLUMNS:
        f.write(f"  - {fc}\n")
    f.write("\n")

    f.write("Performance Metrics:\n")
    f.write(f"  MAE  (Mean Absolute Error):     {mae:.2f} W\n")
    f.write(f"  RMSE (Root Mean Squared Error): {rmse:.2f} W\n")
    f.write(f"  R2   (Coefficient of Determ.):  {r2:.4f}\n\n")

    f.write("=" * 70 + "\n")
    f.write("DEMAND CATEGORY THRESHOLDS (equal-frequency binning)\n")
    f.write("=" * 70 + "\n")
    f.write(f"  Low       : <= {category_thresholds['low']:.2f} W\n")
    f.write(f"  Medium    : <= {category_thresholds['medium']:.2f} W\n")
    f.write(f"  High      : <= {category_thresholds['high']:.2f} W\n")
    f.write(f"  Very High : >  {category_thresholds['high']:.2f} W\n")
    f.write("=" * 70 + "\n")
print("  >> metrics.txt created successfully")


# 6b. model_results.png — 2-panel figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: predicted vs actual scatter
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.4, c='steelblue', edgecolors='k', s=25)
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Zone 1 Power (W)')
ax.set_ylabel('Predicted Zone 1 Power (W)')
ax.set_title(
    f'Predicted vs Actual: Random Forest\n'
    f'R2 = {r2:.4f}, MAE = {mae:.0f} W',
    fontweight='bold'
)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

# Panel 2: feature importance
ax = axes[1]
imps = model.feature_importances_
order = np.argsort(imps)
ax.barh([FEATURE_COLUMNS[i] for i in order], imps[order],
        color='seagreen', edgecolor='black')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('model_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("  >> model_results.png created successfully")


# 6c. Deployment bundle (model + schema + thresholds + metadata)
joblib.dump(deployment_bundle, 'best_model.joblib')
print("  >> best_model.joblib saved successfully")

print("\n" + "=" * 70)
print("ALL ARTEFACTS CREATED SUCCESSFULLY")
print("=" * 70)
print("Output files:")
print("  - train_set.csv          (training data split)")
print("  - test_set.csv           (testing data split)")
print("  - metrics.txt            (performance report)")
print("  - model_results.png      (performance visualisation)")
print("  - best_model.joblib      (deployment bundle)")
print()
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

#test1 github automation

