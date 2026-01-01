# ============================================
# PREDICTIVE MAINTENANCE PIPELINE (FULL CODE)
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("G:\\intership project\\maintenance_data.csv")

df = df.rename(columns={
    'Product ID': 'Product_ID',
    'Air temperature [K]': 'air_temp',
    'Process temperature [K]': 'process_temp',
    'Rotational speed [rpm]': 'rot_speed',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'tool_wear',
    'Machine failure': 'failure'
})

df_encoded = df.copy()

print(df.head())
print(df.info())

# ---------------------------
# Feature Selection
# ---------------------------
le = LabelEncoder()
df['Product_ID'] = le.fit_transform(df['Product_ID'])

features = [
    'Product_ID',
    'air_temp',
    'process_temp',
    'rot_speed',
    'torque',
    'tool_wear'
]


X = df[features]
y = df['failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# Model Training
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("AUC Score:", roc_auc)

importances = model.feature_importances_
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

# ============================================
# VISUALIZATIONS
# ============================================
plt.style.use('ggplot')

# ---------------------------
# Scatter: Air Temp vs Tool Wear
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df['air_temp'], df['tool_wear'], s=10, alpha=0.6)
plt.xlabel('Air Temperature')
plt.ylabel('Tool Wear')
plt.title('Air Temperature vs Tool Wear')
plt.show()

# ---------------------------
# Correlation Heatmap
# ---------------------------
plt.figure(figsize=(10, 6))
corr = df[features + ['failure']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis')
plt.title('Correlation Matrix (Features + Failure)')
plt.show()

# ---------------------------
# ROC Curve
# ---------------------------
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Feature Importances
# ---------------------------
feat_imp = pd.Series(importances, index=features).sort_values()

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh')
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ---------------------------
# Confusion Matrix Heatmap
# ---------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=['No Failure','Failure'],
            yticklabels=['No Failure','Failure'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
