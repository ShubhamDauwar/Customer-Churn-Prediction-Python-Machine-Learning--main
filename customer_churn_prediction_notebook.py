# Customer Churn Prediction
# Ready-to-run Python script / notebook-style (.py with cells) for Jupyter or VS Code

# Instructions:
# 1. Download the Telco dataset from Kaggle and place the CSV in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
#    Kaggle dataset page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# 2. Install required packages (if not already):
#    pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
# 3. Open this file as a notebook in VS Code or run in Jupyter.

# %%
# --- Imports ---
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# %%
# --- Config ---
DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# %%
# --- Load Data ---
print('Loading data from:', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Rows, Columns:', df.shape)

# show head
print(df.head())

# %%
# --- Quick Data Overview ---
print('\nData info:')
print(df.info())
print('\nMissing values per column:')
print(df.isnull().sum())

# There may be whitespace in TotalCharges; convert to numeric
# %%
# Clean TotalCharges if necessary
if df['TotalCharges'].dtype == object:
    # Replace spaces with NaN then convert
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    print('\nAfter conversion, missing TotalCharges:', df['TotalCharges'].isnull().sum())

# %%
# Drop customerID (identifier)
df = df.drop(columns=['customerID'])

# %%
# --- Handle missing values ---
# For Telco dataset, only TotalCharges may have missing; fill with median or 0
if df.isnull().sum().sum() > 0:
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# %%
# --- Exploratory Data Analysis (basic) ---
# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Churn percentage
churn_rate = df['Churn'].value_counts(normalize=True).mul(100)
print('\nChurn percentages:\n', churn_rate)

# Numeric feature distributions
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
print('\nNumeric columns:', num_cols)

plt.figure(figsize=(12,8))
sns.pairplot(df[num_cols + ['Churn']], corner=True, hue='Churn')
plt.suptitle('Pairplot of numeric features (may take time)', y=1.02)
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(8,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation matrix (numeric features)')
plt.show()

# %%
# --- Feature Engineering ---
# Convert TotalCharges and MonthlyCharges to floats (they should be)
# Create tenure groups
bins = [0, 12, 24, 48, 60, 72]
labels = ['0-12','12-24','24-48','48-60','60-72']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

# Convert SeniorCitizen from 0/1 to 'Yes'/'No' for consistency
if df['SeniorCitizen'].dtype in [np.int64, np.int32, np.float64]:
    df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

# %%
# --- Encoding categorical variables ---
# Identify categorical columns
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
print('\nCategorical columns:', cat_cols)

# Exclude target
cat_cols_no_target = [c for c in cat_cols if c != 'Churn']

# For binary categorical columns, use LabelEncoder; for others, use get_dummies
binary_cols = [c for c in cat_cols_no_target if df[c].nunique() == 2]
multi_cols = [c for c in cat_cols_no_target if df[c].nunique() > 2]

print('\nBinary cols:', binary_cols)
print('Multi cols:', multi_cols)

le = LabelEncoder()
for c in binary_cols:
    df[c] = le.fit_transform(df[c])

# One-hot encode multi-category columns
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

print('\nAfter encoding, dataset shape:', df.shape)

# %%
# --- Prepare train/test sets ---
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

# Feature scaling for numeric columns
scaler = StandardScaler()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Fit scaler on training numeric features
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

# %%
# --- Helper: evaluation function ---
def eval_model(model, X_test, y_test, display_cm=True):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:,1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print('Accuracy: {:.4f}'.format(acc))
    print('Precision: {:.4f}'.format(prec))
    print('Recall: {:.4f}'.format(rec))
    print('F1-score: {:.4f}'.format(f1))
    if roc_auc:
        print('ROC-AUC: {:.4f}'.format(roc_auc))

    print('\nClassification Report:\n', classification_report(y_test, y_pred))

    if display_cm:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

# %%
# --- Model 1: Logistic Regression (baseline) ---
print('\nTraining Logistic Regression...')
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print('Logistic Regression evaluation:')
eval_model(logreg, X_test, y_test)
joblib.dump(logreg, os.path.join(MODEL_DIR, 'logistic_regression.joblib'))

# %%
# --- Model 2: Random Forest ---
print('\nTraining Random Forest...')
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print('Random Forest evaluation:')
eval_model(rf, X_test, y_test)
joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.joblib'))

# %%
# --- Model 3: XGBoost ---
print('\nTraining XGBoost...')
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_clf.fit(X_train, y_train)
print('XGBoost evaluation:')
eval_model(xgb_clf, X_test, y_test)
joblib.dump(xgb_clf, os.path.join(MODEL_DIR, 'xgboost.joblib'))

# %%
# --- Feature Importance (Random Forest & XGBoost) ---
importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print('\nTop 15 feature importances (Random Forest):')
print(feat_importances.head(15))

plt.figure(figsize=(8,6))
feat_importances.head(15).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 15 Feature Importances (Random Forest)')
plt.show()

# %%
# --- Save best model (choose by validation, here we choose XGBoost as example) ---
best_model = xgb_clf
joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_churn_model.joblib'))
print('\nSaved best model to models/best_churn_model.joblib')

# %%
# --- Quick predict function example ---

def predict_single(sample_dict, model=best_model, scaler_path=os.path.join(MODEL_DIR, 'scaler.joblib')):
    """sample_dict: mapping of feature_name -> value for a single customer (exclude Churn)
       returns predicted probability of churn and class (0/1)
    """
    # Load scaler
    scaler = joblib.load(scaler_path)
    sample_df = pd.DataFrame([sample_dict])

    # Align columns with training data
    missing_cols = set(X.columns) - set(sample_df.columns)
    for c in missing_cols:
        sample_df[c] = 0
    sample_df = sample_df[X.columns]

    # Scale numeric
    sample_df[num_cols] = scaler.transform(sample_df[num_cols])

    proba = model.predict_proba(sample_df)[:,1][0]
    pred = int(proba >= 0.5)
    return {'probability': float(proba), 'prediction': pred}

# Example usage (you must provide proper fields):
# sample = { 'gender': 1, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0, 'tenure': 5, 'MonthlyCharges': 70.35, 'TotalCharges': 350.5, ... }
# print(predict_single(sample))

# End of notebook/script
