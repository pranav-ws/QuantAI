import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Features the model will learn from (exclude price & target columns)
FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'EMA_9', 'EMA_21',
    'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
    'ATR_14',
    'OBV', 'Volume_Ratio',
    'Daily_Return', 'Return_5d', 'Return_20d',
    'High_Low_Range', 'Close_vs_SMA20', 'Close_vs_SMA50'
]

def prepare_data(df):
    """Splits DataFrame into features (X) and target (y)."""
    X = df[FEATURE_COLS].copy()
    y = df['Target'].copy()
    return X, y

def time_series_split(X, y, test_size=0.2):
    """
    Splits data CHRONOLOGICALLY — never randomly!
    Train on past, test on future. This is critical for time-series.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train: {len(X_train)} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"  Test:  {len(X_test)} rows  ({X_test.index[0].date()} → {X_test.index[-1].date()})")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,       # 200 decision trees
        max_depth=8,            # prevent overfitting
        min_samples_split=20,   # need 20 samples to split a node
        min_samples_leaf=10,    # each leaf needs 10 samples
        class_weight='balanced',# handle UP/DOWN imbalance
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, ticker):
    """Prints full evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # confidence for UP

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Model Results — {ticker}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.2%}  (correct UP/DOWN calls)")
    print(f"  Precision : {prec:.2%}  (when it says UP, how often right?)")
    print(f"  Recall    : {rec:.2%}  (of all UP days, how many caught?)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['DOWN', 'UP']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'':10} Pred DOWN  Pred UP")
    print(f"  Actual DOWN  {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"  Actual UP    {cm[1][0]:>6}     {cm[1][1]:>6}")
    print(f"{'='*50}\n")

    return acc, y_pred, y_proba

def get_feature_importance(model, top_n=10):
    """Returns top N most important features."""
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    return importance.head(top_n)

def save_model(model, ticker):
    """Saves trained model to disk."""
    path = os.path.join(MODELS_DIR, f'{ticker}_rf_model.pkl')
    joblib.dump(model, path)
    print(f"  💾 Model saved → {path}")

def load_model(ticker):
    """Loads a saved model from disk."""
    path = os.path.join(MODELS_DIR, f'{ticker}_rf_model.pkl')
    return joblib.load(path)
