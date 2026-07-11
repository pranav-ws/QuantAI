import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
import joblib
import os

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Same feature set as Random Forest — ensures fair comparison
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

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier.
    Key advantages over Random Forest:
      - Gradient boosting: each tree corrects errors of the previous
      - Built-in regularisation: L1 + L2 to prevent overfitting
      - Better handling of class imbalance via scale_pos_weight
    """
    # Calculate class weight for imbalanced UP/DOWN split
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators      = 500,       # 500 boosting rounds
        max_depth         = 5,         # shallower than RF to avoid overfit
        learning_rate     = 0.05,      # slow learning = better generalisation
        subsample         = 0.8,       # 80% of rows per tree
        colsample_bytree  = 0.8,       # 80% of features per tree
        min_child_weight  = 10,        # min samples per leaf
        gamma             = 0.1,       # min loss reduction to split
        scale_pos_weight  = scale,     # handles UP/DOWN imbalance
        use_label_encoder = False,
        eval_metric       = 'logloss',
        random_state      = 42,
        n_jobs            = -1,        # use all CPU cores
        verbosity         = 0
    )

    # Early stopping — stop if validation loss doesn't improve for 30 rounds
    split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    model.fit(
        X_tr, y_tr,
        eval_set              = [(X_val, y_val)],
        verbose               = False
    )

    return model

def evaluate_model(model, X_test, y_test, ticker):
    """Prints full evaluation metrics — same format as RF for easy comparison."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  XGBoost Results — {ticker}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.2%}")
    print(f"  Precision : {prec:.2%}")
    print(f"  Recall    : {rec:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'':10} Pred DOWN  Pred UP")
    print(f"  Actual DOWN  {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"  Actual UP    {cm[1][0]:>6}     {cm[1][1]:>6}")
    print(f"{'='*50}\n")

    return acc, y_pred, y_proba

def get_feature_importance(model, top_n=10):
    """Returns top N features by XGBoost gain score."""
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    return importance.head(top_n)

def save_model(model, ticker):
    """Saves XGBoost model to disk."""
    path = os.path.join(MODELS_DIR, f'{ticker}_xgb_model.pkl')
    joblib.dump(model, path)
    print(f"  💾 XGBoost model saved → {path}")

def load_model(ticker):
    """Loads saved XGBoost model from disk."""
    path = os.path.join(MODELS_DIR, f'{ticker}_xgb_model.pkl')
    return joblib.load(path)
