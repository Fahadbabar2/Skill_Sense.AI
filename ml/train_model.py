"""
Trains an XGBoost classifier to predict skill level from the feature
vector produced by backend/adaptive_engine.py::get_feature_vector.

Run:
    python ml/generate_synthetic_data.py   # first time / to refresh data
    python ml/train_model.py
Produces:
    backend/models/skill_model.pkl
    backend/models/label_encoder.pkl
Also prints accuracy, a classification report, and a confusion matrix
on the held-out test split -- keep this output, it belongs in your
project write-up as evidence the model works.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

FEATURE_ORDER = [
    "overall_accuracy",
    "easy_accuracy",
    "medium_accuracy",
    "hard_accuracy",
    "avg_time_taken",
    "avg_quality_score",
    "total_hints_used",
    "hard_questions_attempted",
    "claimed_experience_years",
    "claimed_skill_count",
]

DATA_PATH = "ml/merged_training_data.csv"
MODEL_DIR = Path("backend/models")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_ORDER]
    y_raw = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 70 / 15 / 15 split as in the project doc: split off test first, then
    # split the remainder into train/val.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
    )  # 0.1765 * 0.85 ~= 0.15 of the original data -> 70/15/15 overall

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    print("Test accuracy:", round((y_pred == y_test).mean(), 3))
    print()
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "skill_model.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print(f"\nSaved model to {MODEL_DIR / 'skill_model.pkl'}")


if __name__ == "__main__":
    main()
