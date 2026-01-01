# services/health_ml.py

import os
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

from services.health_service import fetch_health_records
from services.ai_service import infer_health_status

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "health_model.pkl")


def _ensure_model_dir():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def train_health_model(test_size: float = 0.2, random_state: int = 42) -> Optional[str]:
    """Train a simple text+categorical classifier to predict health status.
    Labels are bootstrapped from current records via `infer_health_status`.
    Returns path to saved model.
    """
    rows = fetch_health_records()
    if not rows:
        return None

    # health_records columns: (id, tag, species, record_date, diagnosis, treatment, medication, dosage, vet, lab_result, severity, notes, next_check_date, withdrawal_end_date)
    texts = []
    severities = []
    labels = []
    for r in rows:
        diagnosis, treatment, lab_result, severity, notes = r[4], r[5], r[9], r[10], r[11]
        next_check, withdrawal_end = r[12], r[13]
        label = infer_health_status(diagnosis, treatment, severity, lab_result, notes, next_check, withdrawal_end)
        # Create a combined text field
        text = " ".join([str(x or "") for x in [diagnosis, treatment, lab_result, notes]]).strip()
        texts.append(text)
        severities.append(severity or "")
        labels.append(label)

    X = {
        "text": texts,
        "severity": severities,
    }

    X_text = [[t] for t in texts]  # ColumnTransformer expects 2D
    X_sev = [[s] for s in severities]

    pre = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000), 0),
            ("sev", OneHotEncoder(handle_unknown="ignore"), 1),
        ]
    )

    # Custom wrapper to feed both columns
    class DualInput:
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            # X is dict, build list of [text, severity]
            return [[X["text"][i], X["severity"][i]] for i in range(len(X["text"]))]

    pipeline = Pipeline([
        ("dual", DualInput()),
        ("features", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state, stratify=labels)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    try:
        print(classification_report(y_test, preds))
    except Exception:
        pass

    _ensure_model_dir()
    joblib.dump(pipeline, MODEL_PATH)
    return MODEL_PATH


def predict_health_status(diagnosis: str, treatment: str, severity: str, lab_result: str, notes: str) -> Optional[str]:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        return None
    text = " ".join([str(x or "") for x in [diagnosis, treatment, lab_result, notes]]).strip()
    X = {"text": [text], "severity": [severity or ""]}
    try:
        pred = model.predict(X)
        return pred[0]
    except Exception:
        return None
