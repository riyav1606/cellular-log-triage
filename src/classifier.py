"""
Supervised machine learning classifiers for cellular validation log failure triage.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

class CellularLogClassifier:
    """Manages training, cross-validation, and inference across ML models."""

    def __init__(self, model_type: str = "LinearSVC"):
        self.model_type = model_type
        if model_type == "LinearSVC":
            self.model = LinearSVC(C=1.0, random_state=42, dual=False)
        elif model_type == "LogisticRegression":
            self.model = LogisticRegression(C=2.0, max_iter=500, random_state=42)
        elif model_type == "MultinomialNB":
            self.model = MultinomialNB(alpha=0.1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "model_type": self.model_type,
            "accuracy": float(acc),
            "weighted_f1": float(f1),
            "report": report
        }

    def cross_validate(self, X, y, cv: int = 5) -> Tuple[float, float]:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring="f1_weighted")
        return float(scores.mean()), float(scores.std())

    def predict(self, X_vector):
        return self.model.predict(X_vector)
