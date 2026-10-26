"""
End-to-end cellular validation log failure triage orchestrator and benchmark suite.
"""

import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .feature_extractor import CellularLogFeatureExtractor
from .classifier import CellularLogClassifier
from .parser import RegexTriageBaseline
from .clustering import NovelFailureClusterer
from .reporter import TriageReporter, ROOT_CAUSE_MAPPING

class CellularTriagePipeline:
    """Coordinates log ingestion, feature extraction, ML training, baseline benchmarking, and reporting."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.extractor = CellularLogFeatureExtractor()
        self.classifier = CellularLogClassifier(model_type="LinearSVC")
        self.clusterer = NovelFailureClusterer()
        self.df = None
        self.metrics = {}

    def run(self, reports_dir: str = None):
        self.df = pd.read_csv(self.data_path)
        raw_logs = self.df["log_message"].tolist()
        labels = self.df["label"].values

        # 1. Feature extraction
        X = self.extractor.fit_transform(raw_logs)

        # 2. Train/Test Split
        train_idx, test_idx = train_test_split(
            np.arange(len(self.df)), test_size=0.20, random_state=42, stratify=labels
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # 3. Model Training & Evaluation
        self.classifier.train(X_train, y_train)
        eval_res = self.classifier.evaluate(X_test, y_test)
        cv_mean, cv_std = self.classifier.cross_validate(X, labels, cv=5)

        # 4. Regex Baseline Comparator
        test_messages = [raw_logs[i] for i in test_idx]
        rule_preds = [RegexTriageBaseline.predict(msg) for msg in test_messages]
        rule_acc = float(np.mean(np.array(rule_preds) == y_test))

        # 5. Measure Latency
        t0 = time.perf_counter()
        N_LATENCY = 200
        for i in range(N_LATENCY):
            sample_vec = self.extractor.transform([test_messages[i % len(test_messages)]])
            _ = self.classifier.predict(sample_vec)
        t1 = time.perf_counter()
        per_log_ms = ((t1 - t0) / N_LATENCY) * 1000

        # Store Metrics
        self.metrics = {
            "total_logs": len(self.df),
            "ml_accuracy": eval_res["accuracy"],
            "weighted_f1": eval_res["weighted_f1"],
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std,
            "rule_based_accuracy": rule_acc,
            "accuracy_gain": (eval_res["accuracy"] - rule_acc) * 100,
            "latency_ms": per_log_ms
        }

        # 6. Generate Reports if requested
        if reports_dir:
            os.makedirs(reports_dir, exist_ok=True)
            sample_results = []
            for i in range(min(50, len(test_messages))):
                sample_vec = self.extractor.transform([test_messages[i]])
                pred = self.classifier.predict(sample_vec)[0]
                sample_results.append({
                    "log_id": int(self.df.iloc[test_idx[i]]["id"]),
                    "log_message": test_messages[i],
                    "predicted_label": pred,
                    "actual_label": y_test[i],
                    "root_cause_summary": ROOT_CAUSE_MAPPING.get(pred, "Unknown")
                })
            json_out = os.path.join(reports_dir, "sample_triage_report.json")
            md_out = os.path.join(reports_dir, "sample_triage_report.md")
            TriageReporter.generate_json_report(sample_results, self.metrics, json_out)
            TriageReporter.generate_markdown_report(self.metrics, md_out)

        return self.metrics
