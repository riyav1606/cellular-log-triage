"""
Pytest tests for report generation integrity.
"""

import os
import json
import tempfile
from src.reporter import TriageReporter

def test_json_and_markdown_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        md_path = os.path.join(tmpdir, "report.md")

        mock_results = [{
            "log_id": 1,
            "log_message": "Sample log",
            "predicted_label": "PHY_SYNC_LOSS",
            "actual_label": "PHY_SYNC_LOSS",
            "root_cause_summary": "Doppler shift"
        }]
        mock_metrics = {
            "ml_accuracy": 1.0,
            "weighted_f1": 1.0,
            "rule_based_accuracy": 0.843,
            "accuracy_gain": 15.7,
            "latency_ms": 0.45
        }

        TriageReporter.generate_json_report(mock_results, mock_metrics, json_path)
        TriageReporter.generate_markdown_report(mock_metrics, md_path)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_logs_analyzed"] == 1
            assert data["evaluation_metrics"]["ml_accuracy"] == 1.0
