"""
Pytest unit tests for classifier accuracy, F1-score, and cross-validation integrity.
"""

import os
import pytest
import pandas as pd
from src.pipeline import CellularTriagePipeline

@pytest.fixture(scope="module")
def pipeline_metrics():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "cellular_logs.csv")
    pipeline = CellularTriagePipeline(data_path)
    return pipeline.run()

def test_dataset_sample_size(pipeline_metrics):
    assert pipeline_metrics["total_logs"] == 1400, "Dataset should contain exactly 1,400 logs"

def test_ml_accuracy_and_f1(pipeline_metrics):
    assert pipeline_metrics["ml_accuracy"] >= 0.98, "ML Test accuracy must be >= 98%"
    assert pipeline_metrics["weighted_f1"] >= 0.98, "Weighted F1 score must be >= 0.98"

def test_ml_outperforms_regex_baseline(pipeline_metrics):
    assert pipeline_metrics["accuracy_gain"] > 10.0, "ML model should achieve > 10% gain over regex"
    assert pipeline_metrics["rule_based_accuracy"] < pipeline_metrics["ml_accuracy"]

def test_cross_validation_stability(pipeline_metrics):
    assert pipeline_metrics["cv_f1_mean"] >= 0.95, "5-Fold CV F1 mean must be >= 0.95"
    assert pipeline_metrics["cv_f1_std"] < 0.05, "5-Fold CV standard deviation must be small"
