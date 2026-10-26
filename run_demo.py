"""
CLI Demo Script for Cellular Log Failure Triage Assistant.
"""

import os
from src.pipeline import CellularTriagePipeline

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, "data", "cellular_logs.csv")
    reports_dir = os.path.join(root_dir, "reports")

    print("================================================================================")
    print("   AI-POWERED CELLULAR TEST LOG FAILURE TRIAGE ASSISTANT")
    print("================================================================================")
    print(f"[*] Ingesting 3GPP cellular validation logs from: {data_path}")

    pipeline = CellularTriagePipeline(data_path)
    metrics = pipeline.run(reports_dir=reports_dir)

    print(f"[+] Total Logs Processed: {metrics['total_logs']}")
    print(f"[+] Rule-Based Regex Baseline Accuracy: {metrics['rule_based_accuracy']*100:.2f}%")
    print(f"[+] Supervised LinearSVC Test Accuracy:  {metrics['ml_accuracy']*100:.2f}%")
    print(f"[+] Weighted F1-Score:                   {metrics['weighted_f1']:.4f}")
    print(f"[+] 5-Fold Stratified CV F1 Mean:        {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
    print(f"[+] Measured Accuracy Gain over Regex:   +{metrics['accuracy_gain']:.2f}%")
    print(f"[+] Per-Log Inference Latency:           {metrics['latency_ms']:.3f} ms (<1ms target)")
    print(f"[+] Diagnostic reports generated in:     {reports_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
