"""
VERA Engine — Feature Assessment Matrix
Stores metadata dictionary registries, algorithmic method configurations, 
and automated pipeline inference schemas for Stage 3 evaluation.
"""
import pandas as pd
import numpy as np

# Dynamic metadata documentation matching the frontend 'About Feature' card panel
FEATURE_META_DICTIONARY = {
    "F1": "Time between consecutive records and the resulting emit rate — the throughput backbone of the capture.",
    "F4": "Analyzes the frequency and intensity of system alerts to detect abnormal clusters or potential system overloads.",
    "F5": "Monitors the duration spent in each operational state to ensure system processes remain stable and repeatable.",
    "F8": "Tracks the rhythmic cadence of actions over time to catch irregular execution patterns or intermittent stalls.",
    "F9": "Measures changes in accuracy and data consistency across operational sequences to flag calibration errors or degradation.",
    "F24": "Validates data metrics generated during actions against designated baselines to confirm input processing remains uniform.",
    "Fosc": "Evaluates compliance with target operating pressures to confirm oscillation waves stay safely within specified bounds.",
    "F19": "Evaluates the breadth of testing scenarios across all system stages to identify any unverified operational paths.",
    "F21": "Verifies that state modifications follow approved logic flows to block invalid configurations or abrupt exits."
}


def get_f1_assessment(df: pd.DataFrame, chart_type: str = "line") -> dict:
    """
    Stage 3 Feature Calculation Layer (F1 - Throughput Backbone):
    Accepts the standard canonical workspace DataFrame directly, computes 
    consecutive record timelines, and maps metrics for chart layouts.
    """
    # Validation guard matching attribute columns
    if df is None or df.empty or "timestamp" not in df.columns or "seq_id" not in df.columns:
        return {
            
            "type": "empty",
            "verdict": "INFO",
            "metric": "0.000s",
            "x": [],
            "y": [],
            "inference": "Insufficient canonical parameters or data rows found to build feature metrics."
        }

    try:
        # Extract numeric tracks safely
        times = pd.to_numeric(df["timestamp"], errors='coerce').dropna().to_numpy()
    except Exception:
        return {
            "type": "empty",
            "verdict": "INFO",
            "metric": "0.000s",
            "x": [],
            "y": [],
            "inference": "Data formatting exception encountered during real-time timeline parsing."
        }
    
    # Ensure there are at least two samples to compute an interval sequence
    if len(times) < 2:
        return {
            "type": "empty",
            "verdict": "INFO",
            "metric": "Single record",
            "x": [],
            "y": [0.0],
            "inference": "Single log record captured. Baseline intervals require consecutive frames."
        }

    # Calculate consecutive differences (inter-arrival values)
    deltas = np.diff(times)
    median_delta = float(np.median(deltas))
    max_delta = float(np.max(deltas))
    
    # Establish dynamic operational thresholds to notify of delays
    if max_delta > (median_delta * 4.0):
        verdict = "WARNING"
        inference = f"Micro-stalls or jitter detected in throughput backbone. Maximum execution gap hit {max_delta:.3f}s."
    else:
        verdict = "VERIFIED OK"
        inference = f"Throughput backbone is stable. Inter-arrival intervals correspond perfectly with default real-time capture rates."

    # Process rendering layout variants using the x and y structure
    if chart_type == "distribution":
        counts, bin_edges = np.histogram(deltas, bins=min(12, len(deltas)))
        chart_x = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}s" for i in range(len(counts))]
        chart_y = [int(c) for c in counts]
        return {
            "type": "distribution",
            "verdict": verdict,
            "metric": f"{median_delta:.3f}s median",
            "x": chart_x,
            "y": chart_y,
            "inference": inference
        }
    
    else:  # Default Fallback Option: Line Trace Sequence Chart
        # FIX: Generate a sequential index array just like attribute_assessment.py does
        chart_x = np.arange(len(deltas)).tolist()
        chart_y = [round(float(d), 4) for d in deltas]
        return {
            "type": "line",
            "verdict": verdict,
            "metric": f"{median_delta:.3f}s median",
            "x": chart_x,
            "y": chart_y,
            "inference": inference
        }