import pandas as pd
import numpy as np
import re  # <-- FIXED: Added the missing import here!

def profile_attribute(df: pd.DataFrame, attr_name: str) -> dict:
    """
    Analyzes a given canonical column vector to calculate coverage metrics,
    frequency/density distributions, and structural data anomalies.
    """
    if attr_name not in df.columns:
        return {
            "status": "empty",
            "populated": 0,
            "null_pct": "100.0%",
            "verdict": "info",
            "distributions": [],
            "anomalies": ["Attribute missing from active DataFrame collection framework."]
        }
        
    series = df[attr_name]
    total_len = len(series)
    
    # Filter missing value sequences safely
    non_null_mask = series.notna() & (series != "-") & (series != "") & (series.astype(str).str.lower() != "<na>")
    populated_series = series[non_null_mask]
    populated_count = len(populated_series)
    
    if populated_count == 0:
        return {
            "status": "empty",
            "populated": 0,
            "null_pct": "100.0%",
            "verdict": "info",
            "distributions": [],
            "anomalies": ["All row values are entirely unpopulated or unmapped in this capture slice."]
        }
        
    null_count = total_len - populated_count
    null_pct = f"{(null_count / total_len) * 100:.1f}%"
    
    anomalies = []
    verdict = "ok"
    chart_data = []

    # Spec bounds definition helper
    VALID_LOG_LEVELS = {str(i) for i in range(8)}
    VALID_CATEGORIES = {str(i) for i in range(11)}
    VALID_EVENT_TYPES = {str(i) for i in range(5)}
    VALID_PHASES = {str(i) for i in range(10)}
    VALID_STATES = {str(i) for i in range(12)}
    VALID_SUBSTATES = {str(i) for i in range(10)}
    SPEC_ACTIVITIES = {"api power on", "dds rules validate", "dds infusion control", "fsm activate", "pressure sampling"}

    # --- F01: seq_id Assessment ---
    if attr_name == "seq_id":
        try:
            numeric_seq = pd.to_numeric(populated_series, errors='coerce')
            if numeric_seq.isna().any():
                anomalies.append("Non-numeric elements discovered inside the sequence ID stream.")
                verdict = "bad"
            else:
                diffs = numeric_seq.diff().dropna()
                if (diffs <= 0).any():
                    anomalies.append("Sequence regressions or repeating counter packets detected.")
                    verdict = "bad"
                if (diffs > 1).any():
                    anomalies.append("Sequence sequence counter break/gap detected. Missing records.")
                    verdict = "warn"
            vc = populated_series.value_counts().head(6)
            chart_data = [{"label": str(k), "count": int(v)} for k, v in vc.items()]
        except Exception:
            verdict = "bad"

    # --- F02: timestamp Assessment ---
    elif attr_name == "timestamp":
        try:
            numeric_ts = pd.to_numeric(populated_series, errors='coerce')
            if numeric_ts.isna().any():
                anomalies.append("Non-numeric data found inside timestamp telemetry vector.")
                verdict = "bad"
            else:
                t_diffs = numeric_ts.diff().dropna()
                if (t_diffs < 0).any():
                    anomalies.append("Time inversion anomaly: Decreasing sequential values found.")
                    verdict = "bad"
            vc = populated_series.value_counts().head(6)
            chart_data = [{"label": f"{str(k)} ms", "count": int(v)} for k, v in vc.items()]
        except Exception:
            verdict = "bad"

    # --- F03: event_code Assessment ---
    elif attr_name == "event_code":
        invalid_mask = ~populated_series.astype(str).str.contains(r'S\d+', regex=True)
        if invalid_mask.any():
            anomalies.append(f"Discovered {invalid_mask.sum()} entries missing valid structural 'S<num>' prefix templates.")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": str(k), "count": int(v)} for k, v in vc.items()]

    # --- F05: log_level Range Check ---
    elif attr_name == "log_level":
        cleaned = populated_series.astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
        out_of_bounds = ~cleaned.isin(VALID_LOG_LEVELS)
        if out_of_bounds.any():
            anomalies.append(f"Found {out_of_bounds.sum()} logs operating outside standard RFC 5424 levels (0-7).")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": f"Level {k}", "count": int(v)} for k, v in vc.items()]

    # --- F06: event_category Range Check ---
    elif attr_name == "event_category":
        cleaned = populated_series.astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
        out_of_bounds = ~cleaned.isin(VALID_CATEGORIES)
        if out_of_bounds.any():
            anomalies.append(f"Category code domain overflow observed ({out_of_bounds.sum()} records past limit 10).")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": f"Cat {k}", "count": int(v)} for k, v in vc.items()]

    # --- F07: event_type Range Check ---
    elif attr_name == "event_type":
        cleaned = populated_series.astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
        out_of_bounds = ~cleaned.isin(VALID_EVENT_TYPES)
        if out_of_bounds.any():
            anomalies.append(f"Enum variation bounds violated ({out_of_bounds.sum()} entries past type index 4).")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": f"Type {k}", "count": int(v)} for k, v in vc.items()]

    # --- F08: phase Range Check ---
    elif attr_name == "phase":
        cleaned = populated_series.astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
        out_of_bounds = ~cleaned.isin(VALID_PHASES)
        if out_of_bounds.any():
            anomalies.append(f"Lifecycle state tracking indexing error detected on {out_of_bounds.sum()} rows.")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": f"Phase {k}", "count": int(v)} for k, v in vc.items()]

    # --- F09: state Range Check ---
    elif attr_name == "state":
        cleaned = populated_series.astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
        out_of_bounds = ~cleaned.isin(VALID_STATES)
        if out_of_bounds.any():
            anomalies.append(f"FSM Machine out of core operational bounds across {out_of_bounds.sum()} logs.")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": f"State {k}", "count": int(v)} for k, v in vc.items()]

    # --- F10: substate Range Check ---
    elif attr_name == "substate":
        cleaned = populated_series.astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
        out_of_bounds = ~cleaned.isin(VALID_SUBSTATES)
        if out_of_bounds.any():
            anomalies.append(f"FSM secondary substate range boundary overflow observed.")
            verdict = "bad"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": f"Sub {k}", "count": int(v)} for k, v in vc.items()]

    # --- F11: activity Token Validation Check ---
    elif attr_name == "activity":
        clean_act = populated_series.astype(str).str.strip().str.lower()
        invalid_tokens = ~clean_act.isin(SPEC_ACTIVITIES)
        if invalid_tokens.any():
            anomalies.append(f"Custom unlisted trace activities found ({invalid_tokens.sum()} non-spec names).")
            verdict = "warn"
        vc = populated_series.value_counts().head(6)
        chart_data = [{"label": str(k), "count": int(v)} for k, v in vc.items()]

    # --- F12: depth Assessment Profiling Logic ---
    elif attr_name == "depth":
        try:
            numeric_depth = pd.to_numeric(populated_series, errors='coerce').dropna()
            if (numeric_depth > 12).any():
                anomalies.append(f"Deep stack execution nesting observed (Max: {int(numeric_depth.max())}). Vector risk.")
                verdict = "warn"
            vc = populated_series.value_counts().head(6)
            chart_data = [{"label": f"Depth {k}", "count": int(v)} for k, v in vc.items()]
        except Exception:
            verdict = "warn"

    # --- Categorical Attr Variables Profiling Logic (Fallback) ---
    else:
        value_counts = populated_series.value_counts()
        if len(value_counts) == 1:
            anomalies.append("Static invariant feature value recorded persistently across all frame arrays.")
            verdict = "warn"
        top_items = value_counts.head(6)
        chart_data = [{"label": str(k), "count": int(v)} for k, v in top_items.items()]
        
    if not anomalies:
        anomalies.append("Baseline structural profile integrity checks passed gracefully.")
        
    return {
        "status": "success",
        "populated": int(populated_count),
        "null_pct": null_pct,
        "verdict": verdict,
        "distributions": chart_data,
        "anomalies": anomalies
    }