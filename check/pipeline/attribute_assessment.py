import pandas as pd
import numpy as np
from pipeline.make_canonical import make_canonical


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fill_rate(series: pd.Series) -> tuple:
    total = len(series)
    if total == 0:
        return 100.0, 0, 0
    null_count = int(series.isna().sum())
    fill_pct = round(((total - null_count) / total) * 100.0, 1)
    return fill_pct, null_count, total


def _verdict_from_fill(fill_pct: float, bad_thresh=80.0, warn_thresh=95.0) -> str:
    if fill_pct < bad_thresh:
        return "bad"
    elif fill_pct < warn_thresh:
        return "warn"
    return "ok"


def _count_status(count: int, warn_limit: int = 1, bad_limit: int = 5) -> str:
    if count >= bad_limit:
        return "bad"
    elif count >= warn_limit:
        return "warn"
    return "ok"


# ---------------------------------------------------------------------------
# assess_sequence_id
# ---------------------------------------------------------------------------

def assess_sequence_id(df: pd.DataFrame) -> dict:
    if "seq_id" not in df.columns or df.empty:
        return {
            "verdict": "info", "metric": 0.0, "unit": "%",
            "note": "No sequence records available",
            "actual": "N/A", "expected": "≥ 99% continuous",
            "inference": "No sequence ID records available for analysis.",
            "rows": [],
            "details": {"duplicates": 0, "gaps": 0, "change_points": 0}
        }

    seqs = pd.to_numeric(df["seq_id"], errors='coerce')
    valid_seqs = seqs.dropna().astype(int).to_numpy()

    if len(valid_seqs) < 2:
        return {
            "verdict": "ok", "metric": 100.0, "unit": "% continuous",
            "note": "Insufficient samples for sequence tracking",
            "actual": "< 2 samples", "expected": "≥ 99% continuous",
            "inference": "Insufficient samples for sequence tracking — minimum 2 records required.",
            "rows": [],
            "details": {"duplicates": 0, "gaps": 0, "change_points": 0}
        }

    deltas = np.diff(valid_seqs)
    duplicates = int(np.sum(deltas == 0))
    gaps = int(np.sum(deltas > 1))
    change_points = int(np.sum(deltas < 0))
    total_steps = len(deltas)
    conform_count = total_steps - (duplicates + gaps)
    conform_pct = (conform_count / total_steps) * 100.0 if total_steps > 0 else 100.0

    verdict = "bad" if conform_pct < 95.0 else ("warn" if conform_pct < 99.0 else "ok")

    rows = [
        {"label": "Continuity Rate",   "actual": f"{round(conform_pct,1)}%", "expected": "≥ 99%",            "status": verdict},
        {"label": "Gap Count",         "actual": str(gaps),                  "expected": "0",                  "status": _count_status(gaps, 1, 5)},
        {"label": "Duplicates",        "actual": str(duplicates),            "expected": "0",                  "status": _count_status(duplicates, 1, 3)},
        {"label": "Change Points",     "actual": str(change_points),         "expected": "0 (mid-session)",    "status": "ok" if change_points == 0 else "warn"},
        {"label": "Total Steps",       "actual": str(total_steps),           "expected": "—",                  "status": "ok"},
    ]

    if verdict == "ok":
        inference = (f"Sequence continuity is healthy at {round(conform_pct,1)}% — "
                     f"{gaps} gap(s) and {duplicates} duplicate(s) detected across {total_steps} steps.")
    elif verdict == "warn":
        inference = (f"Continuity of {round(conform_pct,1)}% is below the 99% threshold — "
                     f"{gaps} missing segments and {duplicates} collision event(s) may indicate buffer overruns.")
    else:
        inference = (f"Critical continuity failure at {round(conform_pct,1)}% — "
                     f"{gaps} sequence gap(s) and {duplicates} duplicate(s) signal significant record loss.")

    return {
        "verdict": verdict,
        "metric": round(conform_pct, 1),
        "unit": "% continuous",
        "note": f"{gaps} missing segments · {duplicates} collisions · {change_points} resets",
        "actual": f"{round(conform_pct,1)}% continuous ({gaps} gaps · {duplicates} dups)",
        "expected": "≥ 99% continuous · 0 gaps · 0 duplicates",
        "inference": inference,
        "rows": rows,
        "details": {"duplicates": duplicates, "gaps": gaps, "change_points": change_points}
    }


# ---------------------------------------------------------------------------
# assess_timestamp
# ---------------------------------------------------------------------------

def assess_timestamp(df: pd.DataFrame) -> dict:
    if "timestamp" not in df.columns or df.empty:
        return {
            "verdict": "info", "metric": 0.0, "unit": "%",
            "note": "No timestamp records available",
            "actual": "N/A", "expected": "100% monotonic",
            "inference": "No timestamp records available for analysis.",
            "rows": [],
            "details": {"mon_violations": 0, "bursts": 0, "starvation": 0}
        }

    times = pd.to_numeric(df["timestamp"], errors='coerce')
    valid_times = times.dropna().to_numpy()

    if len(valid_times) < 2:
        return {
            "verdict": "ok", "metric": 100.0, "unit": "% rhythmic",
            "note": "Insufficient samples for timeline metrics",
            "actual": "< 2 samples", "expected": "100% monotonic",
            "inference": "Insufficient samples for timeline metrics.",
            "rows": [],
            "details": {"mon_violations": 0, "bursts": 0, "starvation": 0}
        }

    deltas = np.diff(valid_times)
    total_intervals = len(deltas)
    mon_violations = int(np.sum(deltas < 0))
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    bursts = starvation = 0
    if std_delta > 0:
        z = (deltas - mean_delta) / std_delta
        bursts = int(np.sum(z < -2.5))
        starvation = int(np.sum(z > 2.5))

    failed = mon_violations + starvation
    healthy_pct = ((total_intervals - failed) / total_intervals if total_intervals > 0 else 1.0) * 100.0

    verdict = "bad" if (mon_violations > 0 or healthy_pct < 92.0) else ("warn" if healthy_pct < 98.0 else "ok")

    rows = [
        {"label": "Rhythmic Health",       "actual": f"{round(healthy_pct,1)}%", "expected": "≥ 98%",    "status": verdict},
        {"label": "Monotonicity Breaks",   "actual": str(mon_violations),         "expected": "0",        "status": "ok" if mon_violations == 0 else "bad"},
        {"label": "Starvation Drops",      "actual": str(starvation),             "expected": "0",        "status": _count_status(starvation, 1, 5)},
        {"label": "Burst Traps",           "actual": str(bursts),                 "expected": "0",        "status": _count_status(bursts, 1, 5)},
        {"label": "Total Intervals",       "actual": str(total_intervals),        "expected": "—",        "status": "ok"},
    ]

    if verdict == "ok":
        inference = (f"Timestamp rhythm is healthy at {round(healthy_pct,1)}% — "
                     f"{mon_violations} chronology break(s) across {total_intervals} intervals.")
    elif verdict == "warn":
        inference = (f"Timestamp health of {round(healthy_pct,1)}% is degraded — "
                     f"{starvation} starvation drop(s) and {bursts} burst trap(s) detected. "
                     f"Jitter may affect phase-timing analytics.")
    else:
        inference = (f"Critical timestamp failures — {mon_violations} chronology break(s) violate monotonicity. "
                     f"Timeline-dependent analytics are unreliable until the source is corrected.")

    return {
        "verdict": verdict,
        "metric": round(healthy_pct, 1),
        "unit": "% rhythmic",
        "note": f"{mon_violations} chronology breaks · {starvation} starvation drops · {bursts} burst traps",
        "actual": f"{round(healthy_pct,1)}% rhythmic ({mon_violations} breaks · {starvation} starves)",
        "expected": "≥ 98% rhythmic · 0 monotonicity breaks",
        "inference": inference,
        "rows": rows,
        "details": {"mon_violations": mon_violations, "bursts": bursts, "starvation": starvation}
    }


# ---------------------------------------------------------------------------
# Remaining attribute assessors
# ---------------------------------------------------------------------------

def assess_event_code(df: pd.DataFrame) -> dict:
    if "event_code" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No event_code records available",
                "actual": "N/A", "expected": "≥ 95% filled", "inference": "No event_code records available.",
                "rows": [], "details": {"null_count": 0, "unique_codes": 0}}
    fill_pct, null_count, total = _fill_rate(df["event_code"])
    unique_codes = int(df["event_code"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct)
    rows = [
        {"label": "Fill Rate",     "actual": f"{fill_pct}%",      "expected": "≥ 95%",  "status": verdict},
        {"label": "Null Entries",  "actual": str(null_count),      "expected": "0",       "status": _count_status(null_count, 1, int(total*0.05)+1)},
        {"label": "Unique Codes",  "actual": str(unique_codes),    "expected": "> 0",     "status": "ok" if unique_codes > 0 else "warn"},
        {"label": "Total Records", "actual": str(total),           "expected": "—",       "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Event code fill rate is healthy at {fill_pct}% — {unique_codes} distinct codes across {total} records."
    elif verdict == "warn":
        inference = f"Event code fill rate of {fill_pct}% is below the 95% threshold — {null_count} null entries may indicate incomplete upstream record capture."
    else:
        inference = f"Critical event code gap — only {fill_pct}% fill rate. {null_count} missing entries block ASC catalogue resolution for those records."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing entries · {unique_codes} unique codes across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_codes} unique)",
        "expected": "≥ 95% fill · all codes registered in ASC catalogue",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_codes": unique_codes}
    }


def assess_extra_srcinfo(df: pd.DataFrame) -> dict:
    if "extra_srcinfo" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No source info records available",
                "actual": "N/A", "expected": "100% resolved", "inference": "No source info records available.",
                "rows": [], "details": {"null_count": 0, "unknown_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["extra_srcinfo"])
    valid = df["extra_srcinfo"].dropna()
    unknown_count = int(valid.astype(str).str.startswith("SRC_UNKNOWN").sum())
    unknown_pct = round((unknown_count / total * 100.0) if total > 0 else 0.0, 1)
    resolved_pct = round(fill_pct - unknown_pct, 1)
    verdict = "bad" if (unknown_pct > 20.0 or fill_pct < 80.0) else ("warn" if (unknown_pct > 5.0 or fill_pct < 95.0) else "ok")
    rows = [
        {"label": "Fill Rate",        "actual": f"{fill_pct}%",        "expected": "100%",     "status": _verdict_from_fill(fill_pct)},
        {"label": "Resolved Sources", "actual": f"{resolved_pct}%",    "expected": "≥ 95%",    "status": "ok" if resolved_pct >= 95 else ("warn" if resolved_pct >= 80 else "bad")},
        {"label": "Unknown Sources",  "actual": str(unknown_count),    "expected": "0",         "status": _count_status(unknown_count, 1, int(total*0.2)+1)},
        {"label": "Null Entries",     "actual": str(null_count),       "expected": "0",         "status": _count_status(null_count, 1, int(total*0.1)+1)},
    ]
    if verdict == "ok":
        inference = f"Source resolution is clean — {resolved_pct}% of CIs resolve to baselined entries. {unknown_count} unresolved source(s) present."
    elif verdict == "warn":
        inference = f"{unknown_count} unresolved source IDs (SRC_UNKNOWN) detected — {unknown_pct}% of records lack CI traceability, which may affect IEC 62304 attribution."
    else:
        inference = f"Source resolution is critically degraded — {unknown_count} unresolved and {null_count} missing entries. CI attribution is unreliable for these records."
    return {
        "verdict": verdict, "metric": round(fill_pct, 1), "unit": "% resolved",
        "note": f"{unknown_count} unresolved sources · {null_count} missing · {total - null_count - unknown_count} clean",
        "actual": f"{resolved_pct}% clean ({unknown_count} SRC_UNKNOWN · {null_count} null)",
        "expected": "100% resolved to baselined CI · 0 unknowns",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unknown_count": unknown_count}
    }


def assess_log_level(df: pd.DataFrame) -> dict:
    if "log_level" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No log_level records available",
                "actual": "N/A", "expected": "100% valid (0-7)", "inference": "No log_level records available.",
                "rows": [], "details": {"null_count": 0, "invalid_count": 0, "critical_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["log_level"])
    valid = df["log_level"].dropna().astype(str)
    invalid_count = int(valid.str.startswith("INVALID").sum())
    critical_count = int(valid.isin(["0", "1", "2", "3"]).sum())
    critical_pct = round((critical_count / total * 100.0) if total > 0 else 0.0, 1)
    invalid_pct = round((invalid_count / total * 100.0) if total > 0 else 0.0, 1)
    verdict = "bad" if (critical_pct > 20.0 or invalid_pct > 10.0 or fill_pct < 80.0) else ("warn" if (critical_pct > 5.0 or invalid_pct > 2.0 or fill_pct < 95.0) else "ok")
    rows = [
        {"label": "Valid Fill Rate",     "actual": f"{fill_pct}%",         "expected": "≥ 95%",    "status": _verdict_from_fill(fill_pct)},
        {"label": "Critical/Error Rate", "actual": f"{critical_pct}%",     "expected": "< 5%",     "status": "ok" if critical_pct < 5 else ("warn" if critical_pct < 20 else "bad")},
        {"label": "Invalid Codes",       "actual": str(invalid_count),     "expected": "0",         "status": _count_status(invalid_count, 1, int(total*0.1)+1)},
        {"label": "Critical Entries",    "actual": str(critical_count),    "expected": "minimal",   "status": "ok" if critical_count == 0 else ("warn" if critical_pct < 5 else "bad")},
    ]
    if verdict == "ok":
        inference = f"Severity levels are healthy — {fill_pct}% valid, {critical_count} critical/error record(s) ({critical_pct}%)."
    elif verdict == "warn":
        inference = f"{critical_count} critical/error entries ({critical_pct}%) detected — review error burst patterns. {invalid_count} invalid severity code(s) present."
    else:
        inference = f"Critical severity anomaly — {critical_pct}% error rate or {invalid_count} invalid level code(s) exceed thresholds. System stability is at risk."
    return {
        "verdict": verdict, "metric": round(fill_pct, 1), "unit": "% valid",
        "note": f"{critical_count} critical/error entries · {invalid_count} invalid codes · {null_count} missing",
        "actual": f"{fill_pct}% valid ({critical_pct}% critical · {invalid_count} invalid codes)",
        "expected": "≥ 95% fill · < 5% critical · 0 invalid codes",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "invalid_count": invalid_count, "critical_count": critical_count}
    }


def assess_event_category(df: pd.DataFrame) -> dict:
    if "event_category" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No event_category records available",
                "actual": "N/A", "expected": "≥ 95% filled", "inference": "No event_category records available.",
                "rows": [], "details": {"null_count": 0, "unique_categories": 0}}
    fill_pct, null_count, total = _fill_rate(df["event_category"])
    unique_cats = int(df["event_category"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct)
    rows = [
        {"label": "Fill Rate",         "actual": f"{fill_pct}%",      "expected": "≥ 95%",  "status": verdict},
        {"label": "Null Entries",      "actual": str(null_count),      "expected": "0",       "status": _count_status(null_count, 1, int(total*0.05)+1)},
        {"label": "Distinct Categories","actual": str(unique_cats),    "expected": "> 0",     "status": "ok" if unique_cats > 0 else "warn"},
        {"label": "Total Records",     "actual": str(total),           "expected": "—",       "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Event category coverage is healthy at {fill_pct}% — {unique_cats} distinct categories present across {total} records."
    elif verdict == "warn":
        inference = f"Event category fill rate of {fill_pct}% is below the 95% baseline — {null_count} missing entries may reduce domain-attribution accuracy."
    else:
        inference = f"Critical category gap at {fill_pct}% fill — {null_count} records lack semantic domain context, compromising category-based analytics."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing · {unique_cats} distinct categories across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_cats} distinct)",
        "expected": "≥ 95% fill · consistent with event_code domain",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_categories": unique_cats}
    }


def assess_event_type(df: pd.DataFrame) -> dict:
    if "event_type" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No event_type records available",
                "actual": "N/A", "expected": "≥ 95% filled", "inference": "No event_type records available.",
                "rows": [], "details": {"null_count": 0, "unique_types": 0}}
    fill_pct, null_count, total = _fill_rate(df["event_type"])
    unique_types = int(df["event_type"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct)
    rows = [
        {"label": "Fill Rate",       "actual": f"{fill_pct}%",    "expected": "≥ 95%",  "status": verdict},
        {"label": "Null Entries",    "actual": str(null_count),    "expected": "0",       "status": _count_status(null_count, 1, int(total*0.05)+1)},
        {"label": "Distinct Types",  "actual": str(unique_types),  "expected": "≤ 5",    "status": "ok" if unique_types <= 5 else "warn"},
        {"label": "Total Records",   "actual": str(total),         "expected": "—",       "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Event type coverage is healthy at {fill_pct}% — {unique_types} distinct type(s) across {total} records. Types must agree with asc_code_to_event_type()."
    elif verdict == "warn":
        inference = f"Event type fill of {fill_pct}% is below threshold — {null_count} missing entries may hide severity mismatches between logged type and code-derived type."
    else:
        inference = f"Critical event type gap — {fill_pct}% fill with {null_count} missing entries. Type–code agreement cannot be verified for those records."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing · {unique_types} distinct types across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_types} types)",
        "expected": "≥ 95% fill · type equals asc_code_to_event_type(event_code)",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_types": unique_types}
    }


def assess_phase(df: pd.DataFrame) -> dict:
    if "phase" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No phase records available",
                "actual": "N/A", "expected": "≥ 70% filled", "inference": "No phase records available.",
                "rows": [], "details": {"null_count": 0, "unique_phases": 0}}
    fill_pct, null_count, total = _fill_rate(df["phase"])
    unique_phases = int(df["phase"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct, bad_thresh=70.0, warn_thresh=90.0)
    rows = [
        {"label": "Fill Rate",       "actual": f"{fill_pct}%",      "expected": "≥ 90%",  "status": verdict},
        {"label": "Null Entries",    "actual": str(null_count),      "expected": "minimal", "status": _count_status(null_count, int(total*0.1)+1, int(total*0.3)+1)},
        {"label": "Distinct Phases", "actual": str(unique_phases),   "expected": "> 0",     "status": "ok" if unique_phases > 0 else "warn"},
        {"label": "Total Records",   "actual": str(total),           "expected": "—",       "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Phase bracket codes are well-populated at {fill_pct}% — {unique_phases} distinct phase(s) visible. Lifecycle completeness analytics can proceed."
    elif verdict == "warn":
        inference = f"Phase fill of {fill_pct}% is below the 90% nominal baseline — {null_count} records lack FSM phase context, which may obscure lifecycle regressions."
    else:
        inference = f"Only {fill_pct}% of records carry phase codes — {null_count} null entries may hide lifecycle anomalies entirely."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing · {unique_phases} distinct phase codes across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_phases} phases)",
        "expected": "≥ 90% fill · forward-only phase progression",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_phases": unique_phases}
    }


def assess_state(df: pd.DataFrame) -> dict:
    if "state" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No state records available",
                "actual": "N/A", "expected": "≥ 70% filled", "inference": "No state records available.",
                "rows": [], "details": {"null_count": 0, "unique_states": 0}}
    fill_pct, null_count, total = _fill_rate(df["state"])
    unique_states = int(df["state"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct, bad_thresh=70.0, warn_thresh=90.0)
    rows = [
        {"label": "Fill Rate",       "actual": f"{fill_pct}%",     "expected": "≥ 90%",   "status": verdict},
        {"label": "Null Entries",    "actual": str(null_count),     "expected": "minimal",  "status": _count_status(null_count, int(total*0.1)+1, int(total*0.3)+1)},
        {"label": "Distinct States", "actual": str(unique_states),  "expected": "> 0",      "status": "ok" if unique_states > 0 else "warn"},
        {"label": "Total Records",   "actual": str(total),          "expected": "—",        "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"State bracket codes are healthy at {fill_pct}% — {unique_states} distinct state(s) detected. Phase–state combination validation can proceed."
    elif verdict == "warn":
        inference = f"State coverage of {fill_pct}% is below nominal — {null_count} records without state context may hide invalid phase–state combinations."
    else:
        inference = f"State fill is critically low at {fill_pct}% — {null_count} null entries make FSM state-sequence validation unreliable."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing · {unique_states} distinct state codes across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_states} states)",
        "expected": "≥ 90% fill · valid for current phase · contiguous transitions",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_states": unique_states}
    }


def assess_substate(df: pd.DataFrame) -> dict:
    if "substate" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No substate records available",
                "actual": "N/A", "expected": "≥ 60% filled", "inference": "No substate records available.",
                "rows": [], "details": {"null_count": 0, "unique_substates": 0}}
    fill_pct, null_count, total = _fill_rate(df["substate"])
    unique_substates = int(df["substate"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct, bad_thresh=60.0, warn_thresh=85.0)
    rows = [
        {"label": "Fill Rate",          "actual": f"{fill_pct}%",       "expected": "≥ 85%",   "status": verdict},
        {"label": "Null Entries",       "actual": str(null_count),       "expected": "minimal",  "status": _count_status(null_count, int(total*0.15)+1, int(total*0.4)+1)},
        {"label": "Distinct Substates", "actual": str(unique_substates), "expected": "> 0",      "status": "ok" if unique_substates > 0 else "warn"},
        {"label": "Total Records",      "actual": str(total),            "expected": "—",        "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Substate granularity is adequate at {fill_pct}% — {unique_substates} distinct substate(s) present. Short transition clusters look healthy."
    elif verdict == "warn":
        inference = f"Substate fill of {fill_pct}% is below the 85% baseline — {null_count} records lack sub-level FSM context. Over-long persistence may be hidden."
    else:
        inference = f"Substate fill critically low at {fill_pct}% — {null_count} missing substates prevent transition-regression detection at the sub-level."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing · {unique_substates} distinct substate codes across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_substates} substates)",
        "expected": "≥ 85% fill · valid for parent state · no over-long persistence",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_substates": unique_substates}
    }


def assess_activity(df: pd.DataFrame) -> dict:
    if "activity" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No activity records available",
                "actual": "N/A", "expected": "0% unknown", "inference": "No activity records available.",
                "rows": [], "details": {"unknown_count": 0, "unique_activities": 0}}
    total = len(df)
    series = df["activity"].astype(str)
    unknown_count = int(series.str.strip().eq("ACT_UNKNOWN_MODAL").sum())
    unknown_pct = round((unknown_count / total * 100.0) if total > 0 else 0.0, 1)
    resolved_pct = round(100.0 - unknown_pct, 1)
    unique_activities = int(df["activity"].nunique())
    verdict = "bad" if unknown_pct > 30.0 else ("warn" if unknown_pct > 10.0 else "ok")
    rows = [
        {"label": "Resolved Rate",      "actual": f"{resolved_pct}%",    "expected": "≥ 90%",   "status": verdict},
        {"label": "Unknown Activities", "actual": str(unknown_count),    "expected": "0",         "status": _count_status(unknown_count, 1, int(total*0.3)+1)},
        {"label": "Unknown Rate",       "actual": f"{unknown_pct}%",     "expected": "< 10%",    "status": "ok" if unknown_pct < 10 else ("warn" if unknown_pct < 30 else "bad")},
        {"label": "Distinct Labels",    "actual": str(unique_activities), "expected": "> 1",      "status": "ok" if unique_activities > 1 else "warn"},
    ]
    if verdict == "ok":
        inference = f"Activity labels are well-resolved at {resolved_pct}% — {unknown_count} ACT_UNKNOWN_MODAL entry/entries present. ON/OFF pairing can be validated."
    elif verdict == "warn":
        inference = f"{unknown_count} unresolved activities ({unknown_pct}%) detected — action context is partially lost. Source attribution from extra_srcinfo may help recover labels."
    else:
        inference = f"{unknown_count} ACT_UNKNOWN_MODAL entries ({unknown_pct}%) exceed the 30% threshold — action attribution is unreliable and ON/OFF pair validation will be inaccurate."
    return {
        "verdict": verdict, "metric": resolved_pct, "unit": "% resolved",
        "note": f"{unknown_count} unresolved (ACT_UNKNOWN_MODAL) · {unique_activities} distinct activity labels",
        "actual": f"{resolved_pct}% resolved ({unknown_count} ACT_UNKNOWN_MODAL · {unique_activities} labels)",
        "expected": "≥ 90% resolved · 0 ACT_UNKNOWN_MODAL · ON/OFF pairs balanced",
        "inference": inference, "rows": rows,
        "details": {"unknown_count": unknown_count, "unique_activities": unique_activities}
    }


def assess_depth(df: pd.DataFrame) -> dict:
    if "depth" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No depth records available",
                "actual": "N/A", "expected": "≥ 95% valid · 0 negative", "inference": "No depth records available.",
                "rows": [], "details": {"null_count": 0, "negative_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["depth"])
    numeric = pd.to_numeric(df["depth"], errors="coerce")
    negative_count = int((numeric < 0).sum())
    verdict = "bad" if (fill_pct < 80.0 or negative_count > 0) else ("warn" if fill_pct < 95.0 else "ok")
    rows = [
        {"label": "Fill Rate",       "actual": f"{fill_pct}%",        "expected": "≥ 95%",   "status": _verdict_from_fill(fill_pct)},
        {"label": "Negative Values", "actual": str(negative_count),   "expected": "0",        "status": "ok" if negative_count == 0 else "bad"},
        {"label": "Null Entries",    "actual": str(null_count),        "expected": "0",        "status": _count_status(null_count, 1, int(total*0.05)+1)},
        {"label": "Total Records",   "actual": str(total),             "expected": "—",        "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Call depth values are valid — {fill_pct}% coverage and 0 negative values. Stack nesting health can be assessed."
    elif negative_count > 0:
        inference = f"{negative_count} negative depth value(s) flagged — stack tracking integrity is compromised. Enter/exit call balance may be off."
    else:
        inference = f"Depth fill rate of {fill_pct}% is below the 95% baseline — {null_count} missing entries reduce call-stack visibility."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% valid",
        "note": f"{null_count} missing · {negative_count} negative depth values flagged",
        "actual": f"{fill_pct}% valid ({null_count} null · {negative_count} negative)",
        "expected": "≥ 95% fill · 0 negative values · oscillates in small band",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "negative_count": negative_count}
    }


def assess_message(df: pd.DataFrame) -> dict:
    if "message" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No message records available",
                "actual": "N/A", "expected": "≥ 95% filled", "inference": "No message records available.",
                "rows": [], "details": {"null_count": 0, "empty_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["message"])
    empty_count = int(df["message"].dropna().astype(str).str.strip().eq("").sum())
    effective_fill = round(((total - null_count - empty_count) / total * 100.0) if total > 0 else 100.0, 1)
    verdict = _verdict_from_fill(effective_fill)
    rows = [
        {"label": "Effective Fill",  "actual": f"{effective_fill}%",  "expected": "≥ 95%",  "status": verdict},
        {"label": "Null Messages",   "actual": str(null_count),        "expected": "0",       "status": _count_status(null_count, 1, int(total*0.05)+1)},
        {"label": "Empty Messages",  "actual": str(empty_count),       "expected": "0",       "status": _count_status(empty_count, 1, int(total*0.05)+1)},
        {"label": "Total Records",   "actual": str(total),             "expected": "—",       "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Message coverage is healthy at {effective_fill}% — {null_count} null and {empty_count} empty message(s). Critical lifecycle events should carry messages."
    elif verdict == "warn":
        inference = f"Message fill of {effective_fill}% is below threshold — {null_count} null and {empty_count} empty messages may hide context for critical events."
    else:
        inference = f"Message coverage is critically low at {effective_fill}% — {null_count + empty_count} records without interpretability text, reducing audit trail quality."
    return {
        "verdict": verdict, "metric": effective_fill, "unit": "% filled",
        "note": f"{null_count} null · {empty_count} empty messages across {total} records",
        "actual": f"{effective_fill}% filled ({null_count} null · {empty_count} empty)",
        "expected": "≥ 95% fill · present for all critical lifecycle events",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "empty_count": empty_count}
    }


def assess_custom_data_size(df: pd.DataFrame) -> dict:
    if "custom_data_size" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No custom_data_size records available",
                "actual": "N/A", "expected": "0 for control · 1-32 for data", "inference": "No custom_data_size records available.",
                "rows": [], "details": {"null_count": 0, "zero_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["custom_data_size"])
    numeric = pd.to_numeric(df["custom_data_size"], errors="coerce")
    zero_count = int((numeric == 0).sum())
    over32_count = int((numeric > 32).sum())
    verdict = _verdict_from_fill(fill_pct, bad_thresh=70.0, warn_thresh=90.0)
    rows = [
        {"label": "Fill Rate",       "actual": f"{fill_pct}%",       "expected": "≥ 90%",   "status": verdict},
        {"label": "Zero-Size Entries","actual": str(zero_count),      "expected": "control events only", "status": "ok"},
        {"label": "Over-32 Entries", "actual": str(over32_count),    "expected": "0 (clamped)",         "status": "ok" if over32_count == 0 else "warn"},
        {"label": "Null Entries",    "actual": str(null_count),       "expected": "0",                   "status": _count_status(null_count, 1, int(total*0.1)+1)},
    ]
    if verdict == "ok":
        inference = f"Payload size coverage is adequate at {fill_pct}% — {zero_count} zero-size (control) entries and {over32_count} over-32 entry/entries detected."
    elif verdict == "warn":
        inference = f"Payload size fill of {fill_pct}% is below the 90% baseline — {null_count} missing entries reduce size-vs-payload agreement checks."
    else:
        inference = f"Critically low payload size coverage at {fill_pct}% — declared-vs-actual payload validation cannot be completed for {null_count} missing records."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% present",
        "note": f"{null_count} missing · {zero_count} zero-size entries across {total} records",
        "actual": f"{fill_pct}% present ({zero_count} zero · {over32_count} >32 · {null_count} null)",
        "expected": "0 for control events · 1–32 for measurement · 0 pre-clamp losses",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "zero_count": zero_count}
    }


def assess_custom_data(df: pd.DataFrame) -> dict:
    if "custom_data" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No custom_data records available",
                "actual": "N/A", "expected": "present for measurement actions", "inference": "No custom_data records available.",
                "rows": [], "details": {"null_count": 0, "unique_values": 0}}
    fill_pct, null_count, total = _fill_rate(df["custom_data"])
    unique_values = int(df["custom_data"].dropna().nunique())
    verdict = _verdict_from_fill(fill_pct, bad_thresh=70.0, warn_thresh=90.0)
    rows = [
        {"label": "Fill Rate",      "actual": f"{fill_pct}%",      "expected": "≥ 90%",   "status": verdict},
        {"label": "Null Entries",   "actual": str(null_count),      "expected": "0 for measurement", "status": _count_status(null_count, 1, int(total*0.1)+1)},
        {"label": "Unique Values",  "actual": str(unique_values),   "expected": "> 1",              "status": "ok" if unique_values > 1 else "warn"},
        {"label": "Total Records",  "actual": str(total),           "expected": "—",                 "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Payload content coverage is {fill_pct}% — {unique_values} distinct value(s) present. Drift and outlier analytics can proceed."
    elif verdict == "warn":
        inference = f"Payload fill of {fill_pct}% is below threshold — {null_count} missing entries may represent measurements without payloads, blocking drift analysis."
    else:
        inference = f"Custom data fill is critically low at {fill_pct}% — {null_count} missing payloads prevent CUSUM drift and outlier analytics for those records."
    return {
        "verdict": verdict, "metric": fill_pct, "unit": "% filled",
        "note": f"{null_count} missing · {unique_values} distinct values across {total} records",
        "actual": f"{fill_pct}% filled ({null_count} null · {unique_values} unique)",
        "expected": "present for measurement actions · within physical bounds · no discontinuity",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "unique_values": unique_values}
    }


def assess_ext_msg_length(df: pd.DataFrame) -> dict:
    if "ext_msg_length" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No ext_msg_length records available",
                "actual": "N/A", "expected": "0 mismatches", "inference": "No ext_msg_length records available.",
                "rows": [], "details": {"null_count": 0, "mismatch_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["ext_msg_length"])
    mismatch_count = 0
    if "ext_message" in df.columns:
        declared = pd.to_numeric(df["ext_msg_length"], errors="coerce")
        actual_len = df["ext_message"].astype(str).apply(lambda x: len(x) if x not in ("nan", "<NA>") else np.nan)
        valid_mask = declared.notna() & actual_len.notna()
        mismatch_count = int(((declared[valid_mask] - actual_len[valid_mask]).abs() > 0).sum())
    verdict = "bad" if (mismatch_count > 0 or fill_pct < 80.0) else ("warn" if fill_pct < 95.0 else "ok")
    rows = [
        {"label": "Fill Rate",            "actual": f"{fill_pct}%",       "expected": "≥ 95%",   "status": _verdict_from_fill(fill_pct)},
        {"label": "Length Mismatches",    "actual": str(mismatch_count),  "expected": "0",        "status": "ok" if mismatch_count == 0 else "bad"},
        {"label": "Null Entries",         "actual": str(null_count),      "expected": "0",        "status": _count_status(null_count, 1, int(total*0.05)+1)},
        {"label": "Total Records",        "actual": str(total),           "expected": "—",        "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Extended length fields are consistent — {fill_pct}% coverage and {mismatch_count} declared-vs-actual mismatch(es)."
    elif mismatch_count > 0:
        inference = f"{mismatch_count} length-vs-content mismatch(es) detected — declared ext_msg_length does not match actual ext_message character count, indicating potential truncation."
    else:
        inference = f"Extended length fill of {fill_pct}% is below threshold — {null_count} missing entries reduce clamp-scan coverage."
    return {
        "verdict": verdict, "metric": round(fill_pct, 1), "unit": "% consistent",
        "note": f"{null_count} missing · {mismatch_count} length-vs-content mismatches detected",
        "actual": f"{fill_pct}% present · {mismatch_count} length mismatch(es)",
        "expected": "≥ 95% fill · 0 mismatches · equals actual text length",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "mismatch_count": mismatch_count}
    }


def assess_ext_message(df: pd.DataFrame) -> dict:
    if "ext_message" not in df.columns or df.empty:
        return {"verdict": "info", "metric": 0.0, "unit": "%", "note": "No ext_message records available",
                "actual": "N/A", "expected": "sparse · well-formed", "inference": "No ext_message records available.",
                "rows": [], "details": {"null_count": 0, "empty_count": 0}}
    fill_pct, null_count, total = _fill_rate(df["ext_message"])
    empty_count = int(df["ext_message"].dropna().astype(str).str.strip().eq("").sum())
    effective_fill = round(((total - null_count - empty_count) / total * 100.0) if total > 0 else 100.0, 1)
    verdict = _verdict_from_fill(effective_fill, bad_thresh=60.0, warn_thresh=85.0)
    rows = [
        {"label": "Effective Fill",  "actual": f"{effective_fill}%",  "expected": "≥ 85% where expected", "status": verdict},
        {"label": "Null Entries",    "actual": str(null_count),        "expected": "sparse OK",              "status": "ok"},
        {"label": "Empty Entries",   "actual": str(empty_count),       "expected": "0 (use null instead)",   "status": _count_status(empty_count, 1, int(total*0.1)+1)},
        {"label": "Total Records",   "actual": str(total),             "expected": "—",                      "status": "ok"},
    ]
    if verdict == "ok":
        inference = f"Extended messages are present at {effective_fill}% — this field is sparse by design. {empty_count} empty (non-null) entry/entries should use null instead."
    elif verdict == "warn":
        inference = f"Extended message fill of {effective_fill}% is below nominal — {empty_count} empty entries should be null. Parent-record consistency should be checked."
    else:
        inference = f"Extended message coverage critically low at {effective_fill}% — {null_count + empty_count} records without ext_message context where diagnostics were expected."
    return {
        "verdict": verdict, "metric": effective_fill, "unit": "% filled",
        "note": f"{null_count} null · {empty_count} empty ext messages across {total} records",
        "actual": f"{effective_fill}% filled ({null_count} null · {empty_count} empty)",
        "expected": "sparse · well-formed · consistent with parent record",
        "inference": inference, "rows": rows,
        "details": {"null_count": null_count, "empty_count": empty_count}
    }


# ---------------------------------------------------------------------------
# Assessment chart data (chart canvas handler)
# ---------------------------------------------------------------------------

_ASSESS_DISPATCH = None


def _get_dispatch():
    global _ASSESS_DISPATCH
    if _ASSESS_DISPATCH is None:
        _ASSESS_DISPATCH = {
            "seq_id":           assess_sequence_id,
            "timestamp":        assess_timestamp,
            "event_code":       assess_event_code,
            "extra_srcinfo":    assess_extra_srcinfo,
            "log_level":        assess_log_level,
            "event_category":   assess_event_category,
            "event_type":       assess_event_type,
            "phase":            assess_phase,
            "state":            assess_state,
            "substate":         assess_substate,
            "activity":         assess_activity,
            "depth":            assess_depth,
            "message":          assess_message,
            "custom_data_size": assess_custom_data_size,
            "custom_data":      assess_custom_data,
            "ext_msg_length":   assess_ext_msg_length,
            "ext_message":      assess_ext_message,
        }
    return _ASSESS_DISPATCH


def get_assessment_chart_data(
    df: pd.DataFrame,
    attribute_key: str,
    method: str,
    chart_type: str
) -> dict:
    if df is None or df.empty:
        return {"type": "empty", "verdict": "info", "reason": "No active dataset loaded."}

    target_col = attribute_key.lower().strip()

    if target_col not in df.columns:
        return {"type": "empty", "verdict": "info",
                "reason": f"Column '{target_col}' not found in canonical layout."}

    series = pd.to_numeric(df[target_col], errors='coerce')
    valid_mask = series.notna()
    valid_seqs = series[valid_mask].to_numpy()
    valid_indices = np.arange(len(df))[valid_mask].tolist()

    # ------------------------------------------------------------------
    # metrics_summary — route to per-attribute assess function
    # ------------------------------------------------------------------
    if method == "metrics_summary":
        dispatch = _get_dispatch()
        fn = dispatch.get(target_col)
        if fn is None:
            return {"type": "empty", "verdict": "info",
                    "reason": f"No summary handler registered for '{target_col}'."}
        res = fn(df)
        return {
            "type": "metrics_summary",
            "verdict": res["verdict"],
            "rows": res.get("rows", []),
            "inference": res.get("inference", res.get("note", "")),
            "actual": res.get("actual", ""),
            "expected": res.get("expected", ""),
        }

    # ------------------------------------------------------------------
    # frequency_chart — value frequency distribution for categorical cols
    # ------------------------------------------------------------------
    if method == "frequency_chart":
        total = len(df)
        none_count = int(df[target_col].isna().sum())
        display_col = df[target_col].fillna("(none)").astype(str)
        counts = display_col.value_counts()
        top_n = counts.head(20)
        rare_threshold = max(2, int(total * 0.02))
        rare_labels = [v for v, c in counts.items() if c <= rare_threshold and v != "(none)"]
        dispatch = _get_dispatch()
        fn = dispatch.get(target_col)
        res = fn(df) if fn else {}
        rare_note = f"{len(rare_labels)} rare categor{'y' if len(rare_labels)==1 else 'ies'} · " if rare_labels else ""
        scan_note = f"{rare_note}{none_count} missing · {len(counts)} distinct values"
        return {
            "type": "bar_chart",
            "verdict": res.get("verdict", "info"),
            "x": top_n.index.tolist(),
            "y": top_n.values.tolist(),
            "title": f"value frequency · {target_col}",
            "note": scan_note,
            "inference": res.get("inference", scan_note),
            "actual": res.get("actual", ""),
            "expected": res.get("expected", ""),
        }

    # ------------------------------------------------------------------
    # depth_trace — call depth sequence or distribution
    # ------------------------------------------------------------------
    if method == "depth_trace":
        numeric = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if numeric.empty:
            return {"type": "empty", "verdict": "info", "reason": "No numeric depth values found."}
        dispatch = _get_dispatch()
        fn = dispatch.get(target_col)
        res = fn(df) if fn else {}
        vals = numeric.values
        neg_count = int((vals < 0).sum())
        if chart_type == "distribution":
            min_v, max_v = int(vals.min()), int(vals.max())
            bins_count = min(15, max(2, max_v - min_v + 1))
            counts_arr, edges = np.histogram(vals, bins=bins_count)
            labels = [f"{int(edges[i])}" for i in range(len(counts_arr))]
            scan_note = f"min={int(vals.min())} · max={int(vals.max())} · mean={vals.mean():.1f} · {neg_count} negative"
            return {
                "type": "bar_chart",
                "verdict": res.get("verdict", "info"),
                "x": labels,
                "y": counts_arr.tolist(),
                "title": "depth distribution",
                "note": scan_note,
                "inference": res.get("inference", ""),
                "actual": res.get("actual", ""),
                "expected": res.get("expected", ""),
            }
        else:
            max_pts = 120
            if len(vals) > max_pts:
                idx = np.linspace(0, len(vals) - 1, max_pts, dtype=int)
                x_vals = idx.tolist()
                y_vals = vals[idx].tolist()
            else:
                x_vals = list(range(len(vals)))
                y_vals = vals.tolist()
            highlights = [1 if v < 0 else 0 for v in y_vals]
            scan_note = f"min={int(vals.min())} · max={int(vals.max())} · mean={vals.mean():.1f} · {neg_count} negative"
            return {
                "type": "line_chart",
                "verdict": res.get("verdict", "info"),
                "x": x_vals,
                "y": y_vals,
                "highlights": highlights,
                "title": "call depth sequence",
                "note": scan_note,
                "inference": res.get("inference", ""),
                "actual": res.get("actual", ""),
                "expected": res.get("expected", ""),
            }

    # ------------------------------------------------------------------
    # length_distribution — string length histogram for text columns
    # ------------------------------------------------------------------
    if method == "length_distribution":
        text_col = df[target_col].dropna().astype(str)
        if text_col.empty:
            return {"type": "empty", "verdict": "info", "reason": "No text values found."}
        dispatch = _get_dispatch()
        fn = dispatch.get(target_col)
        res = fn(df) if fn else {}
        lengths = text_col.apply(len)
        max_len = int(lengths.max())
        if max_len <= 50:
            step = 5
        elif max_len <= 200:
            step = 20
        else:
            step = 50
        bins = list(range(0, max_len + step + 1, step))
        if len(bins) < 2:
            bins = [0, max_len + 1]
        counts_arr, edges = np.histogram(lengths, bins=bins)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts_arr)) if counts_arr[i] > 0]
        y_vals = [int(counts_arr[i]) for i in range(len(counts_arr)) if counts_arr[i] > 0]
        scan_note = f"min={int(lengths.min())} · max={int(lengths.max())} chars · mean={lengths.mean():.0f}"
        return {
            "type": "bar_chart",
            "verdict": res.get("verdict", "info"),
            "x": labels,
            "y": y_vals,
            "title": f"length distribution · {target_col}",
            "note": scan_note,
            "inference": res.get("inference", ""),
            "actual": res.get("actual", ""),
            "expected": res.get("expected", ""),
        }

    # ------------------------------------------------------------------
    # value_distribution — numeric distribution for custom_data
    # ------------------------------------------------------------------
    if method == "value_distribution":
        numeric = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if numeric.empty:
            return {"type": "empty", "verdict": "info", "reason": "No numeric values found."}
        dispatch = _get_dispatch()
        fn = dispatch.get(target_col)
        res = fn(df) if fn else {}
        vals = numeric.values
        scan_note = f"min={float(vals.min()):.2f} · max={float(vals.max()):.2f} · mean={vals.mean():.2f}"
        if chart_type == "line":
            max_pts = 120
            if len(vals) > max_pts:
                idx = np.linspace(0, len(vals) - 1, max_pts, dtype=int)
                x_vals = idx.tolist()
                y_vals = vals[idx].tolist()
            else:
                x_vals = list(range(len(vals)))
                y_vals = [float(v) for v in vals]
            return {
                "type": "line_chart",
                "verdict": res.get("verdict", "info"),
                "x": x_vals, "y": y_vals, "highlights": [],
                "title": "custom_data value sequence",
                "note": scan_note,
                "inference": res.get("inference", ""),
                "actual": res.get("actual", ""),
                "expected": res.get("expected", ""),
            }
        else:
            n_bins = min(15, max(2, len(set(vals))))
            counts_arr, edges = np.histogram(vals, bins=n_bins)
            labels = [f"{edges[i]:.1f}" for i in range(len(counts_arr))]
            return {
                "type": "bar_chart",
                "verdict": res.get("verdict", "info"),
                "x": labels,
                "y": counts_arr.tolist(),
                "title": "payload value distribution",
                "note": scan_note,
                "inference": res.get("inference", ""),
                "actual": res.get("actual", ""),
                "expected": res.get("expected", ""),
            }

    if len(valid_seqs) < 2:
        return {"type": "empty", "verdict": "info",
                "reason": f"'{target_col}' does not contain enough numeric samples."}

    deltas = np.diff(valid_seqs)

    # ------------------------------------------------------------------
    # change_point
    # ------------------------------------------------------------------
    if method == "change_point":
        cp_indices = np.where(deltas < 0)[0]
        cp_res = assess_sequence_id(df)
        cp_verdict = "warn" if len(cp_indices) > 0 else "ok"

        if chart_type == "line":
            max_points = 120
            if len(valid_seqs) > max_points:
                sample_idx = np.linspace(0, len(valid_seqs) - 1, max_points, dtype=int)
                x_vals = [valid_indices[i] for i in sample_idx]
                y_vals = valid_seqs[sample_idx].tolist()
                highlights = [
                    1 if np.any(
                        (cp_indices >= sample_idx[i]) &
                        (cp_indices < (sample_idx[i + 1] if i + 1 < max_points else len(valid_seqs)))
                    ) else 0
                    for i in range(max_points)
                ]
            else:
                x_vals = valid_indices
                y_vals = valid_seqs.tolist()
                highlights = (deltas < 0).astype(int).tolist()
                highlights.insert(0, 0)
            scan_note = cp_res.get("note", f"{len(cp_indices)} change point(s) detected")
            return {
                "type": "line_chart",
                "verdict": cp_verdict,
                "x": x_vals, "y": y_vals, "highlights": highlights,
                "title": f"Change-Point Trend Timeline: {target_col}",
                "note": scan_note,
                "inference": cp_res.get("inference", ""),
                "actual": cp_res.get("actual", ""),
                "expected": cp_res.get("expected", ""),
            }

        elif chart_type == "distribution":
            # Reset magnitude distribution — how large are the backward jumps
            if len(cp_indices) > 0:
                magnitudes = np.abs(deltas[cp_indices])
                unique_m, m_counts = np.unique(magnitudes, return_counts=True)
                sort_idx = np.argsort(unique_m)
                x_vals = [f"|Δ|={int(unique_m[i])}" for i in sort_idx]
                y_vals = [int(m_counts[i]) for i in sort_idx]
                if len(x_vals) > 20:
                    top_idx = np.argsort(m_counts)[::-1][:20]
                    combined = sorted([(int(unique_m[i]), int(m_counts[i])) for i in top_idx])
                    x_vals = [f"|Δ|={d}" for d, _ in combined]
                    y_vals = [c for _, c in combined]
                scan_note = f"{len(cp_indices)} reset(s) · magnitudes range {int(magnitudes.min())}–{int(magnitudes.max())}"
            else:
                # No change points — show overall delta distribution instead
                unique_d, d_counts = np.unique(deltas, return_counts=True)
                sort_idx = np.argsort(unique_d)
                x_vals = [str(int(unique_d[i])) for i in sort_idx[:20]]
                y_vals = [int(d_counts[i]) for i in sort_idx[:20]]
                scan_note = "No change points detected — showing full delta distribution"
            return {
                "type": "bar_chart",
                "verdict": cp_verdict,
                "x": x_vals, "y": y_vals,
                "title": "reset magnitude distribution · seq_id",
                "note": scan_note,
                "inference": cp_res.get("inference", ""),
                "actual": cp_res.get("actual", ""),
                "expected": cp_res.get("expected", ""),
            }

        elif chart_type == "counts":
            normal  = int(np.sum(deltas == 1))
            gaps    = int(np.sum(deltas > 1))
            dups    = int(np.sum(deltas == 0))
            resets  = int(len(cp_indices))
            x_vals  = ["Normal (Δ=1)", "Gap (Δ>1)", "Duplicate (Δ=0)", "Reset (Δ<0)"]
            y_vals  = [normal, gaps, dups, resets]
            scan_note = f"{normal} normal · {gaps} gap(s) · {dups} dup(s) · {resets} reset(s)"
            return {
                "type": "bar_chart",
                "verdict": cp_verdict,
                "x": x_vals, "y": y_vals,
                "title": "step event type counts · seq_id",
                "note": scan_note,
                "inference": cp_res.get("inference", ""),
                "actual": cp_res.get("actual", ""),
                "expected": cp_res.get("expected", ""),
            }

    # ------------------------------------------------------------------
    # gap_scan
    # ------------------------------------------------------------------
    elif method == "gap_scan":
        gap_res = assess_sequence_id(df)

        if chart_type == "line":
            max_points = 120
            if len(deltas) > max_points:
                sample_idx = np.linspace(0, len(deltas) - 1, max_points, dtype=int)
                x_vals = [valid_indices[i] for i in sample_idx]
                y_vals = deltas[sample_idx].tolist()
            else:
                x_vals = valid_indices[:-1]
                y_vals = deltas.tolist()
            return {
                "type": "line_chart",
                "verdict": gap_res.get("verdict", "ok"),
                "x": x_vals, "y": y_vals,
                "title": f"Step Delta Magnitude Tracker: {target_col}",
                "note": gap_res.get("note", ""),
                "inference": gap_res.get("inference", ""),
                "actual": gap_res.get("actual", ""),
                "expected": gap_res.get("expected", ""),
            }

        elif chart_type == "distribution":
            # Histogram of delta values — shows how gaps are distributed
            unique_d, d_counts = np.unique(deltas, return_counts=True)
            sort_idx = np.argsort(unique_d)
            x_vals = [str(int(unique_d[i])) for i in sort_idx]
            y_vals = [int(d_counts[i]) for i in sort_idx]
            if len(x_vals) > 20:
                top_idx = np.argsort(d_counts)[::-1][:20]
                combined = sorted([(int(unique_d[i]), int(d_counts[i])) for i in top_idx])
                x_vals = [str(d) for d, _ in combined]
                y_vals = [c for _, c in combined]
            scan_note = f"Δ value distribution across {len(deltas)} steps · {int(np.sum(deltas > 1))} gap(s)"
            return {
                "type": "bar_chart",
                "verdict": gap_res.get("verdict", "ok"),
                "x": x_vals, "y": y_vals,
                "title": "delta value distribution · seq_id",
                "note": scan_note,
                "inference": gap_res.get("inference", ""),
                "actual": gap_res.get("actual", ""),
                "expected": gap_res.get("expected", ""),
            }

        elif chart_type == "counts":
            normal = int(np.sum(deltas == 1))
            gaps   = int(np.sum(deltas > 1))
            dups   = int(np.sum(deltas == 0))
            resets = int(np.sum(deltas < 0))
            x_vals = ["Normal (Δ=1)", "Gap (Δ>1)", "Duplicate (Δ=0)", "Reset (Δ<0)"]
            y_vals = [normal, gaps, dups, resets]
            scan_note = f"{normal} normal · {gaps} gap(s) · {dups} dup(s) · {resets} reset(s)"
            return {
                "type": "bar_chart",
                "verdict": gap_res.get("verdict", "ok"),
                "x": x_vals, "y": y_vals,
                "title": "step event type counts · seq_id",
                "note": scan_note,
                "inference": gap_res.get("inference", ""),
                "actual": gap_res.get("actual", ""),
                "expected": gap_res.get("expected", ""),
            }

    # ------------------------------------------------------------------
    # monotonicity_scan
    # ------------------------------------------------------------------
    elif target_col == "timestamp" and method == "monotonicity_scan":
        x_vals = valid_indices
        y_vals = valid_seqs.tolist()
        highlights = [0]
        for dt in deltas:
            highlights.append(2 if dt < 0 else (1 if dt == 0 else 0))
        verdict = "bad" if any(h == 2 for h in highlights) else ("warn" if any(h == 1 for h in highlights) else "ok")
        mon_res = assess_timestamp(df)
        return {
            "type": "line_chart", "verdict": verdict,
            "x": x_vals, "y": y_vals, "highlights": highlights,
            "title": "Timestamp Monotonicity Timeline",
            "note": mon_res.get("note", ""),
            "inference": mon_res.get("inference", ""),
            "actual": mon_res.get("actual", ""),
            "expected": mon_res.get("expected", ""),
        }

    # ------------------------------------------------------------------
    # inter_arrival — inter-arrival Δt distribution for timestamp
    # ------------------------------------------------------------------
    elif target_col == "timestamp" and method == "inter_arrival":
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        if std_d == 0:
            return {"type": "empty", "verdict": "ok", "reason": "All inter-arrival intervals are identical — no outliers."}
        z_scores = (deltas - mean_d) / std_d
        outlier_count = int(np.sum(np.abs(z_scores) > 2.5))
        verdict = "bad" if outlier_count > max(1, int(len(deltas) * 0.05)) else ("warn" if outlier_count > 0 else "ok")
        ia_res = assess_timestamp(df)
        scan_note = f"mean={mean_d:.3f}s · {outlier_count} outlier interval(s) (|z|>2.5)"
        n_bins = min(15, max(3, len(deltas) // 10))
        counts_arr, edges = np.histogram(deltas, bins=n_bins)
        labels = [f"{edges[i]:.3f}" for i in range(len(counts_arr))]
        return {
            "type": "bar_chart",
            "verdict": verdict,
            "x": labels,
            "y": counts_arr.tolist(),
            "title": "inter-arrival Δt distribution",
            "note": scan_note,
            "inference": ia_res.get("inference", ""),
            "actual": ia_res.get("actual", ""),
            "expected": ia_res.get("expected", ""),
        }

    return {
        "type": "empty", "verdict": "info",
        "reason": "No matched schema for this attribute/method/chart combination."
    }


# ---------------------------------------------------------------------------
# Full matrix
# ---------------------------------------------------------------------------

def generate_full_attribute_matrix(df: pd.DataFrame) -> dict:
    assessors = {
        "seq_id":           (assess_sequence_id,    "seq_id (Sequence Alignment)",          "Gap/Duplicate Scan · Change-Point Detection"),
        "timestamp":        (assess_timestamp,       "timestamp (Temporal Sequence)",        "Monotonicity Scan · Rolling Z-Score"),
        "event_code":       (assess_event_code,      "event_code (Event Identifier)",        "Completeness Check · Unique Code Distribution"),
        "extra_srcinfo":    (assess_extra_srcinfo,   "extra_srcinfo (Source Resolution)",    "Source Normalization Audit · Unknown Rate Scan"),
        "log_level":        (assess_log_level,       "log_level (Severity Classification)",  "Critical Rate Scan · Invalid Code Detection"),
        "event_category":   (assess_event_category,  "event_category (Category Mapping)",    "Completeness Check · Category Distribution"),
        "event_type":       (assess_event_type,      "event_type (Type Classification)",     "Completeness Check · Type Distribution"),
        "phase":            (assess_phase,           "phase (Operational Phase)",            "Bracket Code Completeness · Phase Coverage"),
        "state":            (assess_state,           "state (System State)",                 "Bracket Code Completeness · State Coverage"),
        "substate":         (assess_substate,        "substate (Sub-State Granularity)",     "Bracket Code Completeness · Substate Coverage"),
        "activity":         (assess_activity,        "activity (Activity Resolution)",       "Unknown Activity Rate · Label Distribution"),
        "depth":            (assess_depth,           "depth (Call Depth)",                   "Numeric Validity · Negative Value Scan"),
        "message":          (assess_message,         "message (Log Message)",                "Completeness Check · Empty Content Scan"),
        "custom_data_size": (assess_custom_data_size,"custom_data_size (Payload Size)",      "Completeness Check · Zero-Size Detection"),
        "custom_data":      (assess_custom_data,     "custom_data (Payload Content)",        "Completeness Check · Value Distribution"),
        "ext_msg_length":   (assess_ext_msg_length,  "ext_msg_length (Extended Length)",     "Length-Content Consistency · Completeness Check"),
        "ext_message":      (assess_ext_message,     "ext_message (Extended Message)",       "Completeness Check · Empty Content Scan"),
    }

    matrix = {}
    for key, (fn, display_name, methods) in assessors.items():
        res = fn(df)
        matrix[key] = {
            "name":       display_name,
            "ml_methods": methods,
            "verdict":    res["verdict"],
            "metric":     f"{res['metric']}{res['unit']}",
            "note":       res["note"],
            "actual":     res.get("actual", ""),
            "expected":   res.get("expected", ""),
            "inference":  res.get("inference", res.get("note", "")),
        }
    return matrix


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def process_and_assess_raw_log(raw_input_path_or_df) -> tuple:
    canonical_df = make_canonical(raw_input_path_or_df)
    status_matrix = generate_full_attribute_matrix(canonical_df)
    return canonical_df, status_matrix
