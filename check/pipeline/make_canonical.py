import pandas as pd
import re
from typing import Union

CANONICAL_COLUMNS = [
    "seq_id", "timestamp", "event_code", "extra_srcinfo", "log_level",
    "event_category", "event_type", "phase", "state", "substate",
    "activity", "depth", "message", "custom_data_size", "custom_data",
    "ext_msg_length", "ext_message"
]

CSV_HEADER_MAP = {
    "seq_id": "Seq",
    "timestamp": "Timestamp(ms)",
    "event_code": "Event code",
    "extra_srcinfo": "Src",
    "log_level": "Level",
    "event_category": "Category",
    "event_type": "Type",
    "phase": "Ph",
    "state": "St",
    "substate": "Sub",
    "activity": "Act",
    "depth": "Depth",
    "message": "Message",
    "custom_data": "Value",
    "custom_data_size": "Val Len",
    "ext_msg_length": "Ext Len",
    "ext_message": "ExtMsg"
}

_TARGET_MARKERS = {
    "seq", "timestamp", "ms", "event", "code", "src", "level", 
    "category", "type", "ph", "st", "sub", "act", "depth", "msg", "value"
}

SRC_NUMERIC_NORM_MAP = {
    "256": "ACT_SM",
    "512": "ACT_SYS",
    "32": "ACT_API",
    "0": "SYSTEM_CORE"
}

# Maps incoming raw string text directly to canonical numbers
LOG_LEVEL_NORM_MAP = {
    "EMERGENCY": "0",
    "ALERT": "1",
    "CRITICAL": "2",
    "ERROR": "3",
    "WARNING": "4",
    "NOTICE": "5",
    "INFO": "6",
    "DEBUG": "7"
}

def _extract_bracket_code(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    m = re.search(r"(\[[^\]]+\])", s)
    return m.group(1) if m else s

def _clean_value(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    cleaned = re.sub(r"(?i)\s*psi\s*", "", s)
    return cleaned if cleaned != "" else pd.NA

def _clean_seq_id(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    match = re.search(r"\d+", s)
    return match.group(0) if match else s

def _clean_timestamp(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    match = re.search(r"^([\d\.]+)", s)
    return match.group(1) if match else s

def _normalize_src(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    try:
        f_val = float(s)
        if f_val.is_integer():
            val = int(f_val)
            clean_id = str(val & 0x7FFF)
            return SRC_NUMERIC_NORM_MAP.get(clean_id, f"SRC_UNKNOWN({clean_id})")
    except ValueError:
        pass
    return s

def _normalize_log_level(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper()
    
    # 1. Text lookup check (e.g. matching "EMERGENCY" -> "0")
    if s in LOG_LEVEL_NORM_MAP:
        return LOG_LEVEL_NORM_MAP[s]
        
    # 2. Digit lookup fallback check (e.g. if the file contains "Level 3" or raw "3")
    match = re.search(r"\d+", s)
    if match:
        clean_id = match.group(0)
        if clean_id in LOG_LEVEL_NORM_MAP.values():
            return clean_id
        else:
            return f"INVALID({clean_id})"
            
    return "UNKNOWN"

def _clean_message(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    cleaned = re.sub(r"^(\s*\[[^\]]+\]\s*)+", "", s)
    return cleaned.strip()

def _detect_skiprows(path: str) -> int:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                best_row_idx = 0
                max_score = -1
                for i in range(50):
                    line = f.readline()
                    if not line:
                        break
                    tokens = [re.sub(r"\W+", "", t.lower().strip()) for t in re.split(r"[,\t;]", line)]
                    tokens = [t for t in tokens if t]
                    score = sum(1 for t in tokens if any(m in t or t in m for m in _TARGET_MARKERS))
                    if score > max_score and len(tokens) >= 3:
                        max_score = score
                        best_row_idx = i
                if max_score > 0:
                    return best_row_idx
        except Exception:
            continue
    return 0

def make_canonical(raw_input: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        skip = _detect_skiprows(raw_input)
        try:
            df = pd.read_csv(raw_input, skiprows=skip, sep=None, engine="python", header=0, dtype=str)
        except Exception:
            try:
                df = pd.read_csv(raw_input, skiprows=skip, sep=None, engine="python", header=None, dtype=str)
            except Exception:
                return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    canonical_df = pd.DataFrame(index=df.index, columns=CANONICAL_COLUMNS)

    for canonical_field, target_csv_name in CSV_HEADER_MAP.items():
        matched_column = None
        target_lower = target_csv_name.lower()
        target_alpha = re.sub(r"\W+", "", target_lower)
        
        for col_name in df.columns:
            if col_name.lower() == target_lower:
                matched_column = col_name
                break
        if matched_column is None:
            for col_name in df.columns:
                col_lower = col_name.lower()
                col_alpha = re.sub(r"\W+", "", col_lower)
                if col_alpha == target_alpha or (target_alpha in col_alpha) or (col_alpha in target_alpha):
                    matched_column = col_name
                    break
        if matched_column is None and canonical_field == "timestamp":
            for col_name in df.columns:
                if "time" in col_name.lower():
                    matched_column = col_name
                    break
        if matched_column is not None:
            canonical_df[canonical_field] = df[matched_column]
        else:
            canonical_df[canonical_field] = pd.NA

    if "seq_id" in canonical_df.columns:
        canonical_df["seq_id"] = canonical_df["seq_id"].apply(_clean_seq_id)
    if "timestamp" in canonical_df.columns:
        canonical_df["timestamp"] = canonical_df["timestamp"].apply(_clean_timestamp)
        
    if "extra_srcinfo" in canonical_df.columns:
        canonical_df["extra_srcinfo"] = canonical_df["extra_srcinfo"].apply(_normalize_src)

    if "log_level" in canonical_df.columns:
        canonical_df["log_level"] = canonical_df["log_level"].apply(_normalize_log_level)

    for col in ["phase", "state", "substate"]:
        canonical_df[col] = canonical_df[col].apply(_extract_bracket_code)

    if "custom_data" in canonical_df.columns:
        canonical_df["custom_data"] = canonical_df["custom_data"].apply(_clean_value)

    if "activity" in canonical_df.columns:
        invalid_act_markers = {"-", "nan", "none", "", "null", "<na>"}
        
        def assign_activity_fallback(row):
            act_val = str(row["activity"]).strip()
            src_val = str(row["extra_srcinfo"]).strip()
            
            if act_val.lower() in invalid_act_markers or pd.isna(row["activity"]):
                if src_val in {"ACT_SYS", "SYSTEM_CORE"}:
                    return "ACT_SYS_INITIALIZATION"
                else:
                    return "ACT_UNKNOWN_MODAL"
            return row["activity"]

        canonical_df["activity"] = canonical_df.apply(assign_activity_fallback, axis=1)

    return canonical_df