import os
import webview
import threading
import pandas as pd
import numpy as np
import re
from pipeline.make_canonical import make_canonical, _clean_message
class VeraAPI:
    def __init__(self):
        self.active_df_canonical = None

    def _get_window(self):
        """Helper to safely fetch the primary UI frame window handle."""
        return webview.windows[0] if webview.windows else None

    def browse_and_process_file(self):
        window = self._get_window()
        if not window:
            return {"status": "error", "message": "Webview window not initialized"}

        dialog_result = []
        def run_open_dialog():
            res = window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("CSV Files (*.csv)", "Text Log Files (*.txt;*.log)")
            )
            dialog_result.append(res)

        t = threading.Thread(target=run_open_dialog)
        t.start()
        t.join()

        result = dialog_result[0] if dialog_result else None
        if not result:
            return {"status": "cancelled"}

        filepath = result[0] if isinstance(result, (list, tuple)) else result
        filename = os.path.basename(filepath)

        try:
            df = make_canonical(filepath)

            total_records = len(df)
            if total_records == 0:
                self.active_df_canonical = df
                return {
                    "status": "success",
                    "filename": filename,
                    "metrics": {
                        "total": "0", "valid": "0", "invalid": "0",
                        "phase_transitions": "0", "conform": "0",
                        "active_attrs": "0", "unique_activities": "0",
                        "sparsity": "0.00 min"
                    },
                    "preview_records": [],
                    "invalid_preview_records": [],
                    "discovered_maps": {}
                }

            discovered_maps = {}
            try:
                from pipeline.make_canonical import _detect_skiprows, CSV_HEADER_MAP
                skip = _detect_skiprows(filepath)
                raw_df = pd.read_csv(filepath, skiprows=skip, sep=None, engine="python", nrows=1)
                raw_cols = [str(c).strip() for c in raw_df.columns]
                
                for canonical_field, target_csv_name in CSV_HEADER_MAP.items():
                    target_lower = target_csv_name.lower()
                    target_alpha = re.sub(r"\W+", "", target_lower)
                    matched_raw = None
                    for col in raw_cols:
                        col_lower = col.lower()
                        col_alpha = re.sub(r"\W+", "", col_lower)
                        if col_lower == target_lower or col_alpha == target_alpha or (target_alpha in col_alpha) or (col_alpha in target_alpha):
                            matched_raw = col
                            break
                    if matched_raw is None and canonical_field == "timestamp":
                        for col in raw_cols:
                            if "time" in col.lower():
                                matched_raw = col
                                break
                    discovered_maps[canonical_field] = matched_raw if matched_raw else "-"
            except Exception:
                discovered_maps = {k: "-" for k in CSV_HEADER_MAP.keys()}

            def get_extracted_numeric(col_name):
                if col_name not in df.columns:
                    return pd.Series(np.nan, index=df.index)
                clean_text = df[col_name].astype(str).str.replace(r"[\[\]\s]", "", regex=True)
                clean_text = clean_text.replace(["nan", "NaN", "null", "NULL", "None", "-", "", "<na>"], np.nan)
                return pd.to_numeric(clean_text, errors='coerce')

            log_level_num = get_extracted_numeric("log_level")
            valid_log_level = log_level_num.isna() | log_level_num.between(0, 7)

            category_num = get_extracted_numeric("event_category")
            valid_category = category_num.isna() | category_num.between(0, 10)

            type_num = get_extracted_numeric("event_type")
            valid_type = type_num.isna() | type_num.between(0, 4)

            phase_num = get_extracted_numeric("phase")
            valid_phase = phase_num.isna() | phase_num.between(0, 9)

            state_num = get_extracted_numeric("state")
            valid_state = state_num.isna() | state_num.between(0, 11)

            substate_num = get_extracted_numeric("substate")
            valid_substate = substate_num.isna() | substate_num.between(0, 9)

            if "extra_srcinfo" in df.columns:
                valid_srcinfo = ~df["extra_srcinfo"].astype(str).str.contains("SRC_UNKNOWN", na=False)
            else:
                valid_srcinfo = pd.Series(True, index=df.index)

            is_row_valid = (
                valid_log_level & valid_category & valid_type & 
                valid_phase & valid_state & valid_substate & valid_srcinfo
            )
            
            df_valid = df[is_row_valid].copy()
            df_invalid = df[~is_row_valid].copy()

            if len(df_valid) > 0 and "message" in df_valid.columns:
                df_valid["message"] = df_valid["message"].apply(_clean_message)

            self.active_df_canonical = pd.concat([df_valid, df_invalid]).sort_index()

            b1_total = int(total_records)
            b2_valid = int(len(df_valid))
            b3_invalid = int(max(0, b1_total - b2_valid))
            
            depth_num = get_extracted_numeric("depth")
            depth_valid = depth_num[is_row_valid]
            b4_platform = int((depth_valid == 0).sum())
            b5_system = int((depth_valid == 1).sum())

            b6_active_attrs = sum(1 for col in df.columns if df[col].astype(str).str.strip().replace(["nan", "NaN", "null", "NULL", "None", "-", ""], np.nan).notna().any())

            clean_phase_series = df_valid["phase"].astype(str).str.replace(r"[\[\]\s]", "", regex=True)
            clean_phase_series = clean_phase_series.replace(["nan", "NaN", "null", "NULL", "None", "-", ""], np.nan).dropna()
            b7_phases = int(clean_phase_series.nunique())

            try:
                ts_series = pd.to_numeric(df["timestamp"], errors='coerce').dropna()
                if not ts_series.empty:
                    t_min, t_max = float(ts_series.min()), float(ts_series.max())
                    raw_range = t_max - t_min
                    if raw_range > 0:
                        span_ms = raw_range * 1000.0 if t_max < 1e12 else raw_range
                        b8_span = f"{(span_ms / 60000.0):.2f} min" if span_ms >= 60000.0 else f"{(span_ms / 1000.0):.2f} sec"
                    else:
                        b8_span = "0.00 min"
                else:
                    b8_span = "0.00 min"
            except Exception:
                b8_span = "0.00 min"

            preview_records = df_valid.head(20).fillna("-").to_dict(orient="records")
            invalid_preview_records = df_invalid.head(20).fillna("-").to_dict(orient="records")

            return {
                "status": "success",
                "filename": filename,
                "metrics": {
                    "total": str(b1_total), "valid": str(b2_valid), "invalid": str(b3_invalid),
                    "phase_transitions": str(b4_platform), "conform": str(b5_system),
                    "active_attrs": str(b6_active_attrs), "unique_activities": str(b7_phases), "sparsity": b8_span
                },
                "preview_records": preview_records,
                "invalid_preview_records": invalid_preview_records,
                "discovered_maps": discovered_maps
            }
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def get_attribute_assessment(self):
        """
        Runs Stage 2 integrity analytics using the currently cached active dataframe.
        """
        if self.active_df_canonical is None or self.active_df_canonical.empty:
            return {"status": "error", "message": "No active dataframe slice found. Please load a session log first."}
        try:
            from pipeline.attribute_assessment import generate_full_attribute_matrix
            matrix = generate_full_attribute_matrix(self.active_df_canonical)
            return {"status": "success", "results": matrix}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_assessment_chart_data(self, attribute_key: str, method: str, chart_type: str):
        if self.active_df_canonical is None or self.active_df_canonical.empty:
            return {"type": "empty", "verdict": "info", "reason": "No active dataset loaded. Please import a session log file."}
        try:
            from pipeline.attribute_assessment import get_assessment_chart_data
            return get_assessment_chart_data(self.active_df_canonical, attribute_key, method, chart_type)
        except Exception as e:
            return {"type": "empty", "verdict": "info", "reason": str(e)}

    def export_canonical_file(self):
        if self.active_df_canonical is None:
            return {"status": "error", "message": "No active dataset loaded."}
        window = self._get_window()
        if not window:
            return {"status": "error", "message": "Webview window not found."}

        dialog_result = []
        def run_save_dialog():
            res = window.create_file_dialog(
                webview.SAVE_DIALOG, 
                save_filename="canonical_clean_export.csv", 
                file_types=("CSV Files (*.csv)",)
            )
            dialog_result.append(res)

        t = threading.Thread(target=run_save_dialog)
        t.start()
        t.join()

        result = dialog_result[0] if dialog_result else None
        if not result:
            return {"status": "cancelled"}

        path = result[0] if isinstance(result, (list, tuple)) else result
        try:
            def get_extracted_numeric(col_name):
                if col_name not in self.active_df_canonical.columns:
                    return pd.Series(np.nan, index=self.active_df_canonical.index)
                clean_text = self.active_df_canonical[col_name].astype(str).str.replace(r"[\[\]\s]", "", regex=True)
                return pd.to_numeric(clean_text, errors='coerce')

            if "extra_srcinfo" in self.active_df_canonical.columns:
                valid_srcinfo = ~self.active_df_canonical["extra_srcinfo"].astype(str).str.contains("SRC_UNKNOWN", na=False)
            else:
                valid_srcinfo = pd.Series(True, index=self.active_df_canonical.index)

            is_row_valid = (
                (get_extracted_numeric("log_level").isna() | get_extracted_numeric("log_level").between(0, 7)) &
                (get_extracted_numeric("event_category").isna() | get_extracted_numeric("event_category").between(0, 10)) &
                (get_extracted_numeric("event_type").isna() | get_extracted_numeric("event_type").between(0, 4)) &
                (get_extracted_numeric("phase").isna() | get_extracted_numeric("phase").between(0, 9)) &
                (get_extracted_numeric("state").isna() | get_extracted_numeric("state").between(0, 11)) &
                (get_extracted_numeric("substate").isna() | get_extracted_numeric("substate").between(0, 9)) &
                valid_srcinfo
            )
            clean_df = self.active_df_canonical[is_row_valid]
            clean_df.to_csv(path, index=False)
            return {"status": "success", "message": f"Clean dataset exported successfully ({len(clean_df)} rows):\n{path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
    def get_f1_chart_data(self, chart_type="line"):
        """
        Stage 3 Live Feature Assessment Bridge:
        Feeds the shared global canonical dataset directly into the multi-attribute tracker.
        """
        from pipeline.feature_assessment import get_f1_assessment
        
        if self.active_df_canonical is None or self.active_df_canonical.empty:
            return {
                "type": "empty",
                "verdict": "INFO",
                "metric": "0.000s",
                "x": [],
                "y": [],
                "inference": "No data source active. Please import a trace source file in Stage 1 first."
            }
            
        return get_f1_assessment(self.active_df_canonical, chart_type=chart_type)