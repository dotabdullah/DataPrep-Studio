"""
Cleaning "recipes": a recorded, replayable sequence of the cleaning steps applied in a session.
A recipe bundles the step list together with any fitted objects (LabelEncoders, scalers, PCA)
so that replaying it on a new file reuses the exact same transformations rather than refitting
fresh ones — important for keeping feature encoding consistent with an already-trained model.
"""
import io
import numpy as np
import pandas as pd
import joblib


def record_step(step_type: str, **kwargs) -> dict:
    """Build a structured recipe step. Call this alongside push_history() for any cleaning action."""
    return {"type": step_type, **kwargs}


def export_recipe_bytes(recipe_steps, encoders, onehot_columns, value_labels=None) -> bytes:
    """Serialize the recipe (steps + fitted encoders/scalers/PCA + one-hot metadata + value labels) to bytes."""
    payload = {
        "steps": recipe_steps, "encoders": encoders, "onehot_columns": onehot_columns,
        "value_labels": value_labels or {},
    }
    buf = io.BytesIO()
    joblib.dump(payload, buf)
    buf.seek(0)
    return buf.getvalue()


def load_recipe_bytes(file_bytes: bytes) -> dict:
    """Deserialize a previously-downloaded recipe file."""
    buf = io.BytesIO(file_bytes)
    return joblib.load(buf)


def apply_recipe(df: pd.DataFrame, recipe_payload: dict):
    """Replay a recipe's steps on a fresh dataframe.

    Returns (new_df, new_encoders, new_onehot_columns, new_value_labels, warnings_list).
    Steps that reuse a fitted object (label encoding, scaling, PCA) pull that object from the
    recipe itself, so a category/column that existed during recording is handled consistently
    even if it's rare or absent in the new data.
    """
    steps = recipe_payload.get("steps", [])
    stored_encoders = recipe_payload.get("encoders", {})
    stored_onehot = recipe_payload.get("onehot_columns", {})
    stored_value_labels = recipe_payload.get("value_labels", {})

    new_df = df.copy()
    new_encoders = {}
    new_onehot = {}
    new_value_labels = {}
    warnings = []

    for step in steps:
        stype = step.get("type")
        try:
            if stype == "drop_columns":
                cols = [c for c in step["columns"] if c in new_df.columns]
                if cols:
                    new_df = new_df.drop(columns=cols)

            elif stype == "rename_column":
                if step["old_name"] in new_df.columns:
                    new_df = new_df.rename(columns={step["old_name"]: step["new_name"]})

            elif stype == "dropna_rows":
                cols = [c for c in step["columns"] if c in new_df.columns]
                if cols:
                    new_df = new_df.dropna(subset=cols)

            elif stype == "fillna":
                for c in step["columns"]:
                    if c not in new_df.columns:
                        continue
                    method = step["method"]
                    if method == "mean" and pd.api.types.is_numeric_dtype(new_df[c]):
                        new_df[c] = new_df[c].fillna(new_df[c].mean())
                    elif method == "median" and pd.api.types.is_numeric_dtype(new_df[c]):
                        new_df[c] = new_df[c].fillna(new_df[c].median())
                    elif method == "mode":
                        m = new_df[c].mode()
                        if not m.empty:
                            new_df[c] = new_df[c].fillna(m.iloc[0])
                    elif method == "constant":
                        new_df[c] = new_df[c].fillna(step.get("value"))
                    elif method == "ffill":
                        new_df[c] = new_df[c].ffill()
                    elif method == "bfill":
                        new_df[c] = new_df[c].bfill()

            elif stype == "drop_duplicates":
                subset = step.get("subset") or None
                subset = [c for c in subset if c in new_df.columns] if subset else None
                keep = step.get("keep", "first")
                new_df = new_df.drop_duplicates(subset=subset, keep=keep)

            elif stype == "convert_dtype":
                c, new_type = step["column"], step["new_type"]
                if c in new_df.columns:
                    if new_type == "int64":
                        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").astype("Int64")
                    elif new_type == "float64":
                        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").astype("float64")
                    elif new_type == "str (object)":
                        new_df[c] = new_df[c].astype(str)
                    elif new_type == "category":
                        new_df[c] = new_df[c].astype("category")
                    elif new_type == "datetime":
                        new_df[c] = pd.to_datetime(new_df[c], errors="coerce")

            elif stype == "outlier":
                c = step["column"]
                if c in new_df.columns and pd.api.types.is_numeric_dtype(new_df[c]):
                    series = new_df[c].dropna()
                    if step["method"] == "iqr":
                        q1, q3 = series.quantile(0.25), series.quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    else:
                        mean, std = series.mean(), series.std()
                        lower, upper = mean - 3 * std, mean + 3 * std
                    mask = (new_df[c] < lower) | (new_df[c] > upper)
                    if step["action"] == "remove":
                        new_df = new_df[~mask]
                    else:
                        new_df[c] = new_df[c].clip(lower=lower, upper=upper)

            elif stype == "label_encode":
                c = step["column"]
                le = stored_encoders.get(c)
                if c in new_df.columns and le is not None:
                    known = set(str(v) for v in le.classes_)
                    col_vals = new_df[c].astype(object).astype(str)
                    unseen_mask = ~col_vals.isin(known)
                    if unseen_mask.any():
                        warnings.append(f"'{c}': {int(unseen_mask.sum())} row(s) had categories not seen "
                                        f"originally and were dropped.")
                    new_df = new_df[~unseen_mask].copy()
                    new_df[c] = le.transform(col_vals[~unseen_mask])
                    new_encoders[c] = le
                elif c in new_df.columns:
                    warnings.append(f"'{c}': no saved encoder found in this recipe — left unchanged.")

            elif stype == "onehot_encode":
                c = step["column"]
                ohinfo = stored_onehot.get(c)
                if c in new_df.columns and ohinfo is not None:
                    cat_to_col = ohinfo["category_to_column"]
                    dummy_cols = ohinfo["dummy_columns"]
                    for dcol in dummy_cols:
                        new_df[dcol] = 0
                    for idx, val in new_df[c].astype(str).items():
                        target_col = cat_to_col.get(val)
                        if target_col:
                            new_df.at[idx, target_col] = 1
                        else:
                            warnings.append(f"'{c}': value '{val}' not seen originally (row left as all-zero).")
                    new_df = new_df.drop(columns=[c])
                    new_onehot[c] = ohinfo
                elif c in new_df.columns:
                    warnings.append(f"'{c}': no saved one-hot mapping found in this recipe — left unchanged.")

            elif stype == "scale":
                cols = step["columns"]
                scaler_key = step["scaler_key"]
                scaler = stored_encoders.get(scaler_key)
                cols_present = [c for c in cols if c in new_df.columns]
                if scaler is not None and cols_present:
                    new_df[cols_present] = scaler.transform(new_df[cols_present])
                    new_encoders[scaler_key] = scaler
                elif cols_present:
                    warnings.append(f"No saved scaler found for {cols_present} — left unchanged.")

            elif stype == "text_cleanup":
                cols, ops = step["columns"], step["ops"]
                for c in cols:
                    if c not in new_df.columns:
                        continue
                    s = new_df[c].astype(str)
                    for op in ops:
                        if op == "strip":
                            s = s.str.strip()
                        elif op == "lower":
                            s = s.str.lower()
                        elif op == "upper":
                            s = s.str.upper()
                        elif op == "collapse_spaces":
                            s = s.str.replace(r"\s+", " ", regex=True)
                    new_df[c] = s

            elif stype == "pca":
                cols = step["columns"]
                n_components = step["n_components"]
                drop_originals = step["drop_originals"]
                pca_key = step["pca_key"]
                pca_model = stored_encoders.get(pca_key)
                cols_present = [c for c in cols if c in new_df.columns]
                if pca_model is not None and len(cols_present) == len(cols):
                    work_pca = new_df[cols].dropna()
                    scaler = stored_encoders.get(f"{pca_key}_scaler")
                    X_input = scaler.transform(work_pca) if scaler is not None else work_pca.values
                    transformed = pca_model.transform(X_input)
                    comp_cols = [f"PC{i + 1}" for i in range(n_components)]
                    comp_df = pd.DataFrame(transformed, columns=comp_cols, index=work_pca.index)
                    for cc in comp_cols:
                        new_df[cc] = comp_df[cc]
                    if drop_originals:
                        new_df = new_df.drop(columns=cols)
                    new_encoders[pca_key] = pca_model
                    if scaler is not None:
                        new_encoders[f"{pca_key}_scaler"] = scaler
                else:
                    warnings.append(f"Couldn't replay PCA step — required columns {cols} missing or "
                                    f"no saved PCA model found.")
            elif stype == "value_labels":
                c = step["column"]
                if c in new_df.columns:
                    new_value_labels[c] = step["labels"]
                else:
                    warnings.append(f"'{c}': column not found — skipped its saved value labels.")

        except Exception as e:
            warnings.append(f"Step '{stype}' failed: {e}")
            continue

    return new_df, new_encoders, new_onehot, new_value_labels, warnings
