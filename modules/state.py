"""Centralized session-state helpers so every page reads/writes the same data."""
import streamlit as st
import pandas as pd


def init_state():
    defaults = {
        "raw_df": None,          # original uploaded data, never modified
        "df": None,              # working (cleaned) data
        "filename": None,
        "history": [],           # list of (action_description, df_snapshot, recipe_step_or_None) for undo
        "encoders": {},          # store fitted LabelEncoders/scalers, keyed by column name
        "onehot_columns": {},    # one-hot metadata: {original_col: {"dummy_columns": [...], "category_to_column": {...}}}
        "value_labels": {},      # manual display labels for already-numeric columns: {col: {value: label}}
        "recipe": [],            # structured, replayable list of cleaning steps (see modules/recipe.py)
        "trained_model": None,
        "model_info": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_new_dataset(df: pd.DataFrame, filename: str):
    st.session_state.raw_df = df.copy()
    st.session_state.df = df.copy()
    st.session_state.filename = filename
    st.session_state.history = []
    st.session_state.trained_model = None
    st.session_state.model_info = {}
    st.session_state.encoders = {}
    st.session_state.onehot_columns = {}
    st.session_state.value_labels = {}
    st.session_state.recipe = []


def push_history(action: str, steps=None):
    """Call BEFORE mutating st.session_state.df to allow undo.
    `steps`: None, a single dict, or a list of dicts (from modules.recipe.record_step) to also
    make this action replayable as part of a downloadable cleaning recipe."""
    if isinstance(steps, dict):
        steps = [steps]
    if st.session_state.df is not None:
        st.session_state.history.append((action, st.session_state.df.copy(), steps))
        if len(st.session_state.history) > 20:
            st.session_state.history.pop(0)
    if steps:
        st.session_state.recipe.extend(steps)


def undo_last():
    if st.session_state.history:
        action, snapshot, steps = st.session_state.history.pop()
        st.session_state.df = snapshot
        if steps:
            n = len(steps)
            if n and len(st.session_state.recipe) >= n and st.session_state.recipe[-n:] == steps:
                del st.session_state.recipe[-n:]
        return action
    return None


def has_data() -> bool:
    return st.session_state.get("df") is not None
