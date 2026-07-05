"""Centralized session-state helpers so every page reads/writes the same data."""
import streamlit as st
import pandas as pd


def init_state():
    defaults = {
        "raw_df": None,          # original uploaded data, never modified
        "df": None,              # working (cleaned) data
        "filename": None,
        "history": [],           # list of (action_description, df_snapshot) for undo
        "encoders": {},          # store fitted encoders/scalers for reference
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


def push_history(action: str):
    """Call BEFORE mutating st.session_state.df to allow undo."""
    if st.session_state.df is not None:
        st.session_state.history.append((action, st.session_state.df.copy()))
        # keep history bounded
        if len(st.session_state.history) > 20:
            st.session_state.history.pop(0)


def undo_last():
    if st.session_state.history:
        action, snapshot = st.session_state.history.pop()
        st.session_state.df = snapshot
        return action
    return None


def has_data() -> bool:
    return st.session_state.get("df") is not None
