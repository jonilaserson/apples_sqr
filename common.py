# Standardized column names
GT_COL = "_GT_"
SCORE_COL = "_score_"

def format_query_stats(query, total_subset, pos_count, total_samples):
    """Format query statistics with aligned left brackets."""
    neg_count = total_subset - pos_count
    percentage = (total_subset / total_samples) * 100 if total_samples > 0 else 0

    # Format the statistics part with consistent width
    stats = f"[{neg_count:3d}:{pos_count:3d}] {percentage:5.1f}% "
    return f"{stats} {query}"

def get_meta_columns_in_order(df):
    seen = []
    for ns in df.columns.get_level_values(0):
        if ns not in seen:
            seen.append(ns)
    return seen

def cache_data(func=None, **kwargs):
    try:
        import streamlit as st
        return st.cache_data(**kwargs)(func)
    except ImportError:
        return func 