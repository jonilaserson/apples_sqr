#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path
import time
import matplotlib.pyplot as plt

def cache_data(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func

try:
    import streamlit as st
    st.set_page_config(layout="wide")
    HAS_STREAMLIT = True
    cache_data = st.cache_data
except ImportError:
    HAS_STREAMLIT = False
    # No Streamlit installed

def compute_max_f1(y_true, y_score):
    thresholds = np.unique(y_score)
    return max((f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds), default=None)


METRIC_FUNCTIONS = {
    'auc': roc_auc_score,
    'max_f1': compute_max_f1,
}

THRESH_METRICS_FUNCTIONS = {
    'precision': precision_score,
    'recall': recall_score,
    'accuracy': accuracy_score,
    'f1': f1_score,
}

def format_query_stats(query, total_subset, pos_count, total_samples):
    """Format query statistics with aligned left brackets."""
    neg_count = total_subset - pos_count
    percentage = (total_subset / total_samples) * 100 if total_samples > 0 else 0

    # Format the statistics part with consistent width
    stats = f"[{neg_count:3d}:{pos_count:3d}] {percentage:5.1f}% "
    return f"{stats} {query}"


def validate_data(test_df, model_dfs):
    pos_count = test_df['GT'].sum()
    neg_count = len(test_df) - pos_count

    if pos_count == 0 or neg_count == 0:
        raise ValueError("Error: Test set must contain both positive and negative samples after filtering")

    """Validate test and model data compatibility."""
    for model_name, model_df in model_dfs.items():
        # Check for duplicate IDs in model data
        if model_df.index.duplicated().any():
            raise ValueError(f"Model {model_name} contains duplicate IDs")

        # Check if model contains all test IDs
        missing_ids = set(test_df.index) - set(model_df.index)
        if missing_ids:
            raise ValueError(f"Model {model_name} is missing {len(missing_ids)} IDs from the test set")

def join_with_namespace(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    right_ns: str,
    left_ns: str = None
) -> pd.DataFrame:
    """
    Merge two DataFrames by index, adding a namespace to df_right columns
    as the top level of a MultiIndex.

    Assumes:
    - df_left and df_right have their joining key set as index.
    - Always hierarchical columns.
    """

    df_right_named = df_right.copy()
    df_right_named.columns = pd.MultiIndex.from_product([[right_ns], df_right_named.columns])

    # Make sure df_left columns are MultiIndex too
    df_left_named = df_left.copy()
    if not isinstance(df_left_named.columns, pd.MultiIndex):
        df_left_named.columns = pd.MultiIndex.from_product([[left_ns], df_left_named.columns])

    return df_left_named.join(df_right_named)


@cache_data
def load_models(test_file, model_files, queries_file=None, filter_query=None, model_names=None):
    if model_names is None:
        model_names = [Path(mf).stem for mf in model_files]
    model_files = dict(zip(model_names, model_files))
    test_df = pd.read_csv(test_file).set_index("id")

    if filter_query:
        original_size = len(test_df)
        test_df = test_df.query(filter_query)
        filtered_size = len(test_df)
        print(f"Applied filter: '{filter_query}'")
        print(f"Filtered test set from {original_size} to {filtered_size} samples ({filtered_size/original_size*100:.1f}%)")

    # Set index to 'id'
    model_dfs = {model_name: pd.read_csv(model_file).set_index("id") for model_name, model_file in model_files.items()}

    validate_data(test_df, model_dfs)

    # Merge each model with test_df
    models_df = test_df
    for model_name, model_df in model_dfs.items():
        # No need to clean columns anymore
        models_df = join_with_namespace(models_df, model_df, right_ns=model_name, left_ns="test")

    # Remove queries that do not have both classes.
    queries = ['all']
    if queries_file is not None:
        with open(queries_file, 'r') as f:
            queries.extend([q.strip() for q in f.readlines() if q.strip()])

    valid_queries = {}
    total_samples = len(test_df)
    for query in queries:
        subset_df = test_df.copy() if query == 'all' else test_df.query(query)
        pos_count = subset_df['GT'].sum()  # Assuming all models share same GT
        neg_count = len(subset_df) - pos_count
        query_stat = format_query_stats(query, len(subset_df), pos_count, total_samples)
        has_both_classes = pos_count > 0 and neg_count > 0
        if has_both_classes:
            valid_queries[query] = query_stat

    return models_df, valid_queries

def get_meta_columns_in_order(df):
    seen = []
    for ns in df.columns.get_level_values(0):
        if ns not in seen:
            seen.append(ns)
    return seen

@cache_data
def compute_metrics(models_df, queries, metrics, thresholds=None):
    raw_results = {}
    model_names = get_meta_columns_in_order(models_df)[1:]

    for query in queries:
        subset_df = models_df if query == 'all' else models_df.loc[models_df["test"].query(query).index]
        y_true = subset_df["test"]['GT']
        for idx, model_name in enumerate(model_names):
            if model_name == "test":
                continue
            model_data = subset_df[model_name]
            y_pred = model_data['score']
            if thresholds is not None:
                threshold = thresholds[idx]
                y_pred = (y_pred >= threshold).astype(int)

            metric_functions = THRESH_METRICS_FUNCTIONS if thresholds is not None else METRIC_FUNCTIONS
            for metric_name in metrics:
                metric_func = metric_functions.get(metric_name)
                if metric_func is not None:
                    raw_results.setdefault(metric_name, {}).setdefault(model_name, {})
                    metric_value = metric_func(y_true, y_pred)
                    raw_results[metric_name][model_name][query] = metric_value

    return raw_results

def raw_results_to_final_df(raw_results, model_names, metrics, queries: dict):
    columns = pd.MultiIndex.from_product([metrics, model_names], names=['Metric', 'Model'])
    final_df = pd.DataFrame(index=list(queries.values()), columns=columns)

    for metric_name in metrics:
        for model_name in model_names:
            for query, query_label in queries.items():
                value = raw_results.get(metric_name, {}).get(model_name, {}).get(query, None)
                final_df.loc[query_label, (metric_name, model_name)] = value

    return final_df

def apply_thresholds_and_evaluate(models_df, queries_dict, metrics, thresholds):
    raw_results = compute_metrics(models_df, list(queries_dict.keys()), metrics)
    raw_results_thresholds = compute_metrics(models_df, list(queries_dict.keys()), metrics, thresholds)
    raw_results.update(raw_results_thresholds)
    model_names = get_meta_columns_in_order(models_df)[1:]
    final_df = raw_results_to_final_df(raw_results, model_names, metrics, queries_dict)
    return final_df

def display_results(results, metrics, flatten=False):
    if results.empty:
        print("No results to display.")
        return

    if flatten:
        flat_df = results.copy()
        flat_df.columns = pd.MultiIndex.from_tuples(flat_df.columns)
        flat_df.columns = flat_df.columns.swaplevel(0, 1)
        flat_df.columns.names = ['Model', 'Metric']
        flat_df.columns = [f"{metric}\n{model}" for model, metric in flat_df.columns]
        print("\nEvaluation Results:")
        print(flat_df.to_markdown(floatfmt=".4f"))
    else:
        for metric in metrics:
            if metric in results.columns.levels[0]:
                metric_df = results.xs(metric, level='Metric', axis=1)
                print(f"\nEvaluation results for metric \"{metric}\":")
                print(metric_df.to_markdown(floatfmt=".4f"))



def compute_confusion_elements(y_true, y_pred):
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    return tp, fp, fn, tn


def plot_roc_curve_for_model(y_true, y_score, threshold, color, model_name, ax=None, title=None, figsize=(4, 4)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    model_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={model_auc:.2f})", color=color)

    y_pred = (y_score >= threshold).astype(int)
    tp, fp, fn, tn = compute_confusion_elements(y_true, y_pred)

    if (tp + fn) > 0 and (fp + tn) > 0:
        operation_fpr = fp / (fp + tn)
        operation_tpr = tp / (tp + fn)
        ax.plot(operation_fpr, operation_tpr, 'o', color=color)

    if fig is not None: # Created new fig
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title or "ROC Curve")
        ax.grid(True)
        plt.tight_layout()

    ax.legend(loc='lower right', fontsize=8)
    return ax.figure, ax



def plot_pr_curve_for_model(y_true, y_score, threshold, color, model_name, ax=None, title=None, figsize=(4, 4)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.2f})", color=color)

    y_pred = (y_score >= threshold).astype(int)
    operation_precision = precision_score(y_true, y_pred, zero_division=0)
    operation_recall = recall_score(y_true, y_pred, zero_division=0)

    ax.plot(operation_recall, operation_precision, 'o', color=color)

    if fig is not None: # Created new fig
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title or 'Precision-Recall Curve')
        ax.grid(True)
        plt.tight_layout()

    ax.legend(loc='lower left', fontsize=8)
    return ax.figure, ax


def plot_curves(subset_df, thresholds, colors, curve_type='roc', query_string=None, figsize=(5, 4)):
    """
    Plot either ROC or PR curves for all models on the same axis.

    Args:
        subset_df: Subset of the merged test and model dataframe.
        thresholds: List of thresholds, one per model.
        colors: List of colors to use.
        curve_type: 'roc' or 'pr'
        query_string: The query string to display as a subtitle
        figsize: Tuple of (width, height) for the figure size
    Returns:
        fig, ax: The figure and axis with all models plotted.
    """
    fig, ax = None, None

    model_names = get_meta_columns_in_order(subset_df)[1:]
    title = ':'.join([curve_type.upper(), query_string])
    for idx, model_name in enumerate(model_names):
        y_true = subset_df['test']['GT']
        y_score = subset_df[model_name]['score']
        threshold = thresholds[idx]
        color = colors[idx % len(colors)]

        if curve_type == 'roc':
            fig, ax = plot_roc_curve_for_model(y_true, y_score, threshold, color, model_name, ax=ax, title=title, figsize=figsize)
        elif curve_type == 'pr':
            fig, ax = plot_pr_curve_for_model(y_true, y_score, threshold, color, model_name, ax=ax, title=title, figsize=figsize)
        else:
            raise ValueError(f"Unknown curve_type: {curve_type}")

    return fig, ax

def plot_roc_and_pr_curves_for_streamlit(models_df, thresholds, query='all'):
    # Prepare subset
            # Get the actual query string from the label (extracting from formatted label)
    query_string = query
    if isinstance(query, str) and query.startswith('['):
        parts = query.split('% ', 1)
        if len(parts) == 2:
            query = parts[1].strip()

    subset_df = models_df if query == 'all' else models_df.loc[models_df["test"].query(query).index]

    colors = plt.cm.tab10.colors

    st.markdown(f"### ROC and Precision-Recall Curves")

    col1, col2 = st.columns(2)

    # --- ROC Plot ---
    with col1:
        fig1, ax1 = plot_curves(subset_df, thresholds, colors, curve_type='roc', query_string=query_string)
        st.pyplot(fig1)

    # --- PR Plot ---
    with col2:
        fig2, ax2 = plot_curves(subset_df, thresholds, colors, curve_type='pr', query_string=query_string)
        st.pyplot(fig2)

def transform_df_for_model_view(df, selected_query):
    """
    Transform the results dataframe to show models as rows and metrics as columns
    for the selected query.
    """
    # Extract the data for the selected query
    query_data = df.loc[selected_query]

    # Create a new dataframe where models are rows and metrics are columns
    # This effectively unstacks the original multi-index columns
    model_view_df = query_data.unstack(level=0)

    # Clean up the index/columns for better display
    model_view_df.index.name = 'Model'
    model_view_df.columns.name = 'Metric'

    return model_view_df


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Evaluate binary classifiers on test data')
    parser.add_argument('-t', '--test', required=True, help='Path to test set CSV file with ground truth')
    parser.add_argument('-m', '--models', required=True, nargs='+', help='Paths to model prediction CSV files')
    parser.add_argument('-q', '--queries', help='Path to queries text file')
    parser.add_argument('-f', '--filter', help='Initial filter query to apply to test set')
    parser.add_argument('--metrics', default='auc', help='Comma-separated list of metrics to compute (auc,precision,recall,accuracy,f1,max_f1)')
    parser.add_argument('--thresh', type=float, nargs='+', help='Threshold(s) for binary classification metrics. One value or one per model.')
    parser.add_argument('--flatten', action='store_true', help='Flatten the result table instead of showing separate tables per metric')
    parser.add_argument('--gui', action='store_true', help='Launch interactive GUI for threshold tuning')

    args = parser.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(',')]
    METRICS = dict(**METRIC_FUNCTIONS, **THRESH_METRICS_FUNCTIONS)
    invalid_metrics = [m for m in metrics if m not in METRICS]
    if invalid_metrics:
        print(f"Error: Invalid metrics specified: {', '.join(invalid_metrics)}")
        print(f"Available metrics: {', '.join((METRICS).keys())}")
        return

    if args.thresh and len(args.thresh) != 1 and len(args.thresh) != len(args.models):
        print(f"Error: Number of thresholds ({len(args.thresh)}) does not match number of models ({len(args.models)})")
        return

    # Load the data once
    models_df, queries = load_models(
        args.test,
        args.models,
        args.queries,
        filter_query=args.filter
    )

    if args.thresh:
        thresh_values = args.thresh if len(args.thresh) > 1 else [args.thresh[0]] * len(args.models)
    else:
        thresh_values = [0.5] * len(args.models)

    if not args.gui:
        results = apply_thresholds_and_evaluate(models_df, queries, metrics, thresh_values)
        if results is not None and not results.empty:
            if args.filter:
                print(f"Initial filter applied: '{args.filter}'")

            display_results(results, metrics, flatten=args.flatten)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n\nTotal evaluation time: {duration:.2f} seconds")

    else:
        assert HAS_STREAMLIT, "--gui mode requested but streamlit is not installed"

        # Configure pandas display options to show all columns and fix decimal places
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', '{:.03f}'.format)

        st.title("Interactive Model Evaluator")

        threshold_inputs = []
        st.sidebar.header("Thresholds per model")
        model_names = get_meta_columns_in_order(models_df)[1:]
        for i, model_name in enumerate(model_names):
            value = st.sidebar.slider(f"Threshold for {model_name}", min_value=0.0, max_value=1.0, step=0.05, value=float(thresh_values[i]))
            threshold_inputs.append(value)

        results = apply_thresholds_and_evaluate(models_df, queries, metrics, threshold_inputs)

        query_labels = results.index.tolist()
        #plot_container = st.container()
        table_container, plot_container = st.columns([3, 5])

        # Place the dropdown and radio on the same row
        col1, col2 = st.columns([4, 2])
        with col1:
            selected_query_label = st.selectbox(
                "Select a query to visualize:",
                options=query_labels,
                index=0  # pre-select the first query
            )
        with col2:
            view_toggle = st.radio(
                "Select view type:",
                options=["Query-indexed view", "Model-indexed view"],
                horizontal=True
            )

        # Now plot, based on the *selected* query
        with plot_container:
            plot_roc_and_pr_curves_for_streamlit(models_df, threshold_inputs, query=selected_query_label)

        # Show the appropriate dataframe based on the selected view
        if view_toggle == "Model-indexed view":
            results = transform_df_for_model_view(results, selected_query_label)

        with table_container:
            st.subheader(view_toggle)
            st.dataframe(
                results,
                use_container_width=False,
                height=400
            )

if __name__ == "__main__":
    main()
