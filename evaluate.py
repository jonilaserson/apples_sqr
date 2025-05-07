#!/usr/bin/env python3
import argparse
import time


# Import and configure Streamlit first if available
try:
    import streamlit as st
    st.set_page_config(layout="wide")
    import pandas as pd
except ImportError:
    st = None
    
from data_loading import load_models
from metrics import compute_metrics, raw_results_to_final_df
from display import display_results, transform_df_for_model_view, setup_streamlit_display
from common import get_meta_columns_in_order


def validate_thresholds(thresholds: list[float], n_models: int) -> None:
    """Validate that number of thresholds matches number of models."""
    if thresholds and len(thresholds) != 1 and len(thresholds) != n_models:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) does not match number of models ({n_models}). "
            "Provide either one threshold for all models or one threshold per model."
        )

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
    parser.add_argument('--gt_column', default='GT', help='Name of the ground truth column in test set (default: GT)')
    parser.add_argument('--score_column', default='score', help='Name of the score column in model files (default: score)')

    args = parser.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(',')]


    # Load the data once
    models_df, queries = load_models(
        args.test,
        args.models,
        args.queries,
        filter_query=args.filter,
        gt_column=args.gt_column,
        score_column=args.score_column
    )

    # Validate thresholds
    model_names = get_meta_columns_in_order(models_df)[1:]
    validate_thresholds(args.thresh, len(model_names))

    if args.thresh:
        thresh_values = args.thresh if len(args.thresh) > 1 else [args.thresh[0]] * len(model_names)
    else:
        thresh_values = [0.5] * len(model_names)

    if not args.gui:
        # Compute all metrics
        raw_results = compute_metrics(models_df, queries, thresholds=thresh_values, packages=("thresh", "raw", "plots"))
        results = raw_results_to_final_df(raw_results, model_names, metrics, queries.columns)
        
        if results is not None and not results.empty:
            if args.filter:
                print(f"Initial filter applied: '{args.filter}'")

            display_results(results, metrics, flatten=args.flatten)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n\nTotal evaluation time: {duration:.2f} seconds")

    else:
        if st is None:
            print("Error: Streamlit is required for GUI mode. Please install it with: pip install streamlit")
            return

        # Configure pandas display options to show all columns and fix decimal places
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', '{:.03f}'.format)

        st.title("Interactive Model Evaluator")

        threshold_inputs = []
        st.sidebar.header("Thresholds per model")
        for i, model_name in enumerate(model_names):
            value = st.sidebar.slider(
                f"Threshold for {model_name}",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(thresh_values[i]),
                help=f"Classification threshold for {model_name}. Samples with scores >= threshold are classified as positive."
            )
            threshold_inputs.append(value)

        # Compute all metrics including plots
        raw_results = compute_metrics(models_df, queries, thresholds=threshold_inputs, packages=("thresh", "raw", "plots"))
        results = raw_results_to_final_df(raw_results, model_names, metrics, queries.columns)

        query_labels = results.index.tolist()
        table_container, plot_container = st.columns([3, 5])

        # Place the dropdown and radio on the same row
        col1, col2 = st.columns([4, 2])
        with col1:
            selected_query_label = st.selectbox(
                "Select a query to visualize:",
                options=query_labels,
                index=0,  # pre-select the first query
                help="Select a query to view its metrics and performance curves"
            )
        with col2:
            view_toggle = st.radio(
                "Select view type:",
                options=["Query-indexed view", "Model-indexed view"],
                horizontal=True,
                help="Query-indexed: metrics as rows, models as columns. Model-indexed: models as rows, metrics as columns."
            )

        # Now plot, based on the *selected* query
        with plot_container:
            plot_roc_and_pr_curves_for_streamlit = setup_streamlit_display()
            plot_roc_and_pr_curves_for_streamlit(raw_results, query=selected_query_label)

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