#!/usr/bin/env python3
import argparse
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt


# Import and configure Streamlit first if available
try:
    import streamlit as st
    st.set_page_config(layout="wide")
    import pandas as pd
except ImportError:
    st = None

from data_loading import prepare_tables
from metrics import compute_metrics, raw_results_to_final_df
from display import display_results, transform_df_for_model_view, setup_streamlit_display, display_dataset_info, print_confusion_matrix, display_confusion_matrix
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
    parser.add_argument('--score_col', help='Name of the score column in model files (default: None, will infer from pos_classes)')
    parser.add_argument('--pos_classes', nargs='+', default=['1'], help='List of classes to consider as positive (default: ["1"])')
    parser.add_argument('--neg_classes', nargs='+', default=[], help='List of classes to consider as negative (default: [])')

    args = parser.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(',')]

    # Load the data once
    models_df, queries, info = prepare_tables(
        args.test,
        args.models,
        args.queries,
        filter_query=args.filter,
        gt_column=args.gt_column,
        score_column=args.score_col,
        pos_classes=args.pos_classes,
        neg_classes=args.neg_classes
    )
    model_names = list(info.model_paths.keys())

    # Validate thresholds
    validate_thresholds(args.thresh, len(model_names))

    if args.thresh:
        thresh_values = args.thresh if len(args.thresh) > 1 else [args.thresh[0]] * len(model_names)
    else:
        thresh_values = [0.5] * len(model_names)

    if not args.gui:
        # Compute all metrics
        raw_results = compute_metrics(
            models_df,
            info,
            queries_bool_df=queries,
            thresholds=thresh_values,
            packages=("thresh", "raw", "plots", "confusion","multiclass")
        )
        results = raw_results_to_final_df(raw_results, model_names, metrics, queries.columns)

        if results is not None and not results.empty:
            if args.filter:
                print(f"Initial filter applied: '{args.filter}'")

            display_results(results, metrics, flatten=args.flatten)

            # Display confusion matrices for each model
            print("\nConfusion Matrices:")
            for model_name in model_names:
                if 'confusion' in raw_results[model_name]:
                    print(f"\n{model_name} Confusion Matrix:")
                    # Get the first available query label
                    query_label = next(iter(raw_results[model_name]['confusion'].keys()))
                    conf_matrix = raw_results[model_name]['confusion'][query_label]
                    print_confusion_matrix(conf_matrix)

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

        # Multi-class selection UI if multiple classes are available
        if len(info.available_classes) > 1:
            st.sidebar.header("Class Selection")

            # First select positive classes
            selected_pos_classes = st.sidebar.multiselect(
                "Select classes to consider as positive:",
                options=info.available_classes,
                default=info.pos_classes,
                help="Scores for these classes will be summed to determine the positive class score"
            )

            # Then select negative classes from remaining classes
            remaining_classes = [c for c in info.available_classes if c not in selected_pos_classes]
            selected_neg_classes = st.sidebar.multiselect(
                "Select classes to consider as negative:",
                options=remaining_classes,
                default=info.neg_classes,
                help="These classes will be considered as negative class"
            )

            # Recalculate if class selection changed
            if (selected_pos_classes != info.pos_classes or
                selected_neg_classes != info.neg_classes):
                # Reload with new class selection
                models_df, queries, info = prepare_tables(
                    args.test,
                    args.models,
                    args.queries,
                    filter_query=args.filter,
                    gt_column=args.gt_column,
                    score_column=args.score_col,
                    pos_classes=selected_pos_classes,
                    neg_classes=selected_neg_classes
                )

        # Add collapsible info box - moved here to show updated info_dict
        display_dataset_info(info)

        st.sidebar.header("Thresholds per model")
        threshold_inputs = []

        for i, model_name in enumerate(model_names):
            # Initialize threshold values
            # Use get() to avoid the "widget created with default value but also set via Session State" warning
            if i < len(thresh_values):
                default_thresh = thresh_values[i] if isinstance(thresh_values[i], (int, float)) else thresh_values[i][1]
            else:
                default_thresh = 0.5

            # Only set session state if it doesn't already exist
            if f"threshold_{model_name}" not in st.session_state:
                st.session_state[f"threshold_{model_name}"] = float(default_thresh)

            # Use checkbox with key parameter to bind directly to session state
            use_dual_thresh = st.sidebar.checkbox(
                f"Use dual thresholds for {model_name}",
                key=f"use_dual_thresh_{model_name}",
                help=f"Enable to set separate thresholds for negative and positive classifications for {model_name}"
            )

            if use_dual_thresh:
                # For dual threshold, use two separate number inputs
                st.sidebar.text(f"Threshold range for {model_name}")

                # First get the current values
                current_high = st.session_state[f"threshold_{model_name}"]

                # Only set low threshold session state if it doesn't already exist
                if f"threshold_low_{model_name}" not in st.session_state:
                    st.session_state[f"threshold_low_{model_name}"] = max(0.0, current_high - 0.05)

                current_low = st.session_state[f"threshold_low_{model_name}"]
                current_low = max(0.0, min(current_low, current_high))
                current_low = max(0.0, min(current_low, current_high))

                col1, col2 = st.sidebar.columns(2)
                with col1:
                    neg_thresh = st.number_input(
                        "Min",
                        min_value=0.0,
                        max_value=current_high,
                        value=current_low,
                        step=0.05,
                        key=f"threshold_low_{model_name}",
                        help="Lower threshold. Samples with scores below this will be classified as negative."
                    )

                with col2:
                    pos_thresh = st.number_input(
                        "Max",
                        min_value=neg_thresh,
                        max_value=1.0,
                        value=current_high,
                        step=0.05,
                        key=f"threshold_{model_name}",
                        help="Upper threshold. Samples with scores above this will be classified as positive."
                    )
            else:
                # Use number_input instead of slider for better focus retention
                st.sidebar.text(f"Threshold for {model_name}")

                # Get current value from session state
                current_value = st.session_state[f"threshold_{model_name}"]

                # Use the current value directly without referencing session state in the widget
                pos_thresh = st.sidebar.number_input(
                    "Value",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_value,  # Use variable instead of direct session state reference
                    step=0.05,
                    key=f"threshold_{model_name}",
                    help=f"Classification threshold for {model_name}. Samples with scores above this value will be classified as positive.",
                    label_visibility="collapsed"  # Hide the label to avoid focus jumping to it
                )
                neg_thresh = pos_thresh

            # Add the current thresholds to the list
            threshold_inputs.append((neg_thresh, pos_thresh))

        # Compute all metrics including plots
        raw_results = compute_metrics(
            models_df,
            info,
            queries_bool_df=queries,
            thresholds=threshold_inputs,
            packages=("thresh", "raw", "plots", "confusion", "multiclass")
        )
        results = raw_results_to_final_df(raw_results, model_names, metrics, queries.columns)

        # Add query selection to sidebar after we have the results
        st.sidebar.header("Query Selection")
        selected_query_label = st.sidebar.selectbox(
            "Select a query to visualize:",
            options=results.index.tolist(),
            index=0,  # pre-select the first query
            help="Select a query to view its metrics and performance curves"
        )

        # Place the radio on the same row
        col1, col2 = st.columns([4, 3])
        with col1:
            if info.filter_query:
                st.subheader(f"**Filter:** {info.filter_query}")
        with col2:
            view_toggle = st.radio(
                "View",
                options=["Query-indexed view", "Model-indexed view"],
                horizontal=True,
                help="Query-indexed: metrics as rows, models as columns. Model-indexed: models as rows, metrics as columns.",
                label_visibility="hidden"
            )

        table_container, plot_container = st.columns([4, 5])

        # Define model colors from the same palette used for plots
        colors = plt.cm.tab10.colors
        model_colors = {model_name: colors[i % len(colors)] for i, model_name in enumerate(model_names)}

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

        # Display confusion matrices for each model
        st.subheader("Confusion Matrices")
        for i, model_name in enumerate(model_names):
            if 'confusion' in raw_results[model_name]:
                st.write(f"**{model_name}**")
                conf_matrix = raw_results[model_name]['confusion'][selected_query_label]
                display_confusion_matrix(conf_matrix, model_color=model_colors[model_name], pos_classes=info.pos_classes)

if __name__ == "__main__":
    main()