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
    
from data_loading import prepare_tables
from metrics import compute_metrics, raw_results_to_final_df
from display import display_results, transform_df_for_model_view, setup_streamlit_display, display_dataset_info
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
    parser.add_argument('--pos_query', help='Query to define positive cases (samples matching this query will have GT=1)')
    parser.add_argument('--metrics', default='auc', help='Comma-separated list of metrics to compute (auc,precision,recall,accuracy,f1,max_f1)')
    parser.add_argument('--thresh', type=float, nargs='+', help='Threshold(s) for binary classification metrics. One value or one per model.')
    parser.add_argument('--flatten', action='store_true', help='Flatten the result table instead of showing separate tables per metric')
    parser.add_argument('--gui', action='store_true', help='Launch interactive GUI for threshold tuning')
    parser.add_argument('--gt_column', default='GT', help='Name of the ground truth column in test set (default: GT)')
    parser.add_argument('--score_column', default='score', help='Name of the score column in model files (default: score)')
    parser.add_argument('--score_columns', nargs='+', help='For multi-class: names of score columns in model files to sum for positive class score')

    args = parser.parse_args()

    metrics = [m.strip().lower() for m in args.metrics.split(',')]

    # Determine which score column parameter to use
    score_col_param = args.score_columns if args.score_columns else args.score_column

    # Load the data once
    models_df, queries, info_dict = prepare_tables(
        args.test,
        args.models,
        args.queries,
        filter_query=args.filter,
        gt_column=args.gt_column,
        score_column=score_col_param,
        pos_query=args.pos_query
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

        # Add collapsible info box
        display_dataset_info(info_dict)

        # Get available columns from the first model file for multi-class selection
        try:
            first_model_path = info_dict["model_paths"][model_names[0]]
            first_model_df = pd.read_csv(first_model_path)
            # Exclude id and any columns starting with underscore
            model_class_cols = [col for col in first_model_df.columns 
                              if col != 'id' and not col.startswith('_')]
        except Exception as e:
            st.warning(f"Could not read model columns for multi-class detection: {e}")
            model_class_cols = []
        
        # Multi-class selection UI if multiple score columns are available
        selected_score_columns = None
        if len(model_class_cols) > 1:
            st.sidebar.header("Class Selection")
            multi_class_mode = st.sidebar.checkbox(
                "Multi-class Mode", 
                value=isinstance(score_col_param, list),
                help="Enable to select which classes to consider as positive"
            )
            
            if multi_class_mode:
                selected_score_columns = st.sidebar.multiselect(
                    "Select classes to consider as positive:",
                    options=model_class_cols,
                    default=info_dict.get("score_columns", []),
                    help="Scores for these classes will be summed to determine the positive class score"
                )
                
                # Recalculate if multi-class selection changed
                if selected_score_columns and selected_score_columns != info_dict.get("score_columns", []):
                    # Reload with new score columns selection
                    models_df, queries, info_dict = prepare_tables(
                        args.test,
                        args.models,
                        args.queries,
                        filter_query=args.filter,
                        gt_column=args.gt_column,
                        score_column=selected_score_columns,
                        pos_query=args.pos_query
                    )

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
        table_container, plot_container = st.columns([4, 5])

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