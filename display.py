import pandas as pd
from plots import plot_curves

def display_dataset_info(info_dict):
    """Display dataset information in a collapsible Streamlit expander.
    
    Args:
        info_dict: Dictionary containing all dataset information
    """
    import streamlit as st
    
    with st.expander("Dataset Information", expanded=False):
        info_text = f"**Test Set:** {info_dict['test_file']} ({info_dict['total_samples']:,} samples)<br>"
        
        if info_dict['filter_query']:
            info_text += f"**Filter:** `{info_dict['filter_query']}` ({info_dict['filtered_samples']:,} samples after filtering)<br>"
        
        if info_dict['pos_query']:
            info_text += f"**Positive Cases:** `{info_dict['pos_query']}`<br>"
        else:
            info_text += f"**Ground Truth Column:** `{info_dict['gt_column']}`<br>"
        
        # Show score columns if in multi-class mode
        if 'score_columns' in info_dict and len(info_dict['score_columns']) > 1:
            info_text += f"**Score Columns (Positive Class):** {', '.join(f'`{col}`' for col in info_dict['score_columns'])}<br>"
        
        info_text += "**Models:**<br>"
        for name, path in info_dict['model_paths'].items():
            info_text += f"- {name}: `{path}`<br>"
        
        st.markdown(info_text, unsafe_allow_html=True)

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

def setup_streamlit_display():
    """Import and setup Streamlit components for GUI mode."""
    import streamlit as st
    import matplotlib.pyplot as plt
    
    def plot_roc_and_pr_curves_for_streamlit(raw_results, query='all'):
        # Get the actual query string from the label (extracting from formatted label)
        colors = plt.cm.tab10.colors

        #st.markdown(f"### Plots")

        col1, col2 = st.columns(2)

        # --- ROC Plot ---
        with col1:
            fig1, ax1 = plot_curves(raw_results, colors, curve_type='roc', query_string=query)
            st.pyplot(fig1)

        # --- PR Plot ---
        with col2:
            fig2, ax2 = plot_curves(raw_results, colors, curve_type='pr', query_string=query)
            st.pyplot(fig2)
    
    return plot_roc_and_pr_curves_for_streamlit 