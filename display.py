import pandas as pd
from common import GT_COL, SCORE_COL
from plots import plot_curves

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
    
    return plot_roc_and_pr_curves_for_streamlit 