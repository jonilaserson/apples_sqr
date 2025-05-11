import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from plots import plot_curves
from metrics import DatasetInfo

def display_dataset_info(info: DatasetInfo):
    """Display dataset information in a collapsible Streamlit expander.
    
    Args:
        info: DatasetInfo object containing all dataset information
    """
    import streamlit as st
    
    with st.expander("Dataset Information", expanded=False):
        info_text = f"**Test Set:** {info.test_file} ({info.total_samples:,} samples)<br>"
        
        if info.filter_query:
            info_text += f"**Filter:** `{info.filter_query}` ({info.filtered_samples:,} samples after filtering)<br>"
        
        info_text += f"**Ground Truth Column:** `{info.gt_column}`<br>"
        
        # Show available classes
        if info.available_classes:
            info_text += f"**Available Classes:** {', '.join(f'`{col}`' for col in info.available_classes)}<br>"
        
        # Show positive classes
        if info.pos_classes:
            info_text += f"**Positive Classes:** {', '.join(f'`{col}`' for col in info.pos_classes)}<br>"
        
        # Show score column only if explicitly provided
        if info.score_column:
            info_text += f"**Score Column:** `{info.score_column}`<br>"
        
        info_text += "**Models:**<br>"
        for name, path in info.model_paths.items():
            info_text += f"- {name}: `{path}`<br>"
        
        st.markdown(info_text, unsafe_allow_html=True)

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a DataFrame with MultiIndex columns to show only the second level.
    
    Args:
        df: DataFrame with MultiIndex columns
        
    Returns:
        DataFrame with flattened column names
    """
    flat_df = df.copy()
    flat_df.columns = flat_df.columns.get_level_values(1)
    return flat_df

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

def display_confusion_matrix(conf_matrix: pd.DataFrame, model_color=None):
    """Display a confusion matrix with colored cells and stats side by side.
    
    Args:
        conf_matrix: DataFrame containing confusion matrix with MultiIndex columns ['predictions', 'stats']
        model_color: Optional color to use for highlighting cells (matches plot color)
    """
    import streamlit as st
    import matplotlib.colors as mcolors
    
    # Split into confusion matrix and stats
    matrix = conf_matrix["predictions"]
    stats = conf_matrix["stats"]
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns([5, 5])
    
    # Helper function to convert any color format to RGB values
    def get_rgb(color):
        if color is None:
            return (0, 128, 0)  # Default green
        if isinstance(color, (tuple, list, np.ndarray)):
            return [int(c * 255) for c in color[:3]]
        if isinstance(color, str):
            return [int(c * 255) for c in mcolors.to_rgb(color)]
        return (0, 128, 0)  # Fallback to green
    
    # Get RGB values from model color
    r, g, b = get_rgb(model_color)
    
    # Create styling function
    def color_scale(data):
        # Get max value for normalization
        max_val = data.max().max()
        if max_val == 0:
            return pd.DataFrame('background-color: white', index=data.index, columns=data.columns)
        
        # Initialize style DataFrame
        cm = pd.DataFrame('', index=data.index, columns=data.columns)
        
        # Apply styling to each cell
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if val == 0:
                    cm.iloc[i, j] = 'background-color: white'
                    continue
                
                # Use model color with intensity proportional to value
                intensity = val / max_val
                # Diagonal cells get full intensity, others get half
                opacity = intensity if i == j else intensity * 0.5
                cm.iloc[i, j] = f'background-color: rgba({r}, {g}, {b}, {opacity:.2f})'
        
        return cm
    
    # Apply styling and display
    with col1:
        styled_matrix = matrix.style.apply(color_scale, axis=None)
        st.dataframe(styled_matrix, use_container_width=True)
    
    with col2:
        st.dataframe(stats, use_container_width=True)

def print_confusion_matrix(conf_matrix: pd.DataFrame):
    """Print confusion matrix in text mode with flattened column names."""
    print(flatten_multiindex_columns(conf_matrix).to_markdown()) 