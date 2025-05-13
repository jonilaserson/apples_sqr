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

def get_rgb(color):
    if color is None:
        return (0, 128, 0)  # Default green
    if isinstance(color, (tuple, list, np.ndarray)):
        return [int(c * 255) for c in color[:3]]
    if isinstance(color, str):
        return [int(c * 255) for c in mcolors.to_rgb(color)]
    return (0, 128, 0)  # Fallback to green

def color_scale(data, r, g, b):
    max_val = data.max().max()
    if max_val == 0:
        return pd.DataFrame('background-color: white', index=data.index, columns=data.columns)
    cm = pd.DataFrame('', index=data.index, columns=data.columns)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            if val == 0:
                cm.iloc[i, j] = 'background-color: white'
                continue
            intensity = val / max_val
            opacity = intensity if i == j else intensity * 0.5
            cm.iloc[i, j] = f'background-color: rgba({r}, {g}, {b}, {opacity:.2f})'
    return cm

def confusion_matrix_to_html(matrix, block_size, model_color):
    r, g, b = get_rgb(model_color)
    cell_styles = color_scale(matrix, r, g, b)
    last_pos_col = last_pos_row = block_size - 1
    html = ['<table style="border-collapse: collapse; width: 100%;">']
    html.append('<tr>')
    html.append('<th style="border: 1px solid #ddd; padding: 8px;"></th>')
    for i, col in enumerate(matrix.columns):
        col_bg = f'background-color: rgba({r}, {g}, {b}, 0.2);' if i <= last_pos_col else ''
        border_right = 'border-right: 4px solid #222;' if i == last_pos_col else ''
        html.append(f'<th style="border: 1px solid #ddd; {border_right} padding: 8px; text-align: center; {col_bg}">{col}</th>')
    html.append('</tr>')
    for row_i, idx in enumerate(matrix.index):
        html.append('<tr>')
        bg_color = f'background-color: rgba({r}, {g}, {b}, 0.2)' if row_i <= last_pos_row else 'white'
        border_bottom = 'border-bottom: 4px solid #222;' if row_i == last_pos_row else ''
        html.append(f'<th style="border: 1px solid #ddd; {border_bottom} padding: 8px; background-color: {bg_color};">{idx}</th>')
        for col_i, col in enumerate(matrix.columns):
            val = matrix.loc[idx, col]
            cell_style = cell_styles.loc[idx, col]
            border_right = 'border-right: 4px solid #222;' if col_i == last_pos_col else ''
            border_bottom = 'border-bottom: 4px solid #222;' if row_i == last_pos_row else ''
            html.append(f'<td style="border: 1px solid #ddd; {border_right}{border_bottom} padding: 8px; {cell_style}">{val}</td>')
        html.append('</tr>')
    html.append('</table>')
    return ''.join(html)

def display_confusion_matrix(conf_matrix: pd.DataFrame, model_color=None, pos_classes=None):
    """Display a confusion matrix with colored cells and stats side by side.
    
    Args:
        conf_matrix: DataFrame containing confusion matrix with MultiIndex columns ['predictions', 'stats']
        model_color: Optional color to use for highlighting cells (matches plot color)
        pos_classes: List of positive class labels to show first
    """
    import streamlit as st
    # Ensure pos_classes is a list
    if pos_classes is None:
        pos_classes = []
    matrix = conf_matrix["predictions"]
    stats = conf_matrix["stats"]
    # Reorder rows and columns to show positive classes first
    all_classes = [c for c in matrix.columns if c != 'dont_know']
    neg_classes = [c for c in all_classes if c not in pos_classes]
    ordered_columns = pos_classes + neg_classes
    if 'dont_know' in matrix.columns:
        ordered_columns.append('dont_know')
    matrix = matrix.reindex(index=pos_classes + neg_classes, columns=ordered_columns)
    stats = stats.reindex(index=pos_classes + neg_classes)
    # Create two columns for side-by-side display
    col1, col2 = st.columns([5, 5])
    with col1:
        html = confusion_matrix_to_html(matrix, len(pos_classes), model_color)
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        st.dataframe(stats, use_container_width=True)

def print_confusion_matrix(conf_matrix: pd.DataFrame):
    """Print confusion matrix in text mode with flattened column names."""
    print(flatten_multiindex_columns(conf_matrix).to_markdown()) 