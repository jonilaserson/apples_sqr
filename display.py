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
    
    # Calculate intensities
    intensities = data / max_val
    
    # Create diagonal mask and adjust opacities
    diag_mask = pd.DataFrame(np.eye(len(data.index), len(data.columns), dtype=bool), 
                           index=data.index, columns=data.columns)
    
    # Calculate opacities while preserving DataFrame structure
    opacities = intensities.copy()
    opacities[~diag_mask] = opacities[~diag_mask] * 0.5
    
    # Format opacities to 2 decimal places and create style strings
    styles = pd.DataFrame('background-color: white', index=data.index, columns=data.columns)
    styles = styles.mask(data != 0, opacities.map(lambda x: f'background-color: rgba({r}, {g}, {b}, {x:.2f})'))
    
    return styles

def confusion_matrix_to_html(matrix, block_size, model_color, stats_cols=None, stats_rows=None):
    """Display confusion matrix with colored cells and stats.
    
    Args:
        matrix: DataFrame containing the confusion matrix values
        block_size: Number of positive classes (for styling)
        model_color: Color to use for the matrix cells
        stats_cols: DataFrame of statistics to display as columns (e.g., recall, specificity)
        stats_rows: DataFrame of statistics to display as rows (e.g., precision)
    """
    stats_cols = stats_cols if stats_cols is not None else pd.DataFrame()
    stats_rows = stats_rows if stats_rows is not None else pd.DataFrame()
    r, g, b = get_rgb(model_color)
    
    # Define CSS styles without trailing semicolons
    BG_COLOR = f'background-color: rgba({r}, {g}, {b}, 0.2)'
    BORDER_BOTTOM = 'border-bottom: 4px solid #222'
    BORDER_RIGHT = 'border-right: 4px solid #222'
    BORDER2 = 'border: 2px solid #444'
    BORDER1 = 'border: 1px solid #ddd'
    HEADER_BORDER_BOTTOM = 'border-bottom: 2px solid #222'
    INDEX_BORDER_RIGHT = 'border-right: 2px solid #222'
    TOTAL_BG = 'background-color: #f0f0f0'
    STAT_BG = 'background-color: #e0e0e0'
    GRAND_TOTAL_BG = 'background-color: #cccccc'
    FONT_BOLD = 'font-weight: bold'
    PADDING = 'padding: 8px'
    SPACER_COL = 'width: 10px; border: none'
    SPACER_ROW = 'height: 10px; border: none'
    
    # Helper function to properly join CSS styles with semicolons
    def join_styles(*styles):
        return '; '.join(filter(None, styles)) + ';'

    def fmt_stat(val):
        return f"{val:.1%}" if isinstance(val, (float, np.floating, np.float64)) and not pd.isna(val) else ''

    cell_styles = color_scale(matrix, r, g, b)
    last_pos_col = last_pos_row = block_size - 1

    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    grand_total = row_totals.sum()

    html = [f'<table style="border-collapse: collapse; width: 100%;">']
    
    # Header row
    html.append('<tr>')
    html.append(f'<th style="{join_styles(BORDER1, HEADER_BORDER_BOTTOM, INDEX_BORDER_RIGHT, PADDING)}"></th>')
    for i, col in enumerate(matrix.columns):
        # Apply background color based on position
        col_bg = BG_COLOR if i <= last_pos_col else 'background-color: white'
        # Apply right border at division between positive/negative classes
        border_right = BORDER_RIGHT if i == last_pos_col else ''
        html.append(f'<th style="{join_styles(BORDER1, border_right, HEADER_BORDER_BOTTOM, PADDING, "text-align: center", col_bg)}">{col}</th>')
    
    # Add spacer column header
    html.append(f'<th style="{join_styles(SPACER_COL)}"></th>')
    
    html.append(f'<th style="{join_styles(BORDER2, HEADER_BORDER_BOTTOM, TOTAL_BG, PADDING)}">Total</th>')
    for stat in stats_cols.columns:
        html.append(f'<th style="{join_styles(BORDER2, HEADER_BORDER_BOTTOM, STAT_BG, PADDING)}">{stat.capitalize()}</th>')
    html.append('</tr>')

    # Data rows
    for row_i, idx in enumerate(matrix.index):
        html.append('<tr>')
        # Apply background color based on position
        bg_color = BG_COLOR if row_i <= last_pos_row else 'background-color: white'
        # Apply bottom border at division between positive/negative classes
        border_bottom = BORDER_BOTTOM if row_i == last_pos_row else ''
        html.append(f'<th style="{join_styles(BORDER1, border_bottom, INDEX_BORDER_RIGHT, PADDING, bg_color)}">{idx}</th>')
        
        for col_i, col in enumerate(matrix.columns):
            val = matrix.loc[idx, col]
            cell_style = cell_styles.loc[idx, col]
            # Apply right border at division between positive/negative classes
            border_right = BORDER_RIGHT if col_i == last_pos_col else ''
            # Apply bottom border at division between positive/negative classes
            border_bottom = BORDER_BOTTOM if row_i == last_pos_row else ''
            html.append(f'<td style="{join_styles(BORDER1, border_right, border_bottom, PADDING, cell_style)}">{val}</td>')
        
        # Add spacer column
        html.append(f'<td style="{join_styles(SPACER_COL)}"></td>')
        
        # Row total cell
        row_total = row_totals[idx]
        row_pct = f"{(row_total / grand_total * 100):.1f}%"
        html.append(f'<td style="{join_styles(BORDER2, TOTAL_BG, FONT_BOLD, PADDING)}">{row_total} <span style=\'color:#666;font-size:smaller\'>({row_pct})</span></td>')
        
        # Stats columns
        for stat in stats_cols.columns:
            stat_val = stats_cols.loc[idx, stat] if idx in stats_cols.index else ''
            stat_val_str = fmt_stat(stat_val)
            html.append(f'<td style="{join_styles(BORDER2, STAT_BG, PADDING)}">{stat_val_str}</td>')
        html.append('</tr>')
    
    # Add spacer row
    html.append('<tr>')
    total_cols = 1 + len(matrix.columns) + 1 + 1 + len(stats_cols.columns)  # Index + matrix columns + spacer + total + stat columns
    html += [f'<td style="{join_styles(SPACER_ROW)}"></td>'] * total_cols
    html.append('</tr>')

    # Totals row
    html.append('<tr>')
    html.append(f'<th style="{join_styles(BORDER2, INDEX_BORDER_RIGHT, TOTAL_BG, FONT_BOLD, PADDING)}">Total</th>')
    for col in matrix.columns:
        col_total = col_totals[col]
        col_pct = f"{(col_total / grand_total * 100):.1f}%"
        html.append(f'<td style="{join_styles(BORDER2, TOTAL_BG, FONT_BOLD, PADDING)}">{col_total} <span style=\'color:#666;font-size:smaller\'>({col_pct})</span></td>')
    
    # Add spacer cell in totals row
    html.append(f'<td style="{join_styles(SPACER_COL)}"></td>')
    
    html.append(f'<td style="{join_styles(BORDER2, GRAND_TOTAL_BG, FONT_BOLD, PADDING)}">TOTAL = {grand_total}</td>')
    html += ['<td></td>'] * len(stats_cols.columns)
    html.append('</tr>')

    # Stats rows
    for stat in stats_rows.columns:
        html.append('<tr>')
        html.append(f'<th style="{join_styles(BORDER2, INDEX_BORDER_RIGHT, STAT_BG, FONT_BOLD, PADDING)}">{stat.capitalize()}</th>')
        for col in matrix.columns:
            val = stats_rows.loc[col, stat] if col in stats_rows.index else ''
            val_str = fmt_stat(val)
            html.append(f'<td style="{join_styles(BORDER2, STAT_BG, FONT_BOLD, PADDING)}">{val_str}</td>')
        
        # Add spacer cell in stats rows
        html.append(f'<td style="{join_styles(SPACER_COL)}"></td>')
        
        html.append('<td></td>' * (1 + len(stats_cols.columns)))
        html.append('</tr>')
    html.append('</table>')
    return ''.join(html)

def display_confusion_matrix(conf_matrix: pd.DataFrame, model_color=None, pos_classes=None):
    """Display a confusion matrix with colored cells and stats side by side."""
    import streamlit as st
    if pos_classes is None:
        pos_classes = []
    matrix = conf_matrix["predictions"]
    stats = conf_matrix["stats"]
    all_classes = [c for c in matrix.columns if c != 'dont_know']
    neg_classes = [c for c in all_classes if c not in pos_classes]
    ordered_columns = pos_classes + neg_classes
    if 'dont_know' in matrix.columns:
        ordered_columns.append('dont_know')
    matrix = matrix.reindex(index=pos_classes + neg_classes, columns=ordered_columns)
    
    # Pre-select the stats we want to display
    stats_cols = stats[['sensitivity', 'specificity']]
    stats_rows = stats[['precision']]
    
    html = confusion_matrix_to_html(matrix, len(pos_classes), model_color, stats_cols=stats_cols, stats_rows=stats_rows)
    st.markdown(html, unsafe_allow_html=True)

def print_confusion_matrix(conf_matrix: pd.DataFrame):
    """Print confusion matrix in text mode with flattened column names."""
    print(flatten_multiindex_columns(conf_matrix).to_markdown()) 