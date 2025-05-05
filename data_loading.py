import pandas as pd
from pathlib import Path
from common import GT_COL, SCORE_COL, cache_data, format_query_stats

@cache_data
def load_models(test_file, model_files, queries_file=None, filter_query=None, model_names=None, gt_column='GT', score_column='score'):
    if model_names is None:
        model_names = [Path(mf).stem for mf in model_files]
    model_files = dict(zip(model_names, model_files))
    test_df = pd.read_csv(test_file).set_index("id")

    # Create standardized ground truth column
    if gt_column not in test_df.columns:
        raise ValueError(f"Ground truth column '{gt_column}' not found in test file")
    test_df[GT_COL] = test_df[gt_column]

    if filter_query:
        original_size = len(test_df)
        test_df = test_df.query(filter_query)
        filtered_size = len(test_df)
        print(f"Applied filter: '{filter_query}'")
        print(f"Filtered test set from {original_size} to {filtered_size} samples ({filtered_size/original_size*100:.1f}%)")

    # Set index to 'id' and rename score column
    model_dfs = {}
    for model_name, model_file in model_files.items():
        model_df = pd.read_csv(model_file).set_index("id")
        if score_column not in model_df.columns:
            raise ValueError(f"Score column '{score_column}' not found in model file {model_file}")
        model_df = model_df.rename(columns={score_column: SCORE_COL})
        model_dfs[model_name] = model_df

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
        pos_count = subset_df[GT_COL].sum()  # Using standardized column name
        neg_count = len(subset_df) - pos_count
        query_stat = format_query_stats(query, len(subset_df), pos_count, total_samples)
        has_both_classes = pos_count > 0 and neg_count > 0
        if has_both_classes:
            valid_queries[query] = query_stat

    return models_df, valid_queries

def validate_data(test_df, model_dfs):
    pos_count = test_df[GT_COL].sum()
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