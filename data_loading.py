import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from common import GT_COL, SCORE_COL, cache_data, format_query_stats

def get_query_features_df(df_samples: pd.DataFrame, queries: List[str]) -> pd.DataFrame:
    """Apply queries on df_samples and return a boolean feature dataframe.
    
    Args:
        df_samples: DataFrame containing the data to evaluate queries on
        queries: List of query strings to evaluate
        
    Returns:
        DataFrame where:
            - Index matches df_samples
            - Columns are query strings
            - Values are boolean indicating if sample matches query
    """
    df_bool = df_samples[[]].copy()
    for query in queries:
        try:
            if query == 'all':
                df_bool[query] = pd.Series(True, index=df_samples.index)
            else:
                df_bool[query] = df_samples.eval(query)
        except Exception as e:
            print(query, e)
            raise        
    return df_bool

def get_valid_queries(queries_file: Optional[str], test_df: pd.DataFrame) -> pd.DataFrame:
    """Load queries from file and return a boolean DataFrame indicating which queries are valid for each sample.
    
    Args:
        queries_file: Path to file containing queries, one per line
        test_df: DataFrame containing the test data with GT_COL column
        
    Returns:
        DataFrame where:
            - Index matches test_df
            - Columns are query statistics strings
            - Values are boolean indicating if sample matches query
    """
    queries = ['all']
    if queries_file is not None:
        with open(queries_file, 'r') as f:
            queries.extend([q.strip() for q in f.readlines() if q.strip()])

    # First get boolean features for all queries
    query_bools = get_query_features_df(test_df, queries)
    
    # Now validate which queries have both classes
    valid_queries = {}
    total_samples = len(test_df)
    for query in queries:
        # Get samples matching this query
        matching_samples = query_bools[query]
        subset_df = test_df[matching_samples]
        pos_count = subset_df[GT_COL].sum()
        neg_count = len(subset_df) - pos_count
        query_stat = format_query_stats(query, len(subset_df), pos_count, total_samples)
        has_both_classes = pos_count > 0 and neg_count > 0
        if has_both_classes:
            valid_queries[query] = query_stat

    # Create final boolean DataFrame with only valid queries
    valid_query_bools = query_bools[list(valid_queries.keys())]
    # Rename columns to use query statistics strings
    valid_query_bools.columns = list(valid_queries.values())
            
    return valid_query_bools

@cache_data
def load_models(
    test_file: str,
    model_files: List[str],
    queries_file: Optional[str] = None,
    filter_query: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    gt_column: str = 'GT',
    score_column: str = 'score',
    pos_query: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load test data and model predictions, optionally filtered by queries.
    
    Args:
        test_file: Path to test set CSV file with ground truth
        model_files: List of paths to model prediction CSV files
        queries_file: Optional path to file containing queries
        filter_query: Optional initial filter to apply to test set
        model_names: Optional list of names for models (defaults to filenames)
        gt_column: Name of ground truth column in test file
        score_column: Name of score column in model files
        pos_query: Optional query to define positive cases (samples matching this query will have GT=1)
        
    Returns:
        Tuple of:
            - DataFrame with test data and model predictions
            - Boolean DataFrame of valid queries
            - Dictionary containing information about the dataset
    """
    if model_names is None:
        model_names = [Path(mf).stem for mf in model_files]
    model_files = dict(zip(model_names, model_files))
    test_df = pd.read_csv(test_file).set_index("id")
    total_samples = len(test_df)

    # Create standardized ground truth column
    if gt_column not in test_df.columns:
        raise ValueError(f"Ground truth column '{gt_column}' not found in test file")
    
    # If pos_query is provided, use it to define ground truth
    if pos_query is not None:
        test_df[GT_COL] = test_df.eval(pos_query).astype(int)
    else:
        test_df[GT_COL] = test_df[gt_column]

    if filter_query:
        original_size = len(test_df)
        test_df = test_df.query(filter_query)
        filtered_size = len(test_df)
        print(f"Applied filter: '{filter_query}'")
        print(f"Filtered test set from {original_size} to {filtered_size} samples ({filtered_size/original_size*100:.1f}%)")
        filtered_samples = filtered_size
    else:
        filtered_samples = total_samples

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

    valid_queries = get_valid_queries(queries_file, test_df)
    info_dict = {
        "test_file": test_file,
        "total_samples": total_samples,
        "filtered_samples": filtered_samples,
        "filter_query": filter_query,
        "pos_query": pos_query,
        "gt_column": gt_column,
        "model_paths": {name: path for name, path in model_files.items()}
    }
    return models_df, valid_queries, info_dict

def validate_data(test_df: pd.DataFrame, model_dfs: Dict[str, pd.DataFrame]) -> None:
    """Validate test and model data compatibility.
    
    Args:
        test_df: DataFrame containing test data
        model_dfs: Dictionary mapping model names to their prediction DataFrames
        
    Raises:
        ValueError: If data validation fails
    """
    pos_count = test_df[GT_COL].sum()
    neg_count = len(test_df) - pos_count

    if pos_count == 0 or neg_count == 0:
        raise ValueError("Error: Test set must contain both positive and negative samples after filtering")

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
    left_ns: Optional[str] = None
) -> pd.DataFrame:
    """Merge two DataFrames by index, adding a namespace to df_right columns.
    
    Args:
        df_left: Left DataFrame to join
        df_right: Right DataFrame to join
        right_ns: Namespace for right DataFrame columns
        left_ns: Optional namespace for left DataFrame columns
        
    Returns:
        DataFrame with hierarchical columns
    """
    df_right_named = df_right.copy()
    df_right_named.columns = pd.MultiIndex.from_product([[right_ns], df_right_named.columns])

    # Make sure df_left columns are MultiIndex too
    df_left_named = df_left.copy()
    if not isinstance(df_left_named.columns, pd.MultiIndex):
        df_left_named.columns = pd.MultiIndex.from_product([[left_ns], df_left_named.columns])

    return df_left_named.join(df_right_named) 