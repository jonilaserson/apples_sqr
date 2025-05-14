import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from common import GT_COL, SCORE_COL, cache_data, format_query_stats
from metrics import DatasetInfo

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
def load_testset_and_scores(
    test_file: str,
    model_files: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load test data and model predictions from CSV files.
    
    Args:
        test_file: Path to test set CSV file with ground truth
        model_files: Dictionary mapping model names to their file paths
        
    Returns:
            - DataFrame with test data
            - Dictionary mapping model names to their prediction DataFrames
    """
    test_df = pd.read_csv(test_file).set_index("id")
    
    model_dfs = {}
    for model_name, model_file in model_files.items():
        model_df = pd.read_csv(model_file).set_index("id")
        model_dfs[model_name] = model_df
        
    return test_df, model_dfs

@cache_data
def prepare_tables(
    test_file: str,
    model_files: List[str],
    queries_file: Optional[str] = None,
    filter_query: Optional[str] = None,
    model_names: Optional[List[str]] = None,
    gt_column: str = 'GT',
    score_column: Optional[str] = None,
    pos_classes: List[str] = ['1'],
    neg_classes: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, DatasetInfo]:
    """Prepare test data and model predictions for evaluation, optionally filtered by queries.
    
    Args:
        test_file: Path to test set CSV file with ground truth
        model_files: List of paths to model prediction CSV files
        queries_file: Optional path to file containing queries
        filter_query: Optional initial filter to apply to test set
        model_names: Optional list of names for models (defaults to filenames)
        gt_column: Name of ground truth column in test file
        score_column: Name of score column in model files. If None, will use pos_classes.
        pos_classes: List of classes to consider as positive (default: ["1"])
        neg_classes: Optional list of classes to consider as negative (default: None)
        
    Returns:
        Tuple of:
            - DataFrame with test data and model predictions
            - Boolean DataFrame of valid queries
            - DatasetInfo object containing information about the dataset
    """
    if model_names is None:
        model_names = [Path(mf).stem for mf in model_files]
    model_files_dict = dict(zip(model_names, model_files))
    # Load the raw data
    test_df, model_dfs = load_testset_and_scores(test_file, model_files_dict)
    total_samples = len(test_df)

    # Create standardized ground truth column
    if gt_column not in test_df.columns:
        raise ValueError(f"Ground truth column '{gt_column}' not found in test file")
    
    # Convert ground truth to binary (0/1)
    if pd.api.types.is_numeric_dtype(test_df[gt_column]):        # For numeric ground truth, check if values are in pos_classes
        try:
            pos_classes = [int(c) for c in pos_classes]
        except ValueError:
            raise ValueError(f"Ground truth column '{gt_column}' is numeric but pos_classes contains non-numeric values: {pos_classes}")
    test_df[GT_COL] = test_df[gt_column].isin(pos_classes).astype(int)

    # Filter test set to only include samples in pos_classes + neg_classes if neg_classes is non-empty
    if neg_classes is not None and len(neg_classes) > 0:
        # Validate that pos_classes and neg_classes don't intersect
        if set(pos_classes) & set(neg_classes):
            raise ValueError("pos_classes and neg_classes cannot have overlapping values")
        all_classes = pos_classes + neg_classes
        # Create a proper pandas query string for class filtering
        classes_query = f"{gt_column} in {all_classes}"
        if filter_query:
            filter_query = f"{filter_query} and ({classes_query})"
        else:
            filter_query = classes_query
        
    if filter_query:
        original_size = len(test_df)
        test_df = test_df.query(filter_query)
        filtered_size = len(test_df)
        filtered_samples = filtered_size
    else:
        filtered_samples = total_samples

    # Get available classes from the test set's ground truth column.  Put pos_classes first.
    available_classes = list(map(str, sorted(test_df[gt_column].unique().tolist())))
    #pos_classes = [c for c in available_classes if c in pos_classes]
    #available_classes = pos_classes + [c for c in available_classes if c not in pos_classes]

    # Handle score columns
    if score_column is None:
        # If score_column not specified, use all pos_classes
        score_columns = pos_classes
    else:
        score_columns = [score_column]

    # Handle score columns and create SCORE_COL
    for model_name, model_df in model_dfs.items():
        # Check if all columns exist
        missing_cols = [col for col in score_columns if col not in model_df.columns]
        if missing_cols:
            raise ValueError(f"Score columns {missing_cols} not found in model file {model_files_dict[model_name]}")
        
        # Sum the specified columns to create SCORE_COL
        model_df[SCORE_COL] = model_df[score_columns].sum(axis=1)

    validate_data(test_df, model_dfs)

    # Merge each model with test_df
    models_df = test_df
    for model_name, model_df in model_dfs.items():
        models_df = join_with_namespace(models_df, model_df, right_ns=model_name, left_ns="test")

    valid_queries = get_valid_queries(queries_file, test_df)
    
    # Create DatasetInfo object
    info = DatasetInfo(
        test_file=test_file,
        total_samples=total_samples,
        filtered_samples=filtered_samples,
        filter_query=filter_query,
        gt_column=gt_column,
        model_paths=model_files_dict,
        available_classes=available_classes,
        pos_classes=pos_classes,
        neg_classes=neg_classes,
        score_column=score_column
    )
        
    return models_df, valid_queries, info

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