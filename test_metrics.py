import pandas as pd
import numpy as np
from metrics import compute_metrics, apply_thresholds_and_evaluate

def main():
    # Create test data
    # Multi-index DataFrame with 'test' and model columns
    index = pd.MultiIndex.from_tuples([
        ('test', '_GT_'), 
        ('test', '_score_'), 
        ('model1', '_score_'), 
        ('model2', '_score_')
    ], names=['namespace', 'column'])
    
    data = {
        0: [1, 0.9, 0.8, 0.6],  # true positive
        1: [0, 0.3, 0.4, 0.2],  # true negative
        2: [1, 0.8, 0.7, 0.9],  # true positive
        3: [0, 0.2, 0.3, 0.1],  # true negative
        4: [1, 0.9, 0.8, 0.7]   # true positive
    }
    
    df = pd.DataFrame(data, index=index).T
    
    # Test queries - use a valid pandas query syntax
    queries = {
        'all': 'All samples', 
        '_GT_ == 1': 'Ground truth is True'
    }
    
    # Test with threshold metrics
    print("\n=== Testing threshold-based metrics ===")
    metrics = ['thresh@precision', 'thresh@recall', 'thresh@f1', 'thresh@tp', 'thresh@fp']
    thresholds = [0.5, 0.5]
    
    thresh_results = compute_metrics(df, list(queries.keys()), metrics, thresholds)
    print("\nThreshold Metrics Raw Results:")
    for metric, models in thresh_results.items():
        print(f"\n{metric}:")
        for model, queries_data in models.items():
            print(f"  {model}: {queries_data}")
    
    # Test with raw metrics
    print("\n=== Testing raw metrics ===")
    raw_metrics = ['raw@auc', 'raw@max_f1']
    
    raw_results = compute_metrics(df, list(queries.keys()), raw_metrics, None)
    print("\nRaw Metrics Results:")
    for metric, models in raw_results.items():
        print(f"\n{metric}:")
        for model, queries_data in models.items():
            print(f"  {model}: {queries_data}")
    
    # Test formatted output
    print("\n=== Testing formatted output ===")
    all_metrics = ['thresh@precision', 'thresh@recall', 'raw@auc']
    final_df = apply_thresholds_and_evaluate(df, queries, all_metrics, thresholds)
    print("\nFormatted Results:")
    print(final_df)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 