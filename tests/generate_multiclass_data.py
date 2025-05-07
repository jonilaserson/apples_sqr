#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import string
from typing import Union, List

def generate_multiclass_data(classes: Union[int, List[str]] = 5, samples_per_class: int = 50, 
                           score_col_prefix: str = None, seed: int = 42):
    """
    Generate synthetic multiclass classification data.
    
    Args:
        classes: Either an integer (number of classes) or a list of class names
        samples_per_class (int): Number of samples per class
        score_col_prefix (str): Optional prefix for score column names (e.g. 'score_')
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (test_set_df, model1_df, model2_df)
    """
    np.random.seed(seed)
    
    # Handle classes parameter
    if isinstance(classes, int):
        default_classes = [
            "cat", "dog", "horse", "zebra", "mouse",
            "elephant", "giraffe", "lion", "tiger", "bear",
            "penguin", "dolphin", "eagle", "shark", "whale",
            "rabbit", "fox", "wolf", "deer", "monkey"
        ]
        class_names = default_classes[:classes]
        n_classes = classes
    else:
        class_names = classes
        n_classes = len(classes)
    
    total_samples = n_classes * samples_per_class
    
    # Generate random categories (A, B, C)
    categories = list("ABC")
    random_categories = np.random.choice(categories, total_samples)
    
    # Generate features
    feature1 = np.random.normal(0, 1, total_samples)
    feature2 = np.random.normal(2, 1, total_samples)  # Similar to original data
    
    # Generate ground truth labels
    gt = np.random.randint(0, n_classes, total_samples)
    
    # Create test set dataframe
    test_set_data = {
        'id': range(1, total_samples + 1),
        'feature1': feature1,
        'feature2': feature2,
        'category': random_categories,
        'GT': [class_names[i] for i in gt],
        'GT_index': gt
    }
    test_set_df = pd.DataFrame(test_set_data)
    
    # Generate model scores (probabilities for each class)
    # Model1: Strong bias towards correct class (alpha=2.0 for correct class, 0.5 for others)
    # Model2: Moderate bias towards correct class (alpha=1.5 for correct class, 0.5 for others)
    model1_scores = np.zeros((total_samples, n_classes))
    model2_scores = np.zeros((total_samples, n_classes))
    
    for i in range(total_samples):
        # Create alpha parameters for Dirichlet distribution
        # Higher alpha for correct class makes it more likely to predict correctly
        model1_alpha = np.ones(n_classes) * 0.5  # Base alpha for all classes
        model2_alpha = np.ones(n_classes) * 0.5  # Base alpha for all classes
        
        # Increase alpha for the correct class
        model1_alpha[gt[i]] = 2.0  # Stronger bias for model1
        model2_alpha[gt[i]] = 1.5  # Weaker bias for model2
        
        # Generate scores using Dirichlet distribution
        model1_scores[i] = np.random.dirichlet(model1_alpha)
        model2_scores[i] = np.random.dirichlet(model2_alpha)
    
    # Create column names for model scores
    score_columns = [f"{score_col_prefix}{name}" if score_col_prefix else name for name in class_names]
    
    # Create model prediction dataframes
    model1_data = {
        'id': range(1, total_samples + 1),
        **{col: model1_scores[:, i] for i, col in enumerate(score_columns)}
    }
    model2_data = {
        'id': range(1, total_samples + 1),
        **{col: model2_scores[:, i] for i, col in enumerate(score_columns)}
    }
    
    model1_df = pd.DataFrame(model1_data)
    model2_df = pd.DataFrame(model2_data)
    
    return test_set_df, model1_df, model2_df

def main():
    parser = argparse.ArgumentParser(description='Generate multiclass test data')
    parser.add_argument('--classes', type=str, default='5', 
                      help='Number of classes (integer) or comma-separated list of class names')
    parser.add_argument('--samples_per_class', type=int, default=50, help='Samples per class')
    parser.add_argument('--output_dir', type=str, default='test_multiclass', help='Output directory')
    parser.add_argument('--score_col_prefix', type=str, default=None,
                      help='Prefix for score column names (e.g. "score_")')
    parser.add_argument('--score_columns', nargs='+', default=['score'],
                      help='Names of score columns to sum for positive class score (default: score)')
    args = parser.parse_args()
    
    # Parse classes argument
    if args.classes.isdigit():
        classes = int(args.classes)
    else:
        classes = [c.strip() for c in args.classes.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate data
    test_set_df, model1_df, model2_df = generate_multiclass_data(
        classes=classes,
        samples_per_class=args.samples_per_class,
        score_col_prefix=args.score_col_prefix
    )
    
    # Save data
    test_set_df.to_csv(output_dir / 'test_set.csv', index=False)
    model1_df.to_csv(output_dir / 'model1.csv', index=False)
    model2_df.to_csv(output_dir / 'model2.csv', index=False)
    
    # Generate example queries
    queries = [
        "feature1 > 0",
        "feature2 < 2",
        "category == 'A'",
        "feature1 > 0 and feature2 < 3",
        "category != 'C'",
        "GT == 'cat'",
        "GT_index == 0"
    ]
    
    with open(output_dir / 'queries.txt', 'w') as f:
        f.write('\n'.join(queries))
    
    print(f"Generated multiclass test data in {output_dir}/")
    print(f"Classes: {classes}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Total samples: {len(classes) * args.samples_per_class if isinstance(classes, list) else classes * args.samples_per_class}")
    print(f"Score columns for positive class: {args.score_columns}")

if __name__ == '__main__':
    main() 