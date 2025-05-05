# Model Evaluation Tool

A flexible command-line tool for evaluating and comparing binary classifiers. This tool supports multiple evaluation metrics, subset analysis through queries, and includes both CLI and interactive GUI modes.

## Features

- Multiple metric support (AUC-ROC, precision, recall, accuracy, F1, max F1)
- Compare multiple models side by side
- Filter and analyze model performance on data subsets using custom queries
- Interactive GUI mode with threshold tuning and visualization
- ROC and Precision-Recall curve plotting
- Formatted table output for easy comparison

## Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
```bash
uv pip install pandas numpy scikit-learn matplotlib tabulate streamlit
```

## Usage

### Basic Usage

```bash
python evaluate.py -t <test_file> -m <model1_file> <model2_file> [options]
```

### Required Arguments

- `-t, --test`: Path to test set CSV file with ground truth
- `-m, --models`: Paths to model prediction CSV files (one or more)

### Optional Arguments

- `-q, --queries`: Path to text file containing subset queries
- `-f, --filter`: Initial filter query to apply to test set
- `--metrics`: Comma-separated list of metrics (default: 'auc')
  - Available metrics: auc, precision, recall, accuracy, f1, max_f1
- `--thresh`: Threshold(s) for binary classification (default: 0.5)
- `--flatten`: Flatten the result table
- `--gui`: Launch interactive GUI for threshold tuning
- `--gt_column`: Name of ground truth column (default: 'GT')
- `--score_column`: Name of score column (default: 'score')

### Input File Formats

1. Test file (CSV):
   - Must have an 'id' column
   - Must have a ground truth column (default name: 'GT')

2. Model files (CSV):
   - Must have an 'id' column matching test file
   - Must have a score column (default name: 'score')

3. Queries file (TXT):
   - One query per line
   - Uses pandas query syntax
   - Example:
     ```
     feature1 > 0
     feature2 < 2
     category == 'A'
     ```

### Examples

1. Basic evaluation with default metrics:
```bash
python evaluate.py -t test_set.csv -m model1.csv model2.csv
```

2. Multiple metrics and custom thresholds:
```bash
python evaluate.py -t test_set.csv -m model1.csv model2.csv --metrics auc,precision,recall,f1 --thresh 0.7
```

3. Using queries file and initial filter:
```bash
python evaluate.py -t test_set.csv -m model1.csv model2.csv -q queries.txt -f "score > 0.1"
```

4. Launch interactive GUI:
```bash
python evaluate.py -t test_set.csv -m model1.csv model2.csv --gui
```

## GUI Mode Features

The interactive GUI mode (--gui) provides:
- Dynamic threshold adjustment
- Real-time metric updates
- ROC and Precision-Recall curve visualization
- Query selection and filtering
- Multiple view options for results

## Output Format

The tool provides:
- Metric values in formatted tables
- Sample distribution for each query
- Performance comparison across models
- Visual plots (in GUI mode)

Example output format:
```
Evaluation results for metric "auc":
|                                    |   model1 |   model2 |
|:-----------------------------------|---------:|---------:|
| [ 71: 29] 100.0%  all              |   0.9131 |   0.7028 |
| [ 33: 13]  46.0%  feature1 > 0     |   0.8485 |   0.7413 |
```

The `[neg:pos]` prefix shows the distribution of negative and positive samples for each query.