# Model Evaluation Tool

A tool for evaluating binary classifiers on test data, with support for multi-class scenarios.

## Features

- Evaluate binary classifiers on test data
- Support for multi-class scenarios
- Interactive GUI for threshold tuning
- Various metrics: AUC, precision, recall, accuracy, F1, max F1
- Query-based evaluation
- Support for filtering test data
- Interactive confusion matrix visualization with color-coded cells
- Side-by-side display of confusion matrix and statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python evaluate.py -t test_set.csv -m model1.csv model2.csv
```

### Command Line Arguments

- `-t, --test`: Path to test set CSV file with ground truth (required)
- `-m, --models`: Paths to model prediction CSV files (required, can specify multiple)
- `-q, --queries`: Path to queries text file
- `-f, --filter`: Initial filter query to apply to test set
- `--metrics`: Comma-separated list of metrics to compute (default: 'auc')
  - Available metrics: auc, precision, recall, accuracy, f1, max_f1
- `--thresh`: Threshold(s) for binary classification metrics. One value or one per model.
- `--flatten`: Flatten the result table instead of showing separate tables per metric
- `--gui`: Launch interactive GUI for threshold tuning
- `--gt_column`: Name of the ground truth column in test set (default: 'GT')
- `--score_col`: Name of the score column in model files (default: None, will infer from pos_classes)
- `--pos_classes`: List of classes to consider as positive (default: ['1'])

### Multi-class Support

The tool supports multi-class scenarios in two ways:

1. **Explicit Score Column**: Use `--score_col` to specify a single score column to use.

2. **Positive Classes**: Use `--pos_classes` to specify which classes should be considered positive. If `--score_col` is not specified, the tool will automatically sum the scores of all specified positive classes.

Example:
```bash
# Use a specific score column
python evaluate.py -t test_set.csv -m model1.csv --score_col class1

# Use multiple positive classes (scores will be summed)
python evaluate.py -t test_set.csv -m model1.csv --pos_classes class1 class2 class3
```

### Interactive GUI

Launch the interactive GUI with the `--gui` flag:

```bash
python evaluate.py -t test_set.csv -m model1.csv --gui
```

The GUI provides:
- Interactive threshold tuning in the sidebar
- Query selection in the sidebar
- ROC and PR curves
- Multiple view options for results
- Class selection for multi-class scenarios
- Dataset information display
- Interactive confusion matrix visualization:
  - Color-coded cells (green for correct predictions, red for incorrect)
  - Intensity indicates magnitude
  - Side-by-side display with statistics
  - Updates with selected query

### Input File Formats

#### Test Set CSV
- Must contain an 'id' column
- Must contain the ground truth column (default: 'GT')
- Ground truth can be numeric or categorical

#### Model Prediction CSV
- Must contain an 'id' column
- Must contain score column(s) for each class
- For multi-class, each class should have its own score column

### Output

The tool provides:
- Performance metrics for each model
- ROC and PR curves
- Query-based evaluation results
- Interactive visualization in GUI mode
- Confusion matrix visualization:
  - Text mode: Clean tabular format
  - GUI mode: Color-coded cells with statistics

#### Example Text Output

```
Initial filter applied: 'feature1 > 0'

Evaluation results for metric "auc":
|                                    |   model1 |   model2 |
|:-----------------------------------|---------:|---------:|
| [ 71: 29] 100.0%  all              |   0.9131 |   0.7028 |
| [ 33: 13]  46.0%  feature1 > 0     |   0.8485 |   0.7413 |
| [ 12:  8]  20.0%  feature2 < 2     |   0.9123 |   0.6543 |

Evaluation results for metric "precision":
|                                    |   model1 |   model2 |
|:-----------------------------------|---------:|---------:|
| [ 71: 29] 100.0%  all              |   0.8231 |   0.6028 |
| [ 33: 13]  46.0%  feature1 > 0     |   0.7785 |   0.6413 |
| [ 12:  8]  20.0%  feature2 < 2     |   0.8923 |   0.5543 |

Total evaluation time: 0.45 seconds
```

The output format includes:
- Query statistics in the format `[neg:pos] percentage query` where:
  - `neg:pos` shows the count of negative and positive samples
  - `percentage` shows what percentage of total samples this query represents
  - `query` is the actual query string
- Metric values for each model
- Total evaluation time

## Examples

### Basic Binary Classification
```bash
python evaluate.py -t test_set.csv -m model1.csv --metrics auc,precision,recall
```

### Multi-class with Specific Score Column
```bash
python evaluate.py -t test_set.csv -m model1.csv --score_col class1
```

### Multi-class with Multiple Positive Classes
```bash
python evaluate.py -t test_set.csv -m model1.csv --pos_classes class1 class2
```

### Interactive GUI with Filter
```bash
python evaluate.py -t test_set.csv -m model1.csv --gui --filter "feature1 > 0"
```

## Notes

- When using `--pos_classes` without `--score_col`, the tool will automatically sum the scores of all specified positive classes
- The ground truth column is converted to binary (0/1) where 1 indicates membership in the positive classes
- Available classes are determined from the unique values in the ground truth column
- The GUI allows dynamic selection of positive classes and threshold tuning
- The confusion matrix visualization helps identify patterns in model predictions:
  - Green cells indicate correct predictions
  - Red cells indicate incorrect predictions
  - Cell intensity indicates the magnitude of predictions
  - Statistics are shown side-by-side for easy comparison