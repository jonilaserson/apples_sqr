import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from common import GT_COL, SCORE_COL

def compute_max_f1(y_true, y_score):
    thresholds = np.unique(y_score)
    return max((f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds), default=None)

METRIC_FUNCTIONS = {
    'auc': roc_auc_score,
    'max_f1': compute_max_f1,
}

THRESH_METRICS_FUNCTIONS = {
    'precision': precision_score,
    'recall': recall_score,
    'accuracy': accuracy_score,
    'f1': f1_score,
}

def compute_metrics(models_df, queries, metrics, thresholds=None):
    raw_results = {}
    model_names = get_meta_columns_in_order(models_df)[1:]

    for query in queries:
        subset_df = models_df if query == 'all' else models_df.loc[models_df["test"].query(query).index]
        y_true = subset_df["test"][GT_COL]
        for idx, model_name in enumerate(model_names):
            if model_name == "test":
                continue
            model_data = subset_df[model_name]
            y_pred = model_data[SCORE_COL]
            if thresholds is not None:
                threshold = thresholds[idx]
                y_pred = (y_pred >= threshold).astype(int)

            metric_functions = THRESH_METRICS_FUNCTIONS if thresholds is not None else METRIC_FUNCTIONS
            for metric_name in metrics:
                metric_func = metric_functions.get(metric_name)
                if metric_func is not None:
                    raw_results.setdefault(metric_name, {}).setdefault(model_name, {})
                    metric_value = metric_func(y_true, y_pred)
                    raw_results[metric_name][model_name][query] = metric_value

    return raw_results

def apply_thresholds_and_evaluate(models_df, queries_dict, metrics, thresholds):
    raw_results = compute_metrics(models_df, list(queries_dict.keys()), metrics)
    raw_results_thresholds = compute_metrics(models_df, list(queries_dict.keys()), metrics, thresholds)
    raw_results.update(raw_results_thresholds)
    model_names = get_meta_columns_in_order(models_df)[1:]
    final_df = raw_results_to_final_df(raw_results, model_names, metrics, queries_dict)
    return final_df

def raw_results_to_final_df(raw_results, model_names, metrics, queries: dict):
    columns = pd.MultiIndex.from_product([metrics, model_names], names=['Metric', 'Model'])
    final_df = pd.DataFrame(index=list(queries.values()), columns=columns)

    for metric_name in metrics:
        for model_name in model_names:
            for query, query_label in queries.items():
                value = raw_results.get(metric_name, {}).get(model_name, {}).get(query, None)
                final_df.loc[query_label, (metric_name, model_name)] = value

    return final_df

def compute_confusion_elements(y_true, y_pred):
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    return tp, fp, fn, tn 