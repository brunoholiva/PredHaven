from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class Metric:
    """A single evaluation metric.

    Parameters
    ----------
    name : str
        Human-readable name used as a column header in result DataFrames.
    fn : Callable[[pd.Series, pd.Series], float]
        ``fn(y_true, y_score) → float``.
    needs_both_classes : bool
        If *True* the metric returns NaN when only one class is present
        (e.g. AUC-ROC is undefined with a single class).

    """

    name: str
    fn: Callable[[pd.Series, pd.Series], float]
    needs_both_classes: bool = True


def _auc_roc(y_true: pd.Series, y_score: pd.Series) -> float:
    return roc_auc_score(y_true, y_score)


def _auc_pr(y_true: pd.Series, y_score: pd.Series) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


AUC_ROC = Metric("AUC-ROC", _auc_roc)
AUC_PR = Metric("AUC-PR", _auc_pr)
DEFAULT_METRICS: tuple[Metric, ...] = (AUC_ROC, AUC_PR)


class ModelEvaluator:
    """Automates the evaluation of multiple models' performances."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        metrics: Sequence[Metric] | None = None,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing the true labels and model predictions.
        target_col : str
            The name of the column containing the true binary labels.
        metrics : Sequence[Metric] | None
            Metrics to compute.  Defaults to ``DEFAULT_METRICS``
            (AUC-ROC, AUC-PR) when *None*.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.models: dict[str, str] = {}
        self.metrics = (
            list(metrics) if metrics is not None else list(DEFAULT_METRICS)
        )

    def add_model_predictions(self, model_name: str, prediction_col: str):
        """Registers a model's predictions for evaluation."""
        self.models[model_name] = prediction_col

    def _calculate_metrics(
        self, y_true: pd.Series, y_score: pd.Series
    ) -> dict[str, float]:
        """Compute every registered metric, returning ``{name: value}``."""
        single_class = len(y_true.unique()) < 2
        result: dict[str, float] = {}
        for m in self.metrics:
            if m.needs_both_classes and single_class:
                result[m.name] = float("nan")
            else:
                result[m.name] = m.fn(y_true, y_score)
        return result

    def get_metrics(self) -> pd.DataFrame:
        """Calculates overall metrics for all registered models.

        Parameters
        None
        ----------
        Returns
        -------
        pd.DataFrame
            A DataFrame with models as rows and metrics (AUC-ROC, AUC-PR) as columns.

        """
        true_labels = self.df[self.target_col]
        results = []

        for model_name, pred_col in self.models.items():
            metrics = self._calculate_metrics(true_labels, self.df[pred_col])
            metrics["Model"] = model_name
            results.append(metrics)

        return pd.DataFrame(results).set_index("Model")

    def get_metrics_by_group(self, group_col: str) -> pd.DataFrame:
        """Calculates evaluation metrics for each group (for example, AD
        region) for all models."""
        results = []

        for group_name, subset in self.df.groupby(group_col):
            true_labels = subset[self.target_col]

            for model_name, pred_col in self.models.items():
                metrics = self._calculate_metrics(
                    true_labels, subset[pred_col]
                )
                metrics["Group"] = group_name
                metrics["Model"] = model_name
                metrics["Group_Size"] = len(subset)
                results.append(metrics)

        return pd.DataFrame(results)
