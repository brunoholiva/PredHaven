"""Provide classes and functions to evaluate model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class Metric:
    """
    Represent a single evaluation metric.

    Parameters
    ----------
    name : str
        Human-readable name used as a column header in result DataFrames.
    fn : Callable[[pd.Series, pd.Series], float]
        Callable with signature ``fn(y_true, y_score) -> float``.
    needs_both_classes : bool
        Return NaN when only one class is present if *True*.
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
    """Automate evaluation of multiple model predictions."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        metrics: Sequence[Metric] | None = None,
    ):
        """
        Initialize the evaluator with data and target labels.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing true labels and model predictions.
        target_col : str
            Name of the column containing true binary labels.
        metrics : Sequence[Metric] | None
            Metrics to compute. Defaults to ``DEFAULT_METRICS`` when *None*.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.models: dict[str, str] = {}
        self.metrics = (
            list(metrics) if metrics is not None else list(DEFAULT_METRICS)
        )

    def add_model_predictions(self, model_name: str, prediction_col: str):
        """Register model predictions for evaluation."""
        self.models[model_name] = prediction_col

    def _calculate_metrics(
        self, y_true: pd.Series, y_score: pd.Series
    ) -> dict[str, float]:
        """
        Compute registered metrics and return ``{name: value}``.

        Parameters
        ----------
        y_true : pd.Series
            True binary labels.
        y_score : pd.Series
            Predicted probabilities or scores.

        Returns
        -------
        dict[str, float]
            Mapping from metric name to computed value.
        """
        single_class = len(y_true.unique()) < 2
        result: dict[str, float] = {}
        for m in self.metrics:
            if m.needs_both_classes and single_class:
                result[m.name] = float("nan")
            else:
                result[m.name] = m.fn(y_true, y_score)
        return result

    def get_metrics(self) -> pd.DataFrame:
        """
        Get overall metrics for all registered models.

        Returns
        -------
        pd.DataFrame
            DataFrame with models as rows and metrics as columns.
        """
        true_labels = self.df[self.target_col]
        results = []

        for model_name, pred_col in self.models.items():
            metrics = self._calculate_metrics(true_labels, self.df[pred_col])
            metrics["Model"] = model_name
            results.append(metrics)

        return pd.DataFrame(results).set_index("Model")

    def get_metrics_by_group(self, group_col: str) -> pd.DataFrame:
        """
        Calculate evaluation metrics for each group.

        For example, use this to evaluate performance by AD region for all
        registered models.

        Parameters
        ----------
        group_col : str
            Column name used for grouping (for example, ``"region"``).

        Returns
        -------
        pd.DataFrame
            DataFrame with Group, Model, Group_Size, and metric columns.
        """
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

    def get_enrichment_factor(
        self, top_fraction: float = 0.01
    ) -> pd.DataFrame:
        """
        Calculate the enrichment factor for the top-ranked predictions.

        Parameters
        ----------
        top_fraction : float
            Fraction of rows considered as the top set (for example, 0.01).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by model with active-count and enrichment
            columns.
        """
        true_labels = self.df[self.target_col]
        total_actives = true_labels.sum()
        total_molecules = len(true_labels)
        base_hit_rate = total_actives / total_molecules

        top_k = max(1, int(total_molecules * top_fraction))

        results = []
        for model_name, pred_col in self.models.items():
            top_k_df = self.df.nlargest(top_k, pred_col)

            actives_found = top_k_df[self.target_col].sum()
            model_hit_rate = actives_found / top_k

            ef = (
                model_hit_rate / base_hit_rate
                if base_hit_rate > 0
                else float("nan")
            )

            results.append(
                {
                    "Model": model_name,
                    f"Actives_in_Top_{top_fraction * 100}%": actives_found,
                    "Enrichment_Factor": ef,
                }
            )

        return pd.DataFrame(results).set_index("Model")

    def get_model_correlation(self, method: str = "spearman") -> pd.DataFrame:
        """
        Calculate correlation between model predictions.

        Parameters
        ----------
        method : str
            Correlation method (for example, ``"pearson"`` or ``"spearman"``).

        Returns
        -------
        pd.DataFrame
            Correlation matrix with model names in index and columns.
        """
        pred_cols = list(self.models.values())
        corr_matrix = self.df[pred_cols].corr(method=method)

        rename = {col: name for name, col in self.models.items()}
        corr_matrix = corr_matrix.rename(columns=rename, index=rename)

        return corr_matrix

    def get_error_correlation(self, method: str = "spearman") -> pd.DataFrame:
        """
        Calculate correlation between model prediction errors.

        High correlation indicates similar mistakes across models. Low
        correlation indicates more complementary errors.

        Parameters
        ----------
        method : str
            Correlation method (``"spearman"`` is often useful for ranks).

        Returns
        -------
        pd.DataFrame
            Correlation matrix of residuals ``(y_true - y_pred)``.
        """
        true_labels = self.df[self.target_col].astype(float)
        error_df = pd.DataFrame()

        for model_name, pred_col in self.models.items():
            error_df[model_name] = true_labels - self.df[pred_col]

        corr_matrix = error_df.corr(method=method)
        return corr_matrix

    def get_ranking_agreement(self, top_k: int = 100) -> pd.DataFrame:
        """
        Measure overlap of top-k ranked molecules across models.

        Parameters
        ----------
        top_k : int
            Number of top-ranked molecules to compare.

        Returns
        -------
        pd.DataFrame
            Pairwise agreement matrix with top-k overlap fractions.
        """
        results = []

        model_names = list(self.models.keys())
        for i, model_a in enumerate(model_names):
            row = {}
            pred_col_a = self.models[model_a]
            top_a = set(self.df.nlargest(top_k, pred_col_a).index)

            for model_b in model_names:
                if model_a == model_b:
                    row[model_b] = 1.0
                else:
                    pred_col_b = self.models[model_b]
                    top_b = set(self.df.nlargest(top_k, pred_col_b).index)
                    overlap = len(top_a & top_b) / top_k
                    row[model_b] = overlap

            results.append(row)

        return pd.DataFrame(results, index=model_names)
