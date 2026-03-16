"""Provide plotting utilities for evaluated model predictions."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

if TYPE_CHECKING:
    from src.evaluator import ModelEvaluator


class Visualizer:
    """Handle plotting for models evaluated by ``ModelEvaluator``."""

    def __init__(self, evaluator: "ModelEvaluator"):
        """Initialize the visualizer with an evaluator instance."""
        self.evaluator = evaluator
        self.df = evaluator.df
        self.target_col = evaluator.target_col
        self.models = evaluator.models

    def plot_roc_curves(self, save_path: str = None):
        """Plot ROC curves for all registered models."""
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Random")

        y_true = self.df[self.target_col]

        for model_name, pred_col in self.models.items():
            y_score = self.df[pred_col]
            # Only plot if we have both classes
            if len(y_true.unique()) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    def plot_pr_curves(self, save_path: str = None):
        """Plot precision-recall curves for all registered models."""
        plt.figure(figsize=(8, 6))
        y_true = self.df[self.target_col]
        baseline = y_true.sum() / len(y_true)
        plt.plot(
            [0, 1],
            [baseline, baseline],
            "k--",
            label=f"Baseline ({baseline:.2f})",
        )

        for model_name, pred_col in self.models.items():
            y_score = self.df[pred_col]
            if len(y_true.unique()) > 1:
                prec, rec, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(rec, prec)
                plt.plot(rec, prec, label=f"{model_name} (AUC = {pr_auc:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="lower left")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()
