import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from src.evaluator import DEFAULT_METRICS, Metric, ModelEvaluator


@pytest.fixture
def sample_df():
    """Reusable dataset with two models and two groups."""
    return pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 1, 0],
            "model_a_good": [0.05, 0.95, 0.20, 0.85, 0.80, 0.10],
            "model_b_bad": [0.90, 0.10, 0.70, 0.20, 0.30, 0.80],
            "region": ["in", "in", "in", "out", "out", "out"],
        }
    )


@pytest.fixture
def evaluator(sample_df):
    """Reusable evaluator with two registered models."""
    ev = ModelEvaluator(sample_df, target_col="y")
    ev.add_model_predictions("good_model", "model_a_good")
    ev.add_model_predictions("bad_model", "model_b_bad")
    return ev


@pytest.fixture
def ef_df():
    """100-row dataset: 10 actives, 90 inactives."""
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.array(
        [0.9 + 0.01 * i for i in range(10)] + [0.01 * i for i in range(90)]
    )
    bad = 1.0 - perfect
    return pd.DataFrame({"y": y, "perfect": perfect, "bad": bad})


@pytest.fixture
def ef_evaluator(ef_df):
    ev = ModelEvaluator(ef_df, target_col="y")
    ev.add_model_predictions("perfect", "perfect")
    ev.add_model_predictions("bad", "bad")
    return ev


class TestBaseMetrics:
    """Tests for standard metrics calculation and the custom Metric
    dataclass."""

    def test_get_metrics_structure(self, evaluator):
        metrics = evaluator.get_metrics()
        assert set(metrics.index) == {"good_model", "bad_model"}
        assert list(metrics.columns) == ["AUC-ROC", "AUC-PR"]

    @pytest.mark.parametrize(
        "model_name,pred_col",
        [("good_model", "model_a_good"), ("bad_model", "model_b_bad")],
    )
    def test_get_metrics_values(
        self, sample_df, evaluator, model_name, pred_col
    ):
        metrics = evaluator.get_metrics()
        y_true = sample_df["y"]
        y_proba = sample_df[pred_col]

        expected_auc_roc = roc_auc_score(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        expected_auc_pr = auc(recall, precision)

        assert metrics.loc[model_name, "AUC-ROC"] == pytest.approx(
            expected_auc_roc
        )
        assert metrics.loc[model_name, "AUC-PR"] == pytest.approx(
            expected_auc_pr
        )

    def test_good_beats_bad(self, evaluator):
        metrics = evaluator.get_metrics()
        assert (
            metrics.loc["good_model", "AUC-ROC"]
            > metrics.loc["bad_model", "AUC-ROC"]
        )
        assert (
            metrics.loc["good_model", "AUC-PR"]
            > metrics.loc["bad_model", "AUC-PR"]
        )

    @pytest.mark.parametrize(
        "labels,preds",
        [
            ([1, 1, 1], [0.2, 0.8, 0.6]),
            ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]),
        ],
    )
    def test_single_class_returns_nan_metrics(self, labels, preds):
        df = pd.DataFrame({"y": labels, "pred": preds})
        ev = ModelEvaluator(df, target_col="y")
        ev.add_model_predictions("m1", "pred")

        metrics = ev.get_metrics()
        assert np.isnan(metrics.loc["m1", "AUC-ROC"])
        assert np.isnan(metrics.loc["m1", "AUC-PR"])

    def test_default_metrics_are_auc_roc_and_auc_pr(self):
        assert [m.name for m in DEFAULT_METRICS] == ["AUC-ROC", "AUC-PR"]

    def test_custom_metric_is_used(self):
        always_one = Metric("always_one", lambda y, p: 1.0)
        df = pd.DataFrame({"y": [0, 1], "pred": [0.1, 0.9]})
        ev = ModelEvaluator(df, target_col="y", metrics=[always_one])
        ev.add_model_predictions("m", "pred")

        result = ev.get_metrics()
        assert list(result.columns) == ["always_one"]
        assert result.loc["m", "always_one"] == 1.0

    def test_custom_metric_needs_both_classes_false(self):
        count = Metric(
            "count", lambda y, p: float(len(y)), needs_both_classes=False
        )
        df = pd.DataFrame({"y": [1, 1, 1], "pred": [0.5, 0.6, 0.7]})
        ev = ModelEvaluator(df, target_col="y", metrics=[count])
        ev.add_model_predictions("m", "pred")

        result = ev.get_metrics()
        assert result.loc["m", "count"] == 3.0


class TestGroupMetrics:
    """Tests for get_metrics_by_group (e.g., Applicability Domain analysis)."""

    def test_get_metrics_by_group_shape_and_columns(self, evaluator):
        grouped = evaluator.get_metrics_by_group("region")
        assert len(grouped) == 4
        assert {"Group", "Model", "Group_Size", "AUC-ROC", "AUC-PR"}.issubset(
            grouped.columns
        )

    @pytest.mark.parametrize(
        "group_name,expected_size",
        [("in", 3), ("out", 3)],
    )
    def test_get_metrics_by_group_sizes(
        self, evaluator, group_name, expected_size
    ):
        grouped = evaluator.get_metrics_by_group("region")
        sizes = grouped[grouped["Group"] == group_name]["Group_Size"].unique()
        assert len(sizes) == 1
        assert sizes[0] == expected_size

    def test_group_with_single_class_has_nan(self):
        df = pd.DataFrame(
            {
                "y": [0, 1, 1, 1],
                "pred": [0.1, 0.8, 0.7, 0.9],
                "region": ["in", "in", "out", "out"],
            }
        )
        ev = ModelEvaluator(df, target_col="y")
        ev.add_model_predictions("m1", "pred")

        grouped = ev.get_metrics_by_group("region")
        out_row = grouped[
            (grouped["Group"] == "out") & (grouped["Model"] == "m1")
        ].iloc[0]

        assert np.isnan(out_row["AUC-ROC"])
        assert np.isnan(out_row["AUC-PR"])

    def test_custom_metric_in_group_metrics(self):
        fixed = Metric("fixed", lambda y, p: 42.0)
        df = pd.DataFrame(
            {
                "y": [0, 1, 0, 1],
                "pred": [0.1, 0.9, 0.2, 0.8],
                "g": ["a", "a", "b", "b"],
            }
        )
        ev = ModelEvaluator(df, target_col="y", metrics=[fixed])
        ev.add_model_predictions("m", "pred")

        grouped = ev.get_metrics_by_group("g")
        assert "fixed" in grouped.columns
        assert (grouped["fixed"] == 42.0).all()


class TestEnrichmentFactor:
    """Tests for ranking actives and calculating the Enrichment Factor."""

    def test_ef_structure(self, ef_evaluator):
        result = ef_evaluator.get_enrichment_factor(top_fraction=0.1)
        assert set(result.index) == {"perfect", "bad"}
        assert "Enrichment_Factor" in result.columns
        assert "Actives_in_Top_10.0%" in result.columns

    def test_ef_perfect_model(self, ef_evaluator):
        result = ef_evaluator.get_enrichment_factor(top_fraction=0.1)
        assert result.loc["perfect", "Actives_in_Top_10.0%"] == 10
        assert result.loc["perfect", "Enrichment_Factor"] == pytest.approx(
            10.0
        )

    def test_ef_bad_model(self, ef_evaluator):
        result = ef_evaluator.get_enrichment_factor(top_fraction=0.1)
        assert result.loc["bad", "Actives_in_Top_10.0%"] == 0
        assert result.loc["bad", "Enrichment_Factor"] == pytest.approx(0.0)

    def test_ef_perfect_beats_bad(self, ef_evaluator):
        result = ef_evaluator.get_enrichment_factor(top_fraction=0.1)
        assert (
            result.loc["perfect", "Enrichment_Factor"]
            > result.loc["bad", "Enrichment_Factor"]
        )

    def test_ef_custom_fraction(self):
        df = pd.DataFrame(
            {
                "y": [1, 1, 0, 0, 0, 0],
                "pred": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
            }
        )
        ev = ModelEvaluator(df, target_col="y")
        ev.add_model_predictions("m", "pred")

        result = ev.get_enrichment_factor(top_fraction=0.5)
        assert result.loc["m", "Enrichment_Factor"] == pytest.approx(2.0)

    def test_ef_top_k_at_least_one(self):
        df = pd.DataFrame({"y": [0, 1], "pred": [0.1, 0.9]})
        ev = ModelEvaluator(df, target_col="y")
        ev.add_model_predictions("m", "pred")

        result = ev.get_enrichment_factor(top_fraction=0.01)
        assert result.loc["m", "Enrichment_Factor"] == pytest.approx(2.0)

    def test_ef_no_actives_returns_nan(self):
        df = pd.DataFrame({"y": [0, 0, 0], "pred": [0.9, 0.5, 0.1]})
        ev = ModelEvaluator(df, target_col="y")
        ev.add_model_predictions("m", "pred")

        result = ev.get_enrichment_factor(top_fraction=0.5)
        assert np.isnan(result.loc["m", "Enrichment_Factor"])

    def test_ef_all_actives(self):
        df = pd.DataFrame({"y": [1, 1, 1, 1], "pred": [0.9, 0.7, 0.5, 0.3]})
        ev = ModelEvaluator(df, target_col="y")
        ev.add_model_predictions("m", "pred")

        result = ev.get_enrichment_factor(top_fraction=0.5)
        assert result.loc["m", "Enrichment_Factor"] == pytest.approx(1.0)
