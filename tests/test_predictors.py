import numpy as np
import pytest
from src.predictors import Predictor, RandomForestPredictor


class MockPredictor(Predictor):
    """A minimal, fake predictor to test the base class pipeline."""

    def load_model(self, model_path: str) -> None:
        pass

    def prepare_input(self, valid_smiles_list):
        return valid_smiles_list

    def predict_probability(self, prepared_input):
        # Fake probability for each valid input
        return [0.99] * len(prepared_input)


def test_check_validity_drops_invalid_smiles():
    """Test that check_validity correctly filters out bad SMILES."""
    predictor = MockPredictor()
    mixed_smiles = ["CCO", "INVALID_SMILES", "c1ccccc1"]

    valid_smiles = predictor.check_validity(mixed_smiles)

    assert len(valid_smiles) == 2
    assert "INVALID_SMILES" not in valid_smiles
    assert "CCO" in valid_smiles


def test_predict_pipeline_empty_input():
    """Test the edge case where all SMILES are invalid."""
    predictor = MockPredictor()
    bad_smiles = ["FAKE1", "FAKE2"]

    predictions = predictor.predict(bad_smiles)

    assert (
        predictions == []
    )  


def test_rf_prepare_input_generates_correct_arrays():
    """Test if RDKit outputs the exact shape and type we need for Scikit-
    Learn."""
    predictor = RandomForestPredictor()
    valid_smiles = ["CCO", "c1ccccc1"]

    fps = predictor.prepare_input(valid_smiles)

    assert isinstance(fps, np.ndarray), "Output must be a numpy array"
    assert fps.shape == (2, 2048), "Shape must be (num_molecules, 2048 bits)"
    assert fps.dtype == np.int8, "Dtype must be int8 as defined in your code"


class DummySklearnModel:
    """A fake sklearn model."""

    def predict_proba(self, X):
        return np.array([[0.15, 0.85] for _ in range(len(X))])


def test_rf_predict_probability_slices_correctly():
    """Test if we correctly slice the active class probabilities."""
    predictor = RandomForestPredictor()
    predictor.model = DummySklearnModel()

    fake_fingerprints = np.zeros((3, 2048))

    probs = predictor.predict_probability(fake_fingerprints)

    assert isinstance(probs, list), "Must return a Python list"
    assert len(probs) == 3, "Must return one probability per input"
    assert (
        probs[0] == 0.85
    ), "Must extract the 2nd column (index 1) from predict_proba"


def test_rf_unloaded_model_raises_error():
    """Test that the class throws an error if we forget to load the model."""
    predictor = RandomForestPredictor()
    fake_fingerprints = np.zeros((1, 2048))

    with pytest.raises(ValueError, match="Model not loaded"):
        predictor.predict_probability(fake_fingerprints)
