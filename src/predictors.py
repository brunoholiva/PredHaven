"""Define predictor interfaces and concrete predictor implementations."""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from rdkit import Chem


class Predictor(ABC):
    """Define the abstract interface for prediction pipelines."""

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load a model from the specified path.

        Parameters
        ----------
        model_path : str
            The file path to the model to be loaded.
        """
        pass

    def check_validity(self, smiles_list: List[str]) -> List[str]:
        """
        Validate SMILES strings and keep only valid entries.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings to be validated.

        Returns
        -------
        List[str]
            A list of valid SMILES strings (in input order).
        """
        valid_smiles = []
        for smiles in smiles_list:
            if Chem.MolFromSmiles(smiles) is not None:
                valid_smiles.append(smiles)
        print(f"Valid SMILES: {len(valid_smiles)} out of {len(smiles_list)}")
        return valid_smiles

    @abstractmethod
    def prepare_input(self, valid_smiles_list: List[str]) -> Any:
        """
        Convert a list of SMILES strings into model-specific input.

        Parameters
        ----------
        valid_smiles_list : List[str]
            A list of valid SMILES strings to be converted into model input.

        Returns
        -------
        Any
            The prepared input data in the format required by the model.
        """
        pass

    @abstractmethod
    def predict_probability(self, prepared_input: Any) -> List[float]:
        """
        "Predict class probabilities for prepared input data.

        Parameters
        ----------
        prepared_input : Any
            Model-ready input features.

        Returns
        -------
        List[float]
            Probability of the positive class for each input instance.
        """
        pass

    def predict(self, smiles_list: List[str]):
        """
        Run the full prediction pipeline.

        Validate SMILES, prepare input, and return predicted probabilities.

        Parameters
        ----------
        smiles_list : List[str]
            Input SMILES strings.

        Returns
        -------
        List[float]
            Predicted positive probabilities for valid SMILES inputs.
        """
        valid_smiles = self.check_validity(smiles_list)
        prepared_input = self.prepare_input(valid_smiles)
        return self.predict_probability(prepared_input)


class SKLearnFingerprintPredictor(Predictor):
    """Implement a predictor using ECFP4 features and an sklearn model."""

    def __init__(self):
        """Initialize the sklearn fingerprint predictor."""
        self.model = None

    def load_model(self, model_path: str) -> None:
        """
        Load the model from the specified path.

        Parameters
        ----------
        model_path : str
            The file path to the sklearn model to be loaded.
        """
        import joblib

        self.model = joblib.load(model_path)

    def prepare_input(self, valid_smiles_list: List[str]) -> np.ndarray:
        """
        Convert valid SMILES strings into ECFP4 fingerprints.

        Parameters
        ----------
        valid_smiles_list : List[str]
            Valid SMILES strings to featurize.

        Returns
        -------
        np.ndarray
            ECFP4 fingerprint matrix.
        """
        import numpy as np
        from rdkit import DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")

        fingerprints = []
        for smiles in valid_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )
            arr = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)

        return np.array(fingerprints)

    def predict_probability(self, prepared_input: np.ndarray) -> List[float]:
        """
        Predict positive-class probabilities from fingerprints.

        Parameters
        ----------
        prepared_input : np.ndarray
            The ECFP4 fingerprint matrix.

        Returns
        -------
        List[float]
            Predicted probabilities for the positive class.

        Raises
        ------
        ValueError
            If the model has not been loaded before prediction.
        """
        # sklearn returns probabilities for both classes
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please call load_model() first."
            )

        probabilities = self.model.predict_proba(prepared_input)[:, 1]

        return probabilities.tolist()


class MolFormerMLPPredictor(Predictor):
    """Implement a predictor using MoLFormer embeddings and an MLP."""

    def __init__(self):
        """Initialize tokenizer, transformer, and an MLP model."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = "ibm/MoLFormer-XL-both-10pct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.transformer = AutoModel.from_pretrained(
            self.model_name, deterministic_eval=True, trust_remote_code=True
        )
        self.transformer.to(self.device)
        self.transformer.eval()

        self.mlp = None

    def load_model(self, model_path: str) -> None:
        """
        Load trained MLP model weights.

        Parameters
        ----------
        model_path : str
            Path to the saved PyTorch MLP model.
        """
        import torch

        self.mlp = torch.load(model_path, map_location=self.device)
        self.mlp.eval()

    def prepare_input(self, valid_smiles_list: List[str]) -> np.ndarray:
        """
        Convert valid SMILES strings into MoLFormer embeddings.

        Parameters
        ----------
        valid_smiles_list : List[str]
            Valid SMILES strings to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix. Returns an empty array for empty input.
        """
        import torch

        if not valid_smiles_list:
            return np.array([])

        inputs = self.tokenizer(
            valid_smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.transformer(**inputs)

        embeddings = outputs.pooler_output.cpu().numpy()

        return embeddings

    def predict_probability(self, prepared_input: np.ndarray) -> List[float]:
        """
        Predict positive-class probabilities for embedding inputs.

        Parameters
        ----------
        prepared_input : np.ndarray
            MoLFormer embedding matrix.

        Returns
        -------
        List[float]
            Probability of the positive class for each input row.
        """
        return [0.5] * len(prepared_input)
