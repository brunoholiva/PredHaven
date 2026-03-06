from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from rdkit import Chem


class Predictor(ABC):
    """Abstract base class for predictors."""

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the model from the specified path."""
        pass

    def check_validity(self, smiles_list: List[str]) -> List[str]:
        """Checks every SMILES str in the list.

        Drops invalid ones and returns a list of valid SMILES.

        """
        valid_smiles = []
        for smiles in smiles_list:
            if Chem.MolFromSmiles(smiles) is not None:
                valid_smiles.append(smiles)
        print(f"Valid SMILES: {len(valid_smiles)} out of {len(smiles_list)}")
        return valid_smiles

    @abstractmethod
    def prepare_input(self, valid_smiles_list: List[str]) -> Any:
        """Converts a list of SMILES strings into model-specific input."""
        pass

    @abstractmethod
    def predict_probability(self, prepared_input: Any) -> List[float]:
        """Takes the input data and returns a 1D array of probabilities for
        each input."""
        pass

    def predict(self, smiles_list: List[str]):
        """Manages the prediction process by validating input, preparing it,
        and then predicting probabilities."""
        valid_smiles = self.check_validity(smiles_list)
        prepared_input = self.prepare_input(valid_smiles)
        return self.predict_probability(prepared_input)


class SKLearnFingerprintPredictor(Predictor):
    """Predictor implementation using a SKLearn model/pipeline (trained
    with ECFP4 fingerprints)."""

    def __init__(self):
        self.model = None

    def load_model(self, model_path: str) -> None:
        """Load the model from the specified path."""
        import joblib

        self.model = joblib.load(model_path)

    def prepare_input(self, valid_smiles_list: List[str]) -> np.ndarray:
        """Takes a list of valid SMILES strings and converts them into ECFP4
        fingerprints suitable for the SKLearn model.

        Parameters
        ----------
        valid_smiles_list : List[str]
            A list of valid SMILES strings to be converted into fingerprints.

        Returns
        -------
        np.ndarray
            An array of ECFP4 fingerprints.

        """
        import numpy as np
        from rdkit.Chem import AllChem
        from rdkit import DataStructs, RDLogger

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
        """Takes the fingerprints and returns a 1D array of probabilities of
        active class for each input."""
        # sklearn returns probabilities for both classes
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        probabilities = self.model.predict_proba(prepared_input)[:, 1]

        return probabilities.tolist()
