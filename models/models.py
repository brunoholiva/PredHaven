"""Module for model architectures."""

import torch
import torch.nn as nn
from transformers import AutoModel


class MolFormerQSAR(nn.Module):
    """QSAR model mirroring the exact architecture from the MoLFormer paper."""

    def __init__(
        self,
        model_name: str = "ibm/MoLFormer-XL-both-10pct",
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name, deterministic_eval=True, trust_remote_code=True
        )

        hidden_size = self.encoder.config.hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu2 = nn.GELU()

        self.final = nn.Linear(hidden_size, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the forward pass to compute logits for the positive class."""
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        token_embeddings = outputs.last_hidden_state

        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        smiles_emb = sum_embeddings / sum_mask

        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        x_out = x_out + smiles_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)

        logits = self.final(z + x_out)

        return logits

    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict probabilities for the positive class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)
        return probabilities
