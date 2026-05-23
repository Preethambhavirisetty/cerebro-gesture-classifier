"""
GestureNet — 3-layer MLP for gesture classification.

Architecture: 14 → 32 → 16 → 8
  • ReLU activations after layers 1 and 2
  • No softmax in forward() — cross-entropy loss includes log-softmax

Layer naming (fc1, fc2, fc3) is load-bearing: export_onnx.py reads
state_dict keys by these exact names.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_engineer import K_NUM_FEATURES, K_NUM_CLASSES


class GestureNet(nn.Module):
    def __init__(self, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.fc1     = nn.Linear(K_NUM_FEATURES, 32)
        self.fc2     = nn.Linear(32, 16)
        self.fc3     = nn.Linear(16, K_NUM_CLASSES)
        self.drop    = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)   # logits — no softmax here

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class indices (argmax over softmax), no dropout."""
        self.eval()
        return self.forward(x).argmax(dim=-1)
