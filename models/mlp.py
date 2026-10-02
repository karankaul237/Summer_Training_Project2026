"""
Step 5, part 1: model architecture.

Deliberately small (2 hidden layers, dropout) -- per our overfitting
discussion, a large model memorizes small clients like Hospital (16K rows)
and WUSTL's rare spoofing_mitm class easily. This also keeps CPU training
fast on low-RAM machines and keeps DP-SGD's per-sample gradient computation
(Step 8) tractable.
"""

import torch
import torch.nn as nn

try:
    from ..data import schema
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    import schema


class IDSModel(nn.Module):
    def __init__(self, input_dim: int = schema.INPUT_DIM,
                 num_classes: int = len(schema.CLASSES),
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits; loss function applies softmax


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = IDSModel()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    dummy = torch.randn(4, schema.INPUT_DIM)
    out = model(dummy)
    print(f"Output shape for batch of 4: {out.shape}  (expect [4, {len(schema.CLASSES)}])")