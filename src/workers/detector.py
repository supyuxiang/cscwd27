"""Three-layer veracity classifier.

The paper uses three fully connected layers, ReLU, and batch normalization.
ReLU and batch norm are applied on the two hidden layers. The last layer is a
linear map to two logits so a negative class score is not clipped.
"""

from __future__ import annotations

from torch import nn


class Detector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, num_classes),
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.net.modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        return self.net(features)
