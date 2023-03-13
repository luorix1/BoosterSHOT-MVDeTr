import torch
import torch.nn as nn


class GateModule(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.input_dim = in_ch
        self.output_dim = out_ch

        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_dim),
        )

    def forward(self, x):
        res = self.gate_fc(x)
        return res
