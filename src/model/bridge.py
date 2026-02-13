import torch
import torch.nn as nn


class BridgeLayer(nn.Module):
    """Bridge between latent space and decoder: LayerNorm -> Linear with identity init."""

    def __init__(self, d_model: int, identity_init_scale: float = 0.01):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)

        # Identity initialization: weight ≈ I, bias = 0
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.eye(d_model) + identity_init_scale * torch.randn(d_model, d_model)
            )
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(x))
