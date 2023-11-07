import torch
import torch.nn.functional as F
import numpy as np
import argparse

class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super(ResidualMLP, self).__init__()

        self.layernorm = torch.nn.LayerNorm(
            hidden_size
        )

        self.dense_4h = torch.nn.Linear(
            hidden_size,
            4 * hidden_size
        )

        self.dense_h = torch.nn.Linear(
            4 * hidden_size,
            hidden_size
        )

    def forward(self, inputs):
        norm_outputs = self.layernorm(inputs)
        
        # MLP
        proj_outputs = self.dense_4h(norm_outputs)
        proj_outputs = F.gelu(proj_outputs)
        mlp_outputs = self.dense_h(proj_outputs)

        # Residual Connection
        outputs = inputs + mlp_outputs

        return outputs


if __name__ == '__main__':
    # Add config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=128)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()

    # Run MLP
    x = np.random.randn(16, args.hidden_size)
    x = torch.tensor(x, dtype=torch.float32, device=args.device)

    mlp = ResidualMLP(args.hidden_size)
    mlp.to(args.device)

    y = mlp(x)

    print("DONE!")
