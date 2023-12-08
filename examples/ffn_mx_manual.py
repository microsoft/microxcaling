import torch
import torch.nn.functional as F
import numpy as np
import argparse

from mx import Linear, LayerNorm
from mx import gelu, simd_split, simd_add
from mx import add_mx_args, get_mx_specs

class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_size, mx_specs):
        super(ResidualMLP, self).__init__()

        self.mx_specs = mx_specs

        self.layernorm = LayerNorm(
            hidden_size,
            mx_specs=mx_specs
        )

        self.dense_4h = Linear(
            hidden_size,
            4 * hidden_size,
            mx_specs=mx_specs
        )

        self.dense_h = Linear(
            4 * hidden_size,
            hidden_size,
            mx_specs=mx_specs
        )

    def forward(self, inputs):
        inputs, residual = simd_split(inputs)

        norm_outputs = self.layernorm(inputs)

        # MLP
        proj_outputs = self.dense_4h(norm_outputs)
        proj_outputs = gelu(proj_outputs,
                            mx_specs=self.mx_specs)
        mlp_outputs = self.dense_h(proj_outputs)

        # Residual Connection
        outputs = simd_add(residual, mlp_outputs,
                           mx_specs=self.mx_specs)

        return outputs


if __name__ == '__main__':
    # Add config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=128)
    parser.add_argument("--device", default='cuda')
    # Add MX arguments
    parser = add_mx_args(parser)
    args = parser.parse_args()

    # Process args to obtain mx_specs
    mx_specs = get_mx_specs(args)
    assert(mx_specs != None)

    # Run MLP
    x = np.random.randn(16, args.hidden_size)
    x = torch.tensor(x, dtype=torch.float32, device=args.device)

    mlp = ResidualMLP(args.hidden_size, mx_specs)
    mlp.to(args.device)

    y = mlp(x)

    print("DONE!")
