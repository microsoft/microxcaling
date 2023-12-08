import torch
import torch.nn.functional as F
import numpy as np
import argparse

from mx import finalize_mx_specs
from mx import mx_mapping

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

    # Simple MX spec for MXFP6 weights+activations
    mx_specs = {
        'w_elem_format': 'fp6_e3m2',
        'a_elem_format': 'fp6_e3m2',
        'block_size': 32,
        'bfloat': 16,
        'custom_cuda': True,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
    }
    mx_specs = finalize_mx_specs(mx_specs)

    # Auto-inject MX modules and functions
    # This will replace certain torch.nn.* and torch.nn.functional.*
    # modules/functions in the global namespace!
    mx_mapping.inject_pyt_ops(mx_specs)

    # Run MLP
    x = np.random.randn(16, args.hidden_size)
    x = torch.tensor(x, dtype=torch.float32, device=args.device)

    mlp = ResidualMLP(args.hidden_size)
    mlp.to(args.device)

    y = mlp(x)

    print("DONE!")
