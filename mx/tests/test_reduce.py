"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import sys
import pytest
import torch
import numpy as np

from .common_lib import check_diff

# Import custom CUDA library to pre-compile libraries
import mx.custom_extensions as custom_extensions


@pytest.mark.parametrize("func1, func2", [
    (torch.sum, custom_extensions.funcs.reduce_sum_inner_dim),
    (torch.max, custom_extensions.funcs.reduce_max_inner_dim),
])
@pytest.mark.parametrize("S, H", [
    (5, 32),       # uses 1 block of 1024 threads
    (35, 32),      # uses multiple blocks
    (7, 512),
    (3, 1024),
    (15, 2048),
    (11, 4096),
])
def test_reduce(func1, func2, S, H):
    iterations = 5

    for _ in range(iterations):
        m_ = np.random.randn(512, S, H)
        m1 = torch.tensor(m_, dtype=torch.float32, device='cuda', requires_grad=True)
        m2 = torch.tensor(m_, dtype=torch.float32, device='cuda', requires_grad=True)

        s1 = func1(m1, dim=-1)
        # torch.max returns either a tuple or a torch.return_types.max
        if type(s1) is not torch.Tensor:
            s1 = s1[0]

        m2 = m2.contiguous()
        s2 = func2(m2)
        torch.cuda.synchronize()

        check_diff(s1, s2, tol=1e-6)
