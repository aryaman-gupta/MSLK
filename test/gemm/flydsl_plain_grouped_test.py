# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone correctness test for the plain-B FlyDSL grouped GEMM kernel.

Mirrors the MSLK f8f8bf16_groupwise_grouped op contract but feeds PLAIN
(un-preshuffled) wq directly to the new kernel launcher, comparing against a
per-group bf16 reference. Uses MSLK quantizers + scale layouts exactly as the
op wrapper does (scales passed through, no transpose). Run from a neutral cwd:
  cd /tmp && python -m pytest /workspace/MSLK/test/gemm/flydsl_plain_grouped_test.py -q
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
    compile_grouped_gemm_blockscale_contiguous,
)
from mslk.quantize.triton.fp8_quantize import quantize_fp8_block, quantize_fp8_group

_TILE_M = 128


def _run(m_values, N, K, tile_n=128, tile_k=128):
    device = "cuda"
    G = len(m_values)
    m_sizes = torch.tensor(m_values, dtype=torch.int64, device=device)
    TotalM = sum(m_values)

    x = torch.randn((TotalM, K), dtype=torch.bfloat16, device=device) * 0.1
    ws = [torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.01 for _ in range(G)]

    wq_list, ws_list = zip(
        *[quantize_fp8_block(w, block_m=128, block_k=128, k_major=False) for w in ws]
    )
    wq = torch.stack(wq_list, dim=0).contiguous()  # [G, N, K] PLAIN — NOT preshuffled
    w_scale = torch.stack(ws_list, dim=0).contiguous()
    xq, x_scale = quantize_fp8_group(x, m_sizes=m_sizes)

    # Host-known upper bound on M-tiles (matches the op wrapper); the kernel
    # resolves group ownership per tile from m_sizes and self-skips surplus tiles.
    num_m_tiles = TotalM // _TILE_M + G
    m_sizes_i32 = m_sizes.to(torch.int32)

    d = torch.zeros(TotalM, N, dtype=torch.bfloat16, device=device)
    launch_fn = compile_grouped_gemm_blockscale_contiguous(
        n=N, k=K, num_groups=G,
        tile_m=_TILE_M, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=128, scale_block_n=128, out_dtype="bf16",
        b_preshuffled=False,
    )
    launch_fn(
        d.view(-1),
        xq.contiguous().view(-1).view(torch.int8),
        wq.contiguous().view(-1).view(torch.int8),   # PLAIN B
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        m_sizes_i32,
        TotalM,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    # BF16 per-group reference over compact rows.
    ref_parts = []
    row = 0
    for g, m_g in enumerate(m_values):
        ref_parts.append((x[row : row + m_g] @ ws[g].t()).to(torch.bfloat16))
        row += m_g
    ref = torch.cat(ref_parts, dim=0)

    assert not d.isnan().any().item(), "Output contains NaN"
    assert not d.isinf().any().item(), "Output contains Inf"
    torch.testing.assert_close(d, ref, atol=8.0e-2, rtol=8.0e-2)


@pytest.mark.parametrize(
    "m_values,N,K",
    [
        pytest.param([128, 64], 128, 256, id="2g-128-64"),
        pytest.param([512, 256, 128], 256, 512, id="3g-mixed"),
        pytest.param([1, 128, 256], 128, 256, id="decode-prefill"),
        pytest.param([2048, 1024], 512, 512, id="2g-large"),
    ],
)
def test_plain_grouped_fp8_gemm(m_values, N, K):
    _run(m_values, N, K)


if __name__ == "__main__":
    _run([512, 256, 128], 256, 512)
    print("plain grouped GEMM: correctness OK")
