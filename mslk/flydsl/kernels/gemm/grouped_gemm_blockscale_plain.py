# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Plain (non-preshuffled) B grouped FP8 blockscale GEMM — thin alias.

The plain-B and preshuffle-B kernels are unified in
grouped_gemm_blockscale_contiguous.py behind the `b_preshuffled` flag (they
share the entire kernel body; only the B load stage + its LDS allocation
differ). This module preserves the standalone
`compile_grouped_gemm_blockscale_plain(...)` entry point for callers/tests that
want the plain variant explicitly.
"""

from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
    compile_grouped_gemm_blockscale_contiguous,
)


def compile_grouped_gemm_blockscale_plain(
    *,
    n: int,
    k: int,
    num_groups: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
    waves_per_eu: int | None = None,
):
    """Compile the plain-B (non-preshuffled) grouped FP8 GEMM kernel.

    Delegates to compile_grouped_gemm_blockscale_contiguous with
    b_preshuffled=False. B is plain row-major [num_groups, N, K].
    """
    return compile_grouped_gemm_blockscale_contiguous(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        b_preshuffled=False,
    )
