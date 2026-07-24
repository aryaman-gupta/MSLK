# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 groupwise-scaled grouped GEMM (preshuffled weights) via FlyDSL.

Registers ``mslk::f8f8bf16_groupwise_grouped_preshuffle``, the ROCm sibling of
``mslk::f8f8bf16_groupwise_grouped`` that consumes weights already in the MFMA
B-preshuffle layout (see ``mslk.quantize.shuffle.preshuffle_b_mfma``). Callers
shuffle weights once at load time; the op does no shuffling.

Tensor contract:
  XQ      : [TotalM, K]             FP8  -- all groups concatenated along M
  WQ      : [G, N, K]               FP8  -- per-group weights, MFMA-preshuffled
  x_scale : [K//128, TotalM]        FP32 -- per-token per-128K scales (transposed)
  w_scale : [G, K//128, N//128]     FP32 -- per-group per-block scales
  M_sizes : [G]                     int64 -- rows per group (sum to TotalM)
  Output  : [TotalM, N]             BF16
"""

import os

import torch

from mslk.utils.flydsl import is_flydsl_available, run_compiled

_OP_NAME = "mslk::f8f8bf16_groupwise_grouped_preshuffle"

# Only the scale-block granularity is fixed; tile_m/tile_n/tile_k are chosen per
# call -- either by FlyDSL autotune (MSLK_AUTOTUNE_ENABLE set) or a fixed default.
_SCALE_BLOCK = 128

# Default tile when autotuning is disabled. Valid for any supported shape
# (tile_n=tile_k=128 divide every supported N/K, incl. small N=128). This is the
# CI / no-benchmark path -- matches the CUTLASS heuristic fallback tile.
_DEFAULT_TILE = (128, 128, 128)

# Candidate tile space swept by autotune. tile_m multiple of 16; tile_n multiple
# of scale_block_n=128; tile_k=128. Configs invalid for a given shape (tile_n>N
# or N%tile_n) are pruned per-call before benchmarking.
_AUTOTUNE_TILES = (
    (64, 128, 128),
    (128, 128, 128),
    (256, 128, 128),
    (64, 256, 128),
    (128, 256, 128),
    (256, 256, 128),
)


def _next_pow2(x: int) -> int:
    """Smallest power of two >= x (x>=1). Buckets TotalM for the autotune key so
    nearby token counts share one tuned config -- matching the CUDA-graph capture
    buckets a server pre-captures, and bounding the pre-warm set."""
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _launch_kernel(
    XQ, WQ, x_scale, w_scale, m_sizes_i32, output, *, tile_m, tile_n, tile_k, b_preshuffled
):
    """Compile (cached) and launch the grouped GEMM for one tile config. Shared
    by the autotune target and the fixed-config path. Writes into `output`."""
    from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
        compile_grouped_gemm_blockscale_contiguous,
    )

    TotalM, K = XQ.shape
    G, N, _ = WQ.shape
    # Grid M-extent: host-known upper bound (each group wastes at most one partial
    # tile). The kernel resolves group ownership from M_sizes and self-skips
    # surplus tiles.
    num_m_tiles = TotalM // tile_m + G
    launcher = compile_grouped_gemm_blockscale_contiguous(
        n=N,
        k=K,
        num_groups=G,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=_SCALE_BLOCK,
        scale_block_n=_SCALE_BLOCK,
        out_dtype="bf16",
        b_preshuffled=b_preshuffled,
    )
    run_compiled(
        launcher,
        output.view(-1),
        XQ.contiguous().view(-1).view(torch.int8),
        WQ.contiguous().view(-1).view(torch.int8),
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
    return output

torch.library.define(
    _OP_NAME,
    "(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor",
)


@torch.library.impl(_OP_NAME, "Meta")
def _f8f8bf16_groupwise_grouped_preshuffle_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    TotalM = XQ.shape[0]
    N = WQ.shape[1]
    return XQ.new_empty((TotalM, N), dtype=torch.bfloat16)


def _autotune_target(
    XQ, WQ, x_scale, w_scale, m_sizes_i32, output, m_bucket, n, k, b_preshuffled,
    *, tile_m, tile_n, tile_k,
):
    """FlyDSL @autotune benchmarks this per candidate tile. Keyed on
    (m_bucket, n, k, b_preshuffled): m_bucket=nextPow2(TotalM) buckets token
    counts; n/k separate the problem shapes (gate/up vs down-proj want different
    tiles); b_preshuffled distinguishes the two kernels (different B-load path,
    can't share a tuned config). Key args are otherwise passed straight through.
    tile_* arrive as Config kwargs."""
    return _launch_kernel(
        XQ, WQ, x_scale, w_scale, m_sizes_i32, output,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, b_preshuffled=b_preshuffled,
    )


def _prune_tiles(configs, named_args):
    """Drop tile configs invalid for this shape (tile_n must divide N, tile_k
    must divide K) before benchmarking."""
    WQ = named_args.get("WQ")
    XQ = named_args.get("XQ")
    if WQ is None or XQ is None:
        return configs
    N = WQ.shape[1]
    K = XQ.shape[1]
    kept = [
        c for c in configs
        if N % c.kwargs["tile_n"] == 0 and K % c.kwargs["tile_k"] == 0
    ]
    return kept or configs


# Single autotuner for both B-layout variants, built lazily (flydsl.autotune only
# imports when FlyDSL is present). b_preshuffled is a KEY arg (not a Config kwarg)
# so the two kernels get separate tuned entries in one shared disk cache.
_AUTOTUNER = None


def _get_autotuner():
    global _AUTOTUNER
    if _AUTOTUNER is None:
        from flydsl.autotune import Config, autotune

        configs = [
            Config(tile_m=tm, tile_n=tn, tile_k=tk)
            for (tm, tn, tk) in _AUTOTUNE_TILES
        ]
        _AUTOTUNER = autotune(
            configs=configs,
            key=["m_bucket", "n", "k", "b_preshuffled"],
            prune_configs_by=_prune_tiles,
        )(_autotune_target)
    return _AUTOTUNER


def _dispatch_grouped_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
    *,
    b_preshuffled: bool,
) -> torch.Tensor:
    """Shared dispatch for both grouped ops. WQ is already in the layout the
    variant expects (MFMA-preshuffled if b_preshuffled else plain [G,N,K]).

    Tile selection follows the CUTLASS precedent: when MSLK_AUTOTUNE_ENABLE is
    set, FlyDSL autotune benchmarks the candidate tiles on a cache-miss and
    persists the winner (keyed on nextPow2(TotalM) and b_preshuffled); otherwise
    a fixed default tile is used with no benchmarking (the CI / graph-capture-safe
    path).
    """
    assert XQ.ndim == 2, f"XQ must be [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"

    output = torch.empty((TotalM, N), dtype=torch.bfloat16, device=XQ.device)
    if TotalM == 0 or N == 0 or K == 0 or G == 0:
        return output

    # The kernel reads M_sizes as int32 (buffer_load i32 idiom).
    m_sizes_i32 = M_sizes.to(torch.int32)

    if os.environ.get("MSLK_AUTOTUNE_ENABLE"):
        # FlyDSL's Autotuner discards the target's return value, so we rely on the
        # kernel writing into `output` in-place and return that buffer ourselves.
        _get_autotuner()(
            XQ, WQ, x_scale, w_scale, m_sizes_i32, output,
            _next_pow2(TotalM), N, K, b_preshuffled,
        )
        return output

    tile_m, tile_n, tile_k = _DEFAULT_TILE
    assert N % tile_n == 0, f"N={N} must be a multiple of tile_n={tile_n}"
    assert K % tile_k == 0, f"K={K} must be a multiple of tile_k={tile_k}"
    return _launch_kernel(
        XQ, WQ, x_scale, w_scale, m_sizes_i32, output,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, b_preshuffled=b_preshuffled,
    )


def matmul_f8f8bf16_groupwise_grouped_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Preshuffled-B grouped groupwise FP8 GEMM (WQ already MFMA-preshuffled)."""
    return _dispatch_grouped_gemm(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=True
    )


def matmul_f8f8bf16_groupwise_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Plain (non-preshuffled) grouped groupwise FP8 GEMM via FlyDSL.

    Same contract as the preshuffle sibling but WQ is plain row-major [G, N, K]
    (not MFMA-preshuffled) — the drop-in FlyDSL replacement for the Triton impl
    of ``mslk::f8f8bf16_groupwise_grouped``. Uses the unified kernel with
    ``b_preshuffled=False`` (B staged HBM->LDS->registers).
    """
    return _dispatch_grouped_gemm(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=False
    )


if is_flydsl_available():

    @torch.library.impl(_OP_NAME, "CUDA")
    def _f8f8bf16_groupwise_grouped_preshuffle_cuda(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        M_sizes: torch.Tensor,
    ) -> torch.Tensor:
        return matmul_f8f8bf16_groupwise_grouped_preshuffle(
            XQ, WQ, x_scale, w_scale, M_sizes
        )

    # Plain op: FlyDSL is the ROCm impl of mslk::f8f8bf16_groupwise_grouped
    # (C++ schema in gemm_ops.cpp; the Triton impl is being retired). Guard the
    # registration so it no-ops if the schema/op isn't present or already bound.
    if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
        if hasattr(torch.ops.mslk, "f8f8bf16_groupwise_grouped"):
            try:

                @torch.library.impl("mslk::f8f8bf16_groupwise_grouped", "CUDA")
                def _f8f8bf16_groupwise_grouped_cuda(
                    XQ: torch.Tensor,
                    WQ: torch.Tensor,
                    x_scale: torch.Tensor,
                    w_scale: torch.Tensor,
                    M_sizes: torch.Tensor,
                ) -> torch.Tensor:
                    return matmul_f8f8bf16_groupwise_grouped(
                        XQ, WQ, x_scale, w_scale, M_sizes
                    )

            except RuntimeError:
                pass  # already registered
