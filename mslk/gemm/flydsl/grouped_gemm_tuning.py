# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static tile-config tuning for the FlyDSL grouped groupwise FP8 GEMM ops.

Offline-tuned shape -> tile-config tables, looked up at runtime with a pure
host-side dict access (no benchmarking), so dispatch is safe under HIP graph
capture. This mirrors the CK/CUTLASS static-table precedent
(csrc/gemm/cutlass/f8f8bf16_groupwise_grouped.cu get_kernel_via_tuning), NOT
FlyDSL's runtime autotune decorator (which benchmarks on cache-miss and would
break graph capture).

Key: (nextPowerOf2(total_M), N, K, G) — identical to the CUTLASS tuning key
(total_M rounded up to the next power of two to bound the table size). The plain
and preshuffle B variants have SEPARATE tables: they favor different tiles
(preshuffle loads B straight to registers so wide tile_n is cheap; plain stages
B through LDS so wide tile_n costs more).

Values are (tile_m, tile_n, tile_k). The wrapper must use the returned tile_m
consistently for the tile-map / grid construction AND the kernel compile.

Retune with bench/gemm/flydsl_plain_config_sweep.py (or the equivalent
preshuffle sweep) on the target arch and paste the winners here.
"""

from typing import Dict, Tuple

_TileKey = Tuple[int, int, int, int]  # (pow2_total_M, N, K, G)
_TileCfg = Tuple[int, int, int]  # (tile_m, tile_n, tile_k)

# Default for shapes not in the table. Matches the CUTLASS heuristic fallback
# (f8f8bf16_groupwise_grouped_128_128_128_..., see
# csrc/gemm/cutlass/f8f8bf16_groupwise_grouped.cu get_kernel_via_heuristics):
# tile 128x128x128. It must be valid for ANY shape — tile_n=tile_k=128 divide
# every supported N/K (incl. small N=128). The wide-N tile (64,256,128) that
# wins for large N lives only in the tables, guarded by explicit N in the key.
_DEFAULT_PRESHUFFLE: _TileCfg = (128, 128, 128)
_DEFAULT_PLAIN: _TileCfg = (128, 128, 128)

# Tuned on gfx950 (MI350), G=8 DS-V3 gate/up (N2048 K7168) + down-proj
# (N7168 K2304), under CUDA-graph capture with the sync-free dispatch. Proven-safe
# tile space (tile_m in {64,128,256} x tile_n in {128,256}). See bench sweep
# bench/gemm/flydsl_retune_sweep.py, 2026-07-22.
_PRESHUFFLE_TABLE: Dict[_TileKey, _TileCfg] = {
    (1024, 2048, 7168, 8): (64, 128, 128),
    (2048, 2048, 7168, 8): (64, 128, 128),
    (4096, 2048, 7168, 8): (64, 256, 128),
    (8192, 2048, 7168, 8): (64, 256, 128),
    (16384, 2048, 7168, 8): (64, 256, 128),
    (1024, 7168, 2304, 8): (64, 128, 128),
    (2048, 7168, 2304, 8): (64, 128, 128),
    (4096, 7168, 2304, 8): (64, 256, 128),
    (8192, 7168, 2304, 8): (64, 256, 128),
    (16384, 7168, 2304, 8): (64, 256, 128),
}

_PLAIN_TABLE: Dict[_TileKey, _TileCfg] = {
    (1024, 2048, 7168, 8): (64, 128, 128),
    (2048, 2048, 7168, 8): (64, 128, 128),
    (4096, 2048, 7168, 8): (128, 128, 128),
    (8192, 2048, 7168, 8): (64, 256, 128),
    (16384, 2048, 7168, 8): (128, 128, 128),
    (1024, 7168, 2304, 8): (128, 128, 128),
    (2048, 7168, 2304, 8): (64, 256, 128),
    (4096, 7168, 2304, 8): (128, 128, 128),
    (8192, 7168, 2304, 8): (128, 128, 128),
    (16384, 7168, 2304, 8): (128, 128, 128),
}


def _next_pow2(x: int) -> int:
    """Smallest power of two >= x (matches CUTLASS nextPowerOf2). x>=1."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tile_config(
    *,
    b_preshuffled: bool,
    total_m: int,
    n: int,
    k: int,
    g: int,
) -> _TileCfg:
    """Return (tile_m, tile_n, tile_k) for this op variant + problem shape.

    Pure dict lookup on (nextPowerOf2(total_m), n, k, g); falls back to the
    variant's default tile for untuned shapes. No device work — safe to call on
    the CUDA-graph capture path.
    """
    table = _PRESHUFFLE_TABLE if b_preshuffled else _PLAIN_TABLE
    default = _DEFAULT_PRESHUFFLE if b_preshuffled else _DEFAULT_PLAIN
    key = (_next_pow2(int(total_m)), int(n), int(k), int(g))
    return table.get(key, default)
