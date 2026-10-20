# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tune which backend serves ``mslk::f4f4bf16`` on ROCm, per shape.

For every (N, K) in the shape list and every M bucket, runs the op once per
backend (MSLK's Triton kernel, aiter's ``gemm_afp4wfp4``), checks each output,
times it, and writes ``mslk/gemm/aiter/tables/f4f4bf16.csv`` by the rules in
:mod:`mslk.aiter.tuning`. Needs the pinned aiter installed and a gfx950 GPU.

Correctness follows ``MXFP4Tests.test_gemm``: the same inputs and tolerance
(atol = rtol = 8e-2). The reference is the product of the dequantized inputs
rather than of the original ones, so that the check measures the kernel and
not FP4 quantization, whose error grows with K beyond what the test's shapes
reach.

One GPU per shard, then merge:

    for i in 0 1 2 3; do
      HIP_VISIBLE_DEVICES=$i python bench/aiter/tune_f4f4bf16.py --shard $i/4 --out part$i.csv &
    done; wait
    python -m mslk.aiter.tuning merge mslk/gemm/aiter/tables/f4f4bf16.csv part*.csv
"""

import sys
from typing import Dict, List

import click
import mslk.gemm  # noqa: F401  (registers mslk::f4f4bf16)
import torch
from mslk.aiter.routing import force_backend, MSLK
from mslk.aiter.tuning import (
    choose,
    device_columns,
    measure,
    read_shapes,
    shard,
    write_table,
)
from mslk.gemm.aiter.f4f4bf16 import AITER, BACKENDS, OP
from mslk.quantize.triton.fp4_quantize import triton_quantize_mx4_unpack

M_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
ATOL = RTOL = 8e-2  # MXFP4Tests.test_gemm
COLUMNS = ("gfx", "cu_num", "N", "K", "M_max", "backend", "mslk_us", "aiter_us")

# E2M1 values by 4-bit code; codes 8-15 are the negatives of 0-7.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1 = _E2M1 + tuple(-v for v in _E2M1)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed MXFP4 (low nibble first) and E8M0 scales -> float32."""
    q = q.view(torch.uint8)
    lut = torch.tensor(_E2M1, device=q.device)
    values = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], dim=-1)
    values = values.flatten(-2)
    scale = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return values * scale.repeat_interleave(32, dim=-1)


@click.command()
@click.option("--shapes", default="bench/aiter/shapes/f4f4bf16.csv", show_default=True)
@click.option("--out", default="mslk/gemm/aiter/tables/f4f4bf16.csv", show_default=True)
@click.option(
    "--shard", "shard_spec", default="0/1", help="i/n: tune every n-th (N, K) from i."
)
def main(shapes: str, out: str, shard_spec: str) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    rows: List[Dict[str, str]] = []
    mslk_failures: List[str] = []
    for shape in shard(read_shapes(shapes), shard_spec):
        N, K = int(shape["N"]), int(shape["K"])
        B = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.01
        bq, b_scale = triton_quantize_mx4_unpack(B)
        b_ref = dequantize(bq, b_scale)
        for i, M in enumerate(M_BUCKETS):
            A = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
            aq, a_scale = triton_quantize_mx4_unpack(A)
            ref = dequantize(aq, a_scale) @ b_ref.T
            out_buf = torch.empty(M, N, device=device, dtype=torch.bfloat16)

            def check(result: torch.Tensor) -> tuple[bool, float]:
                err = (result.float() - ref).abs()
                return bool((err <= ATOL + RTOL * ref.abs()).all()), err.max().item()

            measurements = []
            # Alternate which backend runs first, so neither always inherits
            # the other's warm-up.
            order = BACKENDS if i % 2 == 0 else BACKENDS[::-1]
            for backend in order:
                with force_backend(OP, backend):
                    measurements.append(
                        measure(
                            backend,
                            lambda: torch.ops.mslk.f4f4bf16(
                                aq, bq, a_scale, b_scale, output=out_buf
                            ),
                            check,
                        )
                    )
            m = {x.backend: x for x in measurements}
            if not m[MSLK].correct:
                mslk_failures.append(
                    f"M={M} N={N} K={K}: "
                    + (m[MSLK].error or f"max err {m[MSLK].max_abs_err:.3g}")
                )
            backend = choose(measurements) if m[MSLK].correct else MSLK
            note = (
                ""
                if m[AITER].correct
                else f"  (aiter excluded: {m[AITER].error or 'incorrect'})"
            )
            print(
                f"N={N:>6} K={K:>6} M={M:>5} | mslk {m[MSLK].us:8.1f} us  "
                f"aiter {m[AITER].us:8.1f} us -> {backend}{note}",
                flush=True,
            )
            rows.append(
                {
                    **device_columns(),
                    "N": str(N),
                    "K": str(K),
                    "M_max": str(M),
                    "backend": backend,
                    "mslk_us": f"{m[MSLK].us:.1f}",
                    "aiter_us": f"{m[AITER].us:.1f}",
                }
            )
    if mslk_failures:
        print("MSLK's kernel failed the correctness check; fix it before tuning:")
        print("\n".join(f"  {f}" for f in mslk_failures))
        sys.exit(1)
    write_table(out, rows, COLUMNS)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
