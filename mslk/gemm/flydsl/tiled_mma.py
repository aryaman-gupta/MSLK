# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Sample tiled MFMA GEMM demonstrating AOT pre-compilation of a FlyDSL kernel.

A single-block tiled matmul (C = A @ B.T, float32) whose block tile is a
compile-time knob. It shows the AOT wiring a real kernel provides:

  - ``AOT_CONFIGS`` / ``AOT_ARCHS``: the (tile, arch) set to pre-compile.
  - ``compile_aot_config(config, arch)``: compile one config with no GPU
    tensors, so the AOT harness can populate the cache off-device.

Adapted from the FlyDSL ``examples/03-tiledMma.py`` sample.
"""

import functools
import os

import torch

from mslk.utils.flydsl import is_flydsl_available, run_compiled

_OP_NAME = "mslk::flydsl_tiled_mma"

# Tile configurations to pre-compile (AOT). Each is a compile-cache key.
AOT_CONFIGS = [
    {"block_m": 64, "block_n": 64, "block_k": 8},
    {"block_m": 32, "block_n": 32, "block_k": 8},
]
AOT_ARCHS = ["gfx942", "gfx950"]

torch.library.define(
    _OP_NAME,
    "(Tensor a, Tensor b) -> Tensor",
)


@torch.library.impl(_OP_NAME, "Meta")
def flydsl_tiled_mma_meta(a, b):
    return a.new_empty((a.shape[0], b.shape[0]))


if is_flydsl_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    @functools.lru_cache(maxsize=None)
    def _compile(block_m: int, block_n: int, block_k: int):
        @flyc.kernel
        def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x

            A = fx.rocdl.make_buffer_tensor(A)
            B = fx.rocdl.make_buffer_tensor(B)
            C = fx.rocdl.make_buffer_tensor(C)

            bA = fx.slice(fx.zipped_divide(A, (block_m, block_k)), (None, bid))
            bB = fx.slice(fx.zipped_divide(B, (block_n, block_k)), (None, bid))
            bC = fx.slice(fx.zipped_divide(C, (block_m, block_n)), (None, bid))

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
            tiled_mma = fx.make_tiled_mma(
                mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0))
            )
            thr_mma = tiled_mma.thr_slice(tid)

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            thr_copy_a = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
            thr_copy_b = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
            thr_copy_c = fx.make_tiled_copy_C(copy_atom, tiled_mma).get_slice(tid)

            frag_a = thr_mma.make_fragment_A(bA)
            frag_b = thr_mma.make_fragment_B(bB)
            frag_c = thr_mma.make_fragment_C(bC)

            fx.copy(
                copy_atom, thr_copy_a.partition_S(bA), thr_copy_a.retile(frag_a),
                pred=None,
            )
            fx.copy(
                copy_atom, thr_copy_b.partition_S(bB), thr_copy_b.retile(frag_b),
                pred=None,
            )

            frag_c.fill(0)
            fx.gemm(mma_atom, frag_c, frag_a, frag_b, frag_c)

            fx.copy(
                copy_atom, thr_copy_c.retile(frag_c), thr_copy_c.partition_S(bC),
                pred=None,
            )

        @flyc.jit
        def launch(
            A: fx.Tensor,
            B: fx.Tensor,
            C: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            gemm_kernel(A, B, C).launch(
                grid=(1, 1, 1), block=(256, 1, 1), stream=stream
            )

        return launch

    def compile_aot_config(config, arch):
        """Compile one config for one arch without GPU tensors (AOT)."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        block_m, block_n, block_k = (
            config["block_m"],
            config["block_n"],
            config["block_k"],
        )
        launcher = _compile(block_m, block_n, block_k)
        prev_arch = os.environ.get("FLYDSL_GPU_ARCH")
        os.environ["FLYDSL_GPU_ARCH"] = arch
        try:
            with FakeTensorMode():
                a = torch.empty(block_m, block_k, dtype=torch.float32, device="cuda")
                b = torch.empty(block_n, block_k, dtype=torch.float32, device="cuda")
                c = torch.empty(block_m, block_n, dtype=torch.float32, device="cuda")
                flyc.compile(launcher, a, b, c)
        finally:
            if prev_arch is None:
                os.environ.pop("FLYDSL_GPU_ARCH", None)
            else:
                os.environ["FLYDSL_GPU_ARCH"] = prev_arch

    @torch.library.impl(_OP_NAME, "CUDA")
    def flydsl_tiled_mma_cuda(a, b):
        assert a.dtype == torch.float32, "sample kernel supports float32 only"
        block_m, block_k = a.shape
        block_n = b.shape[0]
        a = a.contiguous()
        b = b.contiguous()
        c = torch.empty((block_m, block_n), dtype=torch.float32, device=a.device)
        launcher = _compile(block_m, block_n, block_k)
        run_compiled(launcher, a, b, c, torch.cuda.current_stream())
        return c
