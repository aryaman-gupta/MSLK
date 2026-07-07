# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Sample FlyDSL kernel: elementwise vector add.

Demonstrates end-to-end how a FlyDSL kernel is exposed as an MSLK op on
ROCm: a FlyDSL device kernel and host launcher, dispatched through the
shared support helpers and registered inline under ``mslk::flydsl_vector_add``.
Adapted from the FlyDSL ``examples/01-vectorAdd.py`` sample.
"""

import torch

from mslk.utils.flydsl import is_flydsl_available, run_compiled

_OP_NAME = "mslk::flydsl_vector_add"

torch.library.define(
    _OP_NAME,
    "(Tensor a, Tensor b) -> Tensor",
)


@torch.library.impl(_OP_NAME, "Meta")
def flydsl_vector_add_meta(a, b):
    return a.new_empty(a.shape)


if is_flydsl_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    @flyc.kernel
    def _vector_add_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        block_dim: fx.Constexpr[int],
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
        tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
        tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

        tA = fx.slice(tA, (None, bid))
        tB = fx.slice(tB, (None, bid))
        tC = fx.slice(tC, (None, bid))
        tA = fx.logical_divide(tA, fx.make_layout(1, 1))
        tB = fx.logical_divide(tB, fx.make_layout(1, 1))
        tC = fx.logical_divide(tC, fx.make_layout(1, 1))

        copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)

        fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

        vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
        fx.memref_store_vec(vC, rC)

        fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def _vector_add_launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C,
        n: fx.Int32,
        const_n: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        block_dim = 64
        grid_x = (n + block_dim - 1) // block_dim
        _vector_add_kernel(A, B, C, block_dim).launch(
            grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream
        )

    @torch.library.impl(_OP_NAME, "CUDA")
    def flydsl_vector_add_cuda(a, b):
        assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
        assert a.dtype == torch.float32, "sample kernel supports float32 only"
        a = a.contiguous()
        b = b.contiguous()
        c = torch.empty_like(a)
        n = a.numel()
        tA = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=4)
        run_compiled(
            _vector_add_launch, tA, b, c, n, n + 1, torch.cuda.current_stream()
        )
        return c
