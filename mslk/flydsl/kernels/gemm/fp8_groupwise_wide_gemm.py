# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0

"""FP8 groupwise-scaled GEMM built on the wide gfx950 MFMA.

Computes ``D[M, N] = dequant(A) @ dequant(B).T`` for

  A        [M, K]           FP8 E4M3
  B        [N, K]           FP8 E4M3 (already transposed by the caller)
  scale_a  [K // 128, M]    FP32, one scale per (K-group, row)
  scale_b  [K // 128, N // 128] FP32, one scale per (K-group, N-group)

which is the contract of ``mslk::f8f8bf16_groupwise``.

The schedule mirrors what the ROCm Triton kernel compiles to on gfx950, so that
the two can be compared instruction for instruction:

  * ``v_mfma_scale_f32_32x32x64_f8f6f4`` with a neutral E8M0 scale, which makes
    the hardware block-scaling a no-op and lets the block-scaled instruction
    serve this GEMM. One issue covers K=64 of a 32x32 output tile.
  * A two-dimensional wave grid: ``waves_m x waves_n`` waves each own a
    ``(tile_m / waves_m) x (tile_n / waves_n)`` slab of the output tile. LDS read
    traffic per unit work is ``waves_m / tile_m + waves_n / tile_n``, minimised
    when the wave grid is proportioned like the tile, which a one-dimensional
    split cannot do.
  * Both operands staged through LDS, since every wave needs rows and columns
    that no single lane loads.
  * The dot runs against a zero accumulator and the block scales are folded in
    afterwards, because a scale changes every 128 elements of K while the
    accumulator has to persist across the whole contraction.

Fragment layouts for the wide MFMA (verified empirically, not assumed):

  A operand   m = lane % 32,  k = (lane // 32) * 32 + byte
  B operand   n = lane % 32,  k = (lane // 32) * 32 + byte
  accumulator n = lane % 32,  m = 4 * (lane // 32) + 8 * (reg // 4) + reg % 4

so a lane's 16 accumulator values sit in one column, across four groups of four
consecutive rows.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Vector
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl.kernels.mma.mfma_preshuffle_pipeline import swizzle_xor16

# Wide MFMA geometry: one issue is a 32x32 output tile contracting 64 of K.
MFMA_M = 32
MFMA_N = 32
MFMA_K = 64
# Bytes each lane supplies per operand per issue (32 FP8 = 8 i32).
MFMA_OPERAND_BYTES = 32

WAVE = 64

# Scale-block granularity, fixed by the op's contract.
SCALE_BLOCK = 128

# E8M0 exponent bias: a scale of 2^0, so the instruction's block scaling is the
# identity and the FP32 scales can be applied in software instead.
NEUTRAL_E8M0 = 0x7F7F7F7F

# Widest global load, and hence the LDS swizzle granularity.
LOAD_BYTES = 16

# LDS per workgroup on gfx950.
LDS_CAPACITY = 160 * 1024

# LDS buffers per operand. Triton runs three, fed by a direct global-to-LDS DMA
# so the depth costs only LDS. That route is closed here: `buffer_load_lds`
# writes LDS opaquely, so SIInsertWaitcnts cannot prove the pending transfers do
# not alias the fragment reads and drains them with a vmcnt(0) before every
# ds_read -- an explicit vmcnt(9) comes back out of the assembler as vmcnt(0).
# The instruction mix still matches Triton exactly; only the wait operands
# differ, and it measured 20% slower. Staging through registers instead lets the
# compiler pipeline on register dependences it can see, but then each extra
# stage costs a whole tile of VGPRs as well as the LDS, which is why two wins.
STAGES = 2


@functools.lru_cache(maxsize=64)
def compile_groupwise_wide_gemm(
    *,
    n: int,
    k: int,
    tile_m: int = 256,
    tile_n: int = 128,
    tile_k: int = 128,
    waves_m: int = 4,
    waves_n: int = 2,
    waves_per_eu: int | None = None,
):
    """Compile the kernel for one shape and tile config; return the launcher.

    ``waves_m``/``waves_n`` are the wave grid, so the block is
    ``waves_m * waves_n * 64`` threads. Triton reaches the same place by picking
    ``num_warps`` and letting the compiler choose the grid; here it is explicit,
    since the grid shape drives LDS read traffic and cannot be left implicit.
    """
    if tile_k != SCALE_BLOCK:
        raise ValueError(
            f"tile_k ({tile_k}) must equal the scale block ({SCALE_BLOCK}): the "
            "scales change every 128 elements of K, and a tile spanning more "
            "than one block would need a fold per sub-block"
        )
    if k % tile_k or n % tile_n:
        raise ValueError(
            f"n ({n}) and k ({k}) must divide by tile_n ({tile_n}) and tile_k "
            f"({tile_k}); the tail-masked variant is not built yet"
        )
    if n % SCALE_BLOCK:
        raise ValueError(f"n ({n}) must be a multiple of {SCALE_BLOCK}")
    if tile_m % (waves_m * MFMA_M) or tile_n % (waves_n * MFMA_N):
        raise ValueError(
            f"tile {tile_m}x{tile_n} does not divide into {waves_m}x{waves_n} "
            f"waves of {MFMA_M}x{MFMA_N} MFMA tiles"
        )
    if tile_k % MFMA_K:
        raise ValueError(f"tile_k ({tile_k}) must be a multiple of {MFMA_K}")
    if tile_n > SCALE_BLOCK:
        raise ValueError(
            f"tile_n ({tile_n}) must not exceed the scale block ({SCALE_BLOCK}), "
            "or a tile would span several B scales"
        )

    num_waves = waves_m * waves_n
    total_threads = num_waves * WAVE
    wave_tile_m = tile_m // waves_m
    wave_tile_n = tile_n // waves_n
    # MFMA tiles each wave owns, and hence its accumulator count.
    acc_m = wave_tile_m // MFMA_M
    acc_n = wave_tile_n // MFMA_N
    k_steps = tile_k // MFMA_K
    num_k_tiles = k // tile_k
    scale_n = n // SCALE_BLOCK

    # Each thread's share of a tile, in whole 16-byte loads.
    a_bytes_per_thread = tile_m * tile_k // total_threads
    b_bytes_per_thread = tile_n * tile_k // total_threads
    if a_bytes_per_thread % LOAD_BYTES or b_bytes_per_thread % LOAD_BYTES:
        raise ValueError(
            f"tile {tile_m}x{tile_n}x{tile_k} does not split into whole "
            f"{LOAD_BYTES}-byte loads across {total_threads} threads"
        )
    num_a_loads = a_bytes_per_thread // LOAD_BYTES
    num_b_loads = b_bytes_per_thread // LOAD_BYTES

    # 16-byte chunks per row, the modulus of the XOR swizzle.
    k_chunks = tile_k // LOAD_BYTES
    # Accumulator registers per MFMA tile.
    acc_regs = MFMA_M * MFMA_N // WAVE
    # 16-byte reads per operand fragment.
    frag_loads = MFMA_OPERAND_BYTES // LOAD_BYTES

    gpu_arch = get_hip_arch()
    if not str(gpu_arch).startswith("gfx95"):
        raise ValueError(
            f"the wide f8f6f4 MFMA is gfx950-only; arch is {gpu_arch}"
        )

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_gw_wide")
    lds_a_elems = tile_m * tile_k
    lds_b_elems = tile_n * tile_k
    # A tile is written into one buffer while the other is being read, which is
    # what lets a single barrier per K tile suffice.
    lds_bytes = STAGES * (lds_a_elems + lds_b_elems)
    if lds_bytes > LDS_CAPACITY:
        raise ValueError(
            f"tile {tile_m}x{tile_n}x{tile_k} double-buffered needs "
            f"{lds_bytes} B of LDS, over the {LDS_CAPACITY} B budget"
        )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

    module_name = (
        f"gw_wide_n{n}_k{k}_t{tile_m}x{tile_n}x{tile_k}_w{waves_m}x{waves_n}"
    )

    # The AMDGPU default caps a workgroup at 256 threads; an eight-wave block
    # needs the larger bound declared up front.
    @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
    def groupwise_wide_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        # MLIR types need a live context, so they are built during tracing.
        v_acc_ty = Vector.make_type(acc_regs, fx.Float32)

        m_in = fx.Index(i32_m)
        n_in = fx.Index(i32_n)
        k_in = fx.Index(i32_k)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block
        bx = gpu.block_id("y")  # M-tile

        bx_m = bx * fx.Index(tile_m)
        by_n = by * fx.Index(tile_n)

        # Wave grid: wave w owns rows [wm * wave_tile_m, +wave_tile_m) and
        # columns [wn * wave_tile_n, +wave_tile_n).
        wave_id = tx // fx.Index(WAVE)
        lane = tx % fx.Index(WAVE)
        wm = wave_id // fx.Index(waves_n)
        wn = wave_id % fx.Index(waves_n)
        # The wide MFMA splits a lane's role by half-wave: which 32 of K it
        # supplies, and which four of the 32 output rows it holds.
        lane_lo = lane % fx.Index(32)
        lane_hi = lane // fx.Index(32)

        wave_m0 = wm * fx.Index(wave_tile_m)
        wave_n0 = wn * fx.Index(wave_tile_n)

        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=m_in * k_in
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=n_in * k_in
        )
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=m_in * n_in * fx.Index(2)
        )
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a,
            max_size=False,
            num_records_bytes=m_in * fx.Index(num_k_tiles * 4),
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b,
            max_size=False,
            num_records_bytes=fx.Index(scale_n * num_k_tiles * 4),
        )

        base_ptr = allocator.get_base()
        # One arena: the A buffers, then the B buffers. Both operands are
        # row-major over K with the same stride, so one indexing helper serves
        # both.
        lds_a = SmemPtr(base_ptr, lds_off, T.f8, shape=(lds_bytes,)).get()

        def a_buf(slot):
            return slot * fx.Index(lds_a_elems)

        def b_buf(slot):
            return fx.Index(STAGES * lds_a_elems) + slot * fx.Index(lds_b_elems)
        c_chunks = fx.Index(k_chunks)
        c4 = fx.Index(4)

        # ---- staging coordinates -------------------------------------------
        # Load i of thread tx covers tile bytes [(i * threads + tx) * 16, +16),
        # so consecutive lanes walk a row and the reads coalesce.
        def stage_coords(num_loads):
            coords = []
            for i in range_constexpr(num_loads):
                linear = (fx.Index(i * total_threads) + tx) * fx.Index(LOAD_BYTES)
                coords.append((linear // fx.Index(tile_k), linear % fx.Index(tile_k)))
            return coords

        a_coords = stage_coords(num_a_loads)
        b_coords = stage_coords(num_b_loads)

        k_dwords = k_in // c4
        tile_k_dwords = fx.Index(tile_k // 4)

        def load_tile(rsrc, coords, row_base, kt):
            """Read this thread's share of a tile from global into registers.

            Reads past the last tile fall outside the buffer descriptor and come
            back as zero, so the K tail needs no branch.
            """
            parts = []
            for i in range_constexpr(len(coords)):
                row, col = coords[i]
                idx_dw = (row_base + row) * k_dwords + kt * tile_k_dwords + col // c4
                parts.append(
                    Vector(
                        buffer_ops.buffer_load(rsrc, idx_dw, vec_width=4, dtype=T.i32)
                    ).bitcast(fx.Int32)
                )
            return parts

        def lds_index(row, col, lds_base):
            """Byte offset of tile element ``(row, col)`` under the XOR16 swizzle.

            Row-major LDS would start every row in bank 0, so a fragment read --
            which is exactly "each lane takes a different row" -- would serialise
            32 ways. XORing the chunk index with the row spreads them out.
            """
            return row * fx.Index(tile_k) + swizzle_xor16(row, col, c_chunks) + lds_base

        def store_tile(parts, coords, lds_base):
            for i in range_constexpr(len(coords)):
                row, col = coords[i]
                idx = lds_index(row, col, lds_base)
                vector.store(vector.bitcast(T.f8x16, parts[i]), lds_a, [idx])

        # ---- fragment reads --------------------------------------------------
        def read_frag(row, lds_base, ks):
            """Gather one lane's 32-byte operand fragment for MFMA step ``ks``."""
            halves = []
            for j in range_constexpr(frag_loads):
                col = (
                    fx.Index(ks * MFMA_K)
                    + lane_hi * fx.Index(32)
                    + fx.Index(j * LOAD_BYTES)
                )
                idx = lds_index(row, col, lds_base)
                halves.append(Vector(Vector.load(T.f8x16, lds_a, [idx])).bitcast(fx.Int32))
            return Vector(halves[0]).shuffle(Vector(halves[1]), list(range(8))).ir_value()

        def mfma(a_op, b_op, c_in):
            return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                v_acc_ty,
                _raw(a_op),
                _raw(b_op),
                _raw(c_in),
                0,
                0,
                0,
                _raw(fx.Int32(NEUTRAL_E8M0)),
                0,
                _raw(fx.Int32(NEUTRAL_E8M0)),
            ).result

        zero_acc = Vector.from_elements(
            [fx.Float32(0.0) for _ in range_constexpr(acc_regs)], fx.Float32
        ).ir_value()

        def tile_product(a_base, b_base):
            """Contract one K tile into a fresh accumulator.

            The dot starts from zero rather than the running accumulator because
            the scales apply per K tile; folding them afterwards is what lets a
            single accumulator span the whole of K.
            """
            tiles = [zero_acc for _ in range_constexpr(acc_m * acc_n)]
            for ks in range_constexpr(k_steps):
                a_frags = []
                for ai in range_constexpr(acc_m):
                    row = wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
                    a_frags.append(read_frag(row, a_base, ks))
                b_frags = []
                for aj in range_constexpr(acc_n):
                    row = wave_n0 + fx.Index(aj * MFMA_N) + lane_lo
                    b_frags.append(read_frag(row, b_base, ks))
                for ai in range_constexpr(acc_m):
                    for aj in range_constexpr(acc_n):
                        idx = ai * acc_n + aj
                        # Operands swapped on purpose. The hardware puts the
                        # second operand's index on lane % 32 and the first
                        # one's across the registers; feeding B first therefore
                        # transposes the result, so a lane ends up holding one
                        # output row and four runs of four adjacent columns.
                        # That buys a vectorised store, and it makes the block
                        # scale constant across a lane's whole accumulator.
                        tiles[idx] = mfma(b_frags[aj], a_frags[ai], tiles[idx])
            return tiles

        def load_scales(kt):
            """The combined scale for each of this lane's accumulators.

            With the transposed accumulator a lane owns exactly one output row
            per M subtile, so its A scale is a single value rather than one per
            register. Every column of the tile falls in one N scale block, so B
            contributes one scalar for the whole tile. The product is therefore
            constant across an accumulator, and the fold is 16 FMAs against a
            broadcast rather than 16 separate multipliers.
            """
            sb_idx = kt * fx.Index(scale_n) + (by_n // fx.Index(SCALE_BLOCK))
            sb = fx.Float32(
                buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.f32)
            )
            out = []
            for ai in range_constexpr(acc_m):
                row = bx_m + wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
                sa = fx.Float32(
                    buffer_ops.buffer_load(
                        sa_rsrc, kt * m_in + row, vec_width=1, dtype=T.f32
                    )
                )
                out.append(sa * sb)
            return out

        # ---- K loop ----------------------------------------------------------
        # Software pipeline, STAGES - 1 tiles deep, with nothing staged in
        # registers: each iteration waits for the tile whose DMA was issued
        # STAGES - 1 iterations ago, barriers, then issues the DMA for the tile
        # STAGES - 1 ahead and computes.
        #
        # The one barrier does double duty. It publishes the tile just waited
        # for, and it separates the reads of slot (kt - 1) % STAGES -- which the
        # DMA below is about to overwrite -- from that overwrite, because those
        # reads happened before it in every wave's program order.
        #
        # The scale loads are issued *before* the DMA rather than with the fold
        # that consumes them. vmcnt retires in order, so consuming a scale
        # loaded in the same iteration would force a wait that also drained
        # every DMA behind it, collapsing the pipeline to nothing. Loading them
        # two tiles ahead puts them behind the DMAs in the queue instead.
        n_acc = acc_m * acc_n

        def split_state(state):
            vals = list(state) if isinstance(state, (list, tuple)) else [state]
            return (
                [Vector(v, (acc_regs,), fx.Float32) for v in vals[:n_acc]],
                [Vector(v) for v in vals[n_acc : n_acc + num_a_loads]],
                [Vector(v) for v in vals[n_acc + num_a_loads :]],
            )

        def fold(cur, tiles, scales):
            out = []
            for ai in range_constexpr(acc_m):
                for aj in range_constexpr(acc_n):
                    i = ai * acc_n + aj
                    tile = Vector(tiles[i], (acc_regs,), fx.Float32)
                    # Emitted as an explicit fused multiply-add. Written as
                    # `acc + tile * scale` the two arith ops do not contract --
                    # contraction changes rounding, so the compiler will not do
                    # it unasked -- and the fold costs a separate v_pk_mul_f32
                    # and v_pk_add_f32 per pair instead of one v_pk_fma_f32.
                    vals = [
                        fx.Float32(
                            math_dialect.fma(
                                fx.Float32(tile[r]).ir_value(),
                                scales[ai].ir_value(),
                                fx.Float32(cur[i][r]).ir_value(),
                            )
                        )
                        for r in range_constexpr(acc_regs)
                    ]
                    out.append(Vector.from_elements(vals, fx.Float32))
            return out

        accs = [Vector(zero_acc, (acc_regs,), fx.Float32) for _ in range_constexpr(n_acc)]

        a_pre = load_tile(a_rsrc, a_coords, bx_m, fx.Index(0))
        b_pre = load_tile(b_rsrc, b_coords, by_n, fx.Index(0))

        for _kt, _st in fx.range(
            0, num_k_tiles, 1, init=list(accs) + list(a_pre) + list(b_pre)
        ):
            _cur, _a_cur, _b_cur = split_state(_st)
            _now = _kt % fx.Index(STAGES)
            _a_now, _b_now = a_buf(_now), b_buf(_now)

            store_tile(_a_cur, a_coords, _a_now)
            store_tile(_b_cur, b_coords, _b_now)
            gpu.barrier()

            # Issued after the barrier so they are in flight underneath this
            # tile's MFMAs; the compiler pipelines them on register dependences
            # it can see, which is the thing the opaque DMA denied it.
            _a_next = load_tile(a_rsrc, a_coords, bx_m, _kt + fx.Index(1))
            _b_next = load_tile(b_rsrc, b_coords, by_n, _kt + fx.Index(1))

            _tiles = tile_product(_a_now, _b_now)
            _res = yield fold(
                _cur, _tiles, load_scales(_kt)
            ) + list(_a_next) + list(_b_next)

        accs, _, _ = split_state(_res)

        # ---- epilogue --------------------------------------------------------
        # The transposed accumulator gives a lane one output row and four runs of
        # four adjacent columns, so each run converts to four BF16 and leaves as
        # one 8-byte store. No LDS round trip and no cross-lane shuffle: the
        # layout the MFMA produced is already the layout the store wants.
        for ai in range_constexpr(acc_m):
            row = bx_m + wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
            for aj in range_constexpr(acc_n):
                acc = accs[ai * acc_n + aj]
                for g in range_constexpr(acc_regs // 4):
                    col = (
                        by_n
                        + wave_n0
                        + fx.Index(aj * MFMA_N)
                        + lane_hi * c4
                        + fx.Index(g * 8)
                    )
                    quad = Vector.from_elements(
                        [
                            fx.BFloat16(arith.trunc_f(T.bf16, fx.Float32(acc[g * 4 + e])))
                            for e in range_constexpr(4)
                        ],
                        fx.BFloat16,
                    )
                    buffer_ops.buffer_store(
                        quad.bitcast(fx.Int32),
                        d_rsrc,
                        (row * n_in + col) * fx.Index(2),
                        offset_is_bytes=True,
                    )

    @flyc.jit
    def launch_groupwise_wide_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        m_in = fx.Index(i32_m)
        gx = fx.Index(i32_n) // fx.Index(tile_n)
        gy = (m_in + fx.Index(tile_m - 1)) // fx.Index(tile_m)

        launcher = groupwise_wide_gemm_kernel(
            arg_d, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n, i32_k
        )
        if waves_per_eu is not None and int(waves_per_eu) >= 1:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )
        launcher.launch(
            grid=(gx, gy, fx.Index(1)), block=(total_threads, 1, 1), stream=stream
        )

    return launch_groupwise_wide_gemm
