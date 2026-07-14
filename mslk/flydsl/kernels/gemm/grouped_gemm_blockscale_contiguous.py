# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Contiguous Grouped FP8 GEMM kernel with block scaling.

Groups are concatenated along M with arbitrary (not tile-aligned) per-group row
counts, and the output is compact [M_total, N]. Each output M-tile belongs to
exactly one group; work is dispatched over a precomputed per-tile map rather
than a per-row group id, so a tile never spans a group boundary.

Scales are FP32 (software scaling) on all architectures.

Tensors:
  - A: [M_total, K] FP8 - concatenated rows from all groups
  - scale_a: [scale_k, M_total] FP32 - per-token, per-128K scales (transposed)
  - B: [num_groups, N, K] FP8 - one weight matrix per group, preshuffled
  - scale_b: [num_groups, scale_k, scale_n] FP32 - per-block scales
  - D: [M_total, N] BF16 - output

Per-tile dispatch metadata (length = number of M-tiles; index by M-tile id):
  - tile_group: INT32 - group id owning the tile (-1 marks a surplus/no-op tile)
  - tile_row_start: INT32 - global row (into A/scale_a/D) of the tile's first row
  - tile_row_limit: INT32 - exclusive global row end of the tile's group; rows at
    or beyond it are the partial-tile tail and are masked out of the store

Block scaling granularity:
  - A: (1, 128) - per-token, per-128-K-elements
  - B: (128, 128) - per-128-N, per-128-K block
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_common import (
    compute_compile_constants,
    compute_mfma_tiling,
    init_accumulators,
    make_a_tile_loaders,
    make_b_loader,
    make_compute_tile,
    make_epilogue_writers,
    make_hot_loop_scheduler,
    make_lds_loader,
    make_n_block_coords,
    make_pingpong_kloop,
    make_prefetch_scales,
    out_mlir_for,
    setup_lds_allocation,
    validate_params,
)
from mslk.flydsl.kernels.mma.mfma_epilogues import mfma_epilog


@functools.lru_cache(maxsize=128)
def compile_grouped_gemm_blockscale_contiguous(
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
    """Compile grouped FP8 GEMM kernel and return the JIT launcher.

    Args:
        n: N dimension (output columns per group)
        k: K dimension (reduction dimension)
        num_groups: Number of groups (experts)
        tile_m: M tile size (default 128)
        tile_n: N tile size (default 128)
        tile_k: K tile size (default 128)
        scale_block_k: K-dimension scale block size (default 128)
        scale_block_n: N-dimension scale block size (default 128)
        out_dtype: Output data type ("bf16" or "f16")

    Returns:
        JIT launcher function.
    """
    gpu_arch = get_hip_arch()
    # This FP8 kernel always uses the FP32 software-scaling path; the shared
    # helpers' hardware E8M0 microscaling path is not used here.
    _use_hw_scale = False

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_grouped_gemm")

    validate_params(
        n=n,
        k=k,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    out_mlir = out_mlir_for(out_dtype)

    _c = compute_compile_constants(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
    )
    total_threads = _c.total_threads
    elem_bytes = _c.elem_bytes
    num_k_tiles = _c.num_k_tiles
    scale_k = _c.scale_k
    scale_n = _c.scale_n
    sb_per_tile = _c.sb_per_tile
    k_unroll = _c.k_unroll
    kpack_bytes = _c.kpack_bytes
    tile_k_bytes = _c.tile_k_bytes
    tile_k_dwords = _c.tile_k_dwords
    chunk_i32_a = _c.chunk_i32_a
    num_a_loads = _c.num_a_loads

    lds_alloc_offset, lds_tile_elems = setup_lds_allocation(
        allocator=allocator,
        tile_m=tile_m,
        tile_k=tile_k,
        tile_n=tile_n,
        elem_bytes=elem_bytes,
    )

    # Module name for caching
    module_name = (
        f"grouped_gemm_blockscale_contiguous_{out_dtype}"
        f"_n{n}_k{k}_g{num_groups}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_pingpong"
    ).replace("-", "_")

    @flyc.kernel(name=module_name)
    def grouped_gemm_blockscale_contiguous_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_tile_group: fx.Tensor,
        arg_tile_row_start: fx.Tensor,
        arg_tile_row_limit: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
        i32_num_m_tiles: fx.Int32,
    ):
        # Convert runtime parameters to index type
        m_in = fx.Index(i32_m)
        n_in = fx.Index(i32_n)
        k_in = fx.Index(i32_k)
        num_groups_in = fx.Index(i32_num_groups)

        # Thread and block IDs
        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block index
        bx = gpu.block_id("y")  # M-tile index (into the per-tile dispatch map)

        # N-block position; bx_m (global row base) is loaded from the tile map below.
        by_n = by * fx.Index(tile_n)

        # Wave/lane decomposition (256 threads = 4 waves x 64 lanes)
        layout_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(fx.Int32(tx), layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        # Lane decomposition for MFMA (lane_id -> lane_div_16, lane_mod_16)
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # LDS setup: single memref for both ping-pong buffers
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(base_ptr, lds_alloc_offset, T.f8, shape=(2 * tile_m * tile_k,)).get()
        lds_stride = tile_k
        layout_lds = fx.make_layout((tile_m, tile_k), stride=(lds_stride, 1))
        lds_base_pong = fx.Index(0)
        lds_base_ping = fx.Index(lds_tile_elems)

        # CShuffle epilogue LDS (aliased from same base, bf16 element type)
        lds_out = SmemPtr(base_ptr, lds_alloc_offset, out_mlir(), shape=(tile_m * tile_n,)).get()

        # Buffer resources
        a_nbytes = m_in * k_in
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False, num_records_bytes=a_nbytes)

        b_nbytes = num_groups_in * n_in * k_in
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=False, num_records_bytes=b_nbytes)

        d_nbytes = m_in * n_in * fx.Index(2)  # bf16/f16 = 2 bytes
        d_rsrc = buffer_ops.create_buffer_resource(arg_d, max_size=False, num_records_bytes=d_nbytes)

        # Scale buffers — gfx950 HW E8M0 path consumes int8 (one byte/scale,
        # pre-packed on host); gfx942 SW path consumes f32.
        scale_byte_size = 1 if _use_hw_scale else 4

        # scale_a: [scale_k, M] - transposed layout
        sa_nbytes = fx.Index(scale_k) * m_in * fx.Index(scale_byte_size)
        sa_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False, num_records_bytes=sa_nbytes)

        # scale_b: [num_groups, scale_n, scale_k]
        sb_nbytes = num_groups_in * fx.Index(scale_n * scale_k * scale_byte_size)
        sb_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=False, num_records_bytes=sb_nbytes)

        # Per-tile dispatch map: one int32 entry per M-tile.
        num_m_tiles_in = fx.Index(i32_num_m_tiles)
        tm_nbytes = num_m_tiles_in * fx.Index(4)
        tg_rsrc = buffer_ops.create_buffer_resource(arg_tile_group, max_size=False, num_records_bytes=tm_nbytes)
        trs_rsrc = buffer_ops.create_buffer_resource(arg_tile_row_start, max_size=False, num_records_bytes=tm_nbytes)
        trl_rsrc = buffer_ops.create_buffer_resource(arg_tile_row_limit, max_size=False, num_records_bytes=tm_nbytes)

        # Group id for this M-tile; -1 marks a surplus tile (grid may be launched
        # to a host-known upper bound that exceeds the actual tile count).
        group_id_i32 = buffer_ops.buffer_load(tg_rsrc, bx, vec_width=1, dtype=T.i32)
        is_valid = arith.cmpi(arith.CmpIPredicate.sge, group_id_i32, fx.Int32(0))

        # Early exit for surplus/no-op tiles.
        if is_valid:
            group_idx = fx.Index(group_id_i32)

            # Global row base of this tile and the exclusive row end of its group
            # (the group end masks the partial-tile tail in the epilogue store).
            row_start_i32 = buffer_ops.buffer_load(trs_rsrc, bx, vec_width=1, dtype=T.i32)
            row_limit_i32 = buffer_ops.buffer_load(trl_rsrc, bx, vec_width=1, dtype=T.i32)
            bx_m = fx.Index(row_start_i32)

            _t = compute_mfma_tiling(tile_m=tile_m, tile_n=tile_n)
            m_repeat = _t.m_repeat
            n_per_wave = _t.n_per_wave
            num_acc_n = _t.num_acc_n

            acc_init, accs = init_accumulators(_t.num_accs)

            _nb = make_n_block_coords(
                wave_id=wave_id,
                by_n=by_n,
                group_idx=group_idx,
                num_groups_in=num_groups_in,
                n_in=n_in,
                k_in=k_in,
                lane_mod_16=lane_mod_16,
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                scale_block_n=scale_block_n,
                scale_k=scale_k,
                n_per_wave=n_per_wave,
                num_acc_n=num_acc_n,
            )
            n_tile_base = _nb.n_tile_base
            n_block_for_scale = _nb.n_block_for_scale
            layout_b = _nb.layout_b
            n_blk_list = _nb.n_blk_list
            n_intra_list = _nb.n_intra_list
            c_scale_k = _nb.c_scale_k

            prefetch_a_tile, store_a_tile_to_lds, a_row_local, a_col_local_i32, k_blocks16 = make_a_tile_loaders(
                a_rsrc=a_rsrc,
                lds_a=lds_a,
                layout_lds=layout_lds,
                bx_m=bx_m,
                tx=tx,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_k_bytes=tile_k_bytes,
                tile_k_dwords=tile_k_dwords,
                chunk_i32_a=chunk_i32_a,
                num_a_loads=num_a_loads,
                total_threads=total_threads,
                elem_bytes=elem_bytes,
                k_in=k_in,
            )

            lds_load_packs_k64 = make_lds_loader(
                lds_a=lds_a,
                layout_lds=layout_lds,
                k_blocks16=k_blocks16,
            )

            load_b_tile = make_b_loader(
                arg_b=arg_b,
                b_rsrc=b_rsrc,
                layout_b=layout_b,
                n_blk_list=n_blk_list,
                n_intra_list=n_intra_list,
                lane_div_16=lane_div_16,
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                k_unroll=k_unroll,
                num_acc_n=num_acc_n,
            )

            # Base coordinates for A0 prefetch (mi=0, ku=0)
            row_a_lds_base = lane_mod_16  # mi=0
            col_offset_base_bytes = lane_div_16 * fx.Index(16)  # ku=0

            mfma_res_ty = T.f32x4

            ku_per_sb = scale_block_k // 64
            rocdl.sched_barrier(0)

            hot_loop_scheduler = make_hot_loop_scheduler(
                _use_hw_scale=_use_hw_scale,
                sb_per_tile=sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                k_unroll=k_unroll,
                num_a_loads=num_a_loads,
                ku_per_sb=ku_per_sb,
            )

            prefetch_scales = make_prefetch_scales(
                _use_hw_scale=_use_hw_scale,
                sa_rsrc=sa_rsrc,
                sb_rsrc=sb_rsrc,
                group_idx=group_idx,
                scale_n=scale_n,
                scale_k=scale_k,
                c_scale_k=c_scale_k,
                n_block_for_scale=n_block_for_scale,
                bx_m=bx_m,
                lane_mod_16=lane_mod_16,
                m_in=m_in,
                sb_per_tile=sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
            )

            compute_tile = make_compute_tile(
                _use_hw_scale=_use_hw_scale,
                lds_load_packs_k64=lds_load_packs_k64,
                sa_rsrc=sa_rsrc,
                sb_rsrc=sb_rsrc,
                group_idx=group_idx,
                scale_n=scale_n,
                scale_k=scale_k,
                c_scale_k=c_scale_k,
                n_block_for_scale=n_block_for_scale,
                bx_m=bx_m,
                lane_mod_16=lane_mod_16,
                lane_div_16=lane_div_16,
                m_in=m_in,
                sb_per_tile=sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                ku_per_sb=ku_per_sb,
                col_offset_base_bytes=col_offset_base_bytes,
                mfma_res_ty=mfma_res_ty,
                acc_init=acc_init,
            )

            run_kloop = make_pingpong_kloop(
                num_k_tiles=num_k_tiles,
                tile_k=tile_k,
                prefetch_a_tile=prefetch_a_tile,
                store_a_tile_to_lds=store_a_tile_to_lds,
                load_b_tile=load_b_tile,
                prefetch_scales=prefetch_scales,
                compute_tile=compute_tile,
                hot_loop_scheduler=hot_loop_scheduler,
                lds_load_packs_k64=lds_load_packs_k64,
                lds_base_pong=lds_base_pong,
                lds_base_ping=lds_base_ping,
                row_a_lds_base=row_a_lds_base,
                col_offset_base_bytes=col_offset_base_bytes,
            )
            accs = run_kloop(accs)

            # ===== Epilogue: CShuffle vectorized stores =====
            c_n = n_in
            e_vec = 4 if (tile_n % (32 * 4)) == 0 else 2

            write_row_to_lds, store_pair = make_epilogue_writers(
                accs=accs,
                d_rsrc=d_rsrc,
                out_mlir=out_mlir,
                e_vec=e_vec,
                c_n=c_n,
            )

            # Mask the partial-tile tail: skip stores for global rows at or beyond
            # the owning group's end. Returning (ctx, pred) lets the epilogue skip
            # the whole N-store loop for out-of-group rows.
            def precompute_row(*, row_local, row):
                row_i32 = arith.index_cast(T.i32, row)
                row_valid = arith.cmpi(arith.CmpIPredicate.ult, row_i32, row_limit_i32)
                return (None, row_valid)

            mfma_epilog(
                use_cshuffle=True,
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=tile_n,
                e_vec=e_vec,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lds_out=lds_out,
                frag_elem_type=out_mlir(),
                write_row_to_lds=write_row_to_lds,
                precompute_row=precompute_row,
                store_pair=store_pair,
            )

    # ===== JIT Launcher =====
    @flyc.jit
    def launch_grouped_gemm_blockscale_contiguous(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_tile_group: fx.Tensor,
        arg_tile_row_start: fx.Tensor,
        arg_tile_row_limit: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
        i32_num_m_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid dimensions. The M axis indexes the per-tile dispatch map; its
        # extent is the (host-known) tile count, which is the length of the
        # tile_group/tile_row_start/tile_row_limit arrays.
        n_in = fx.Index(i32_n)
        gx = n_in // fx.Index(tile_n)  # N-blocks
        gy = fx.Index(i32_num_m_tiles)  # M-tiles

        launcher = grouped_gemm_blockscale_contiguous_kernel(
            arg_d,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            arg_tile_group,
            arg_tile_row_start,
            arg_tile_row_limit,
            i32_m,
            i32_n,
            i32_k,
            i32_num_groups,
            i32_num_m_tiles,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        launcher.launch(grid=(gx, gy, 1), block=(total_threads, 1, 1), stream=stream)

    return launch_grouped_gemm_blockscale_contiguous
