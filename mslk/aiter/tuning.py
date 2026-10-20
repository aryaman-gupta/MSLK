# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared pieces of the per-op aiter tuning scripts (``bench/aiter/``).

An op's tuning script runs MSLK's kernel and each aiter candidate on every
shape in the op's shape list (``bench/aiter/shapes/<op>.csv``), checks every output against
the reference and tolerance of the op's unit test, and writes the op's tuning
table (format: :mod:`mslk.aiter.routing`) from the fastest correct candidate per
shape.

The rules for choosing, shared by all ops:

- MSLK's kernel is the default. An aiter candidate is chosen only if it is
  correct and at least ``MIN_SPEEDUP`` times faster, so that timing noise does
  not flip shapes between backends.
- An aiter candidate that fails the correctness check, or raises (e.g. a
  launch config that does not fit the GPU), is left out for that shape.
- MSLK's own kernel failing the check is a bug to fix, not a shape to route
  around, so the script reports it and does not write a table.

Merging the tables of several runs (e.g. one per GPU, see ``--shard``)::

    python -m mslk.aiter.tuning merge mslk/gemm/aiter/tables/<op>.csv part0.csv part1.csv ...
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import triton
from mslk.aiter.routing import device_key, MSLK

MIN_SPEEDUP: float = 1.05


@dataclass
class Measurement:
    backend: str
    us: float
    correct: bool
    max_abs_err: float
    # Why the candidate could not run, if it raised; it then counts as incorrect.
    error: Optional[str] = None


def time_us(fn: Callable[[], object], rep_ms: int = 100) -> float:
    """GPU time of one call, from CUDA-graph replays (no launch overhead).

    Two calls first so that autotuning and compilation happen before capture.
    """
    fn()
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=rep_ms) * 1e3


def measure(
    backend: str,
    call: Callable[[], torch.Tensor],
    check: Callable[[torch.Tensor], Tuple[bool, float]],
) -> Measurement:
    """Run ``call`` once, ``check`` its output (-> correct, max abs error), then
    time it. A candidate that raises is recorded as incorrect, not fatal."""
    try:
        correct, max_abs_err = check(call())
        return Measurement(backend, time_us(call), correct, max_abs_err)
    except Exception as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        return Measurement(backend, float("inf"), False, float("inf"), reason)


def choose(measurements: Sequence[Measurement]) -> str:
    """The backend for one shape, by the rules in the module docstring."""
    by_backend = {m.backend: m for m in measurements}
    mslk = by_backend[MSLK]
    best = min((m for m in measurements if m.correct), key=lambda m: m.us)
    if best.backend != MSLK and mslk.us < best.us * MIN_SPEEDUP:
        return MSLK
    return best.backend


def device_columns() -> Dict[str, str]:
    gfx, cu_num = device_key(torch.cuda.current_device())
    return {"gfx": gfx, "cu_num": str(cu_num)}


def read_shapes(path: str) -> List[Dict[str, str]]:
    """A shape list: CSV with a header row; lines starting with ``#`` are comments."""
    with open(path) as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def shard(items: Sequence[Dict[str, str]], spec: str) -> List[Dict[str, str]]:
    """Every n-th item starting at i, for ``spec`` "i/n" (e.g. one shard per GPU)."""
    i, n = (int(x) for x in spec.split("/"))
    return [item for j, item in enumerate(items) if j % n == i]


def _sort_key(row: Dict[str, str], columns: Sequence[str]) -> Tuple[object, ...]:
    return tuple(int(row[c]) if row[c].isdigit() else row[c] for c in columns)


def write_table(
    path: str, rows: Sequence[Dict[str, str]], columns: Sequence[str]
) -> None:
    rows = sorted(rows, key=lambda r: _sort_key(r, columns))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_tables(out: str, parts: Sequence[str]) -> int:
    """Merge the tables of several runs into ``out``; returns the row count."""
    rows: List[Dict[str, str]] = []
    columns: List[str] = []
    for part in parts:
        with open(part) as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or columns)
            rows.extend(reader)
    write_table(out, rows, columns)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="merge the tables of several runs")
    merge.add_argument("out")
    merge.add_argument("parts", nargs="+")
    args = parser.parse_args()
    n = merge_tables(args.out, args.parts)
    print(f"{n} rows from {len(args.parts)} tables -> {args.out}")
