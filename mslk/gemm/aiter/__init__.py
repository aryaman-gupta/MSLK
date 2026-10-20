# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""aiter backends for MSLK GEMM ops on ROCm.

Each module adapts one MSLK op to aiter's public op for the same computation,
and owns the op's tuning table under ``tables/``. Importing these modules does
not import aiter; see :mod:`mslk.aiter.common`.
"""
