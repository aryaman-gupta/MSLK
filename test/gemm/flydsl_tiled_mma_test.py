# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest

import mslk.gemm  # noqa: F401
import torch
from mslk.utils import flydsl_aot
from mslk.utils.flydsl import is_flydsl_available


@unittest.skipUnless(
    torch.version.hip is not None
    and torch.cuda.is_available()
    and is_flydsl_available(),
    "requires ROCm GPU with FlyDSL",
)
class FlyDSLTiledMmaTest(unittest.TestCase):
    def test_op_matches_reference(self) -> None:
        a = torch.randn(64, 8, dtype=torch.float32).cuda()
        b = torch.randn(64, 8, dtype=torch.float32).cuda()
        c = torch.ops.mslk.flydsl_tiled_mma(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a @ b.T, atol=1e-4, rtol=1e-4)

    def test_aot_compile_populates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            flydsl_aot.compile_aot(cache_dir)
            import glob
            import os

            pkls = glob.glob(os.path.join(cache_dir, "**", "*.pkl"), recursive=True)
            # 2 configs x 2 archs = 4 cache entries.
            self.assertEqual(len(pkls), 4)


if __name__ == "__main__":
    unittest.main()
