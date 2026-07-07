# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import mslk.gemm  # noqa: F401
import torch
from mslk.utils.flydsl import is_flydsl_available


@unittest.skipUnless(
    torch.version.hip is not None
    and torch.cuda.is_available()
    and is_flydsl_available(),
    "requires ROCm GPU with FlyDSL",
)
class FlyDSLVectorAddTest(unittest.TestCase):
    def test_vector_add(self) -> None:
        n = 256
        a = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
        b = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
        c = torch.ops.mslk.flydsl_vector_add(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a + b)


if __name__ == "__main__":
    unittest.main()
