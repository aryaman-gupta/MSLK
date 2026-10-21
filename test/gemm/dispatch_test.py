# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""What each GEMM dispatch module (mslk/gemm/dispatch/) asks its table, and
which backend it then calls.

Table lookup and choosing are tested once, in test/aiter/routing_test.py; the
backends' results are tested by each op's own tests, with every backend
forced. What is specific to an op, and tested here, is how its dispatch module
turns the op's arguments into a table key, which arguments each backend can
take, and that it calls the backend it chose. Tensors are on the meta device
and backends are mocked, so no GPU or aiter is needed.
"""

import unittest
from unittest import mock

import torch
from mslk.aiter.routing import force_backend, MSLK
from mslk.gemm.dispatch import f4f4bf16


def _meta(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.uint8, device="meta")


class F4F4BF16DispatchTest(unittest.TestCase):
    # M=5, N=1280, K=2048: XQ and WQ hold two FP4 values per byte.
    XQ: torch.Tensor = _meta(5, 1024)
    WQ: torch.Tensor = _meta(1280, 1024)

    def test_table_key_is_n_k_and_m(self) -> None:
        with mock.patch.object(
            f4f4bf16.ROUTING, "backend", return_value=f4f4bf16.AITER
        ) as lookup:
            self.assertEqual(f4f4bf16.backend_for(self.XQ, self.WQ), f4f4bf16.AITER)
        lookup.assert_called_once_with(self.XQ.get_device(), (1280, 2048), 5)

    def test_arguments_aiter_cannot_take_go_to_mslk(self) -> None:
        with mock.patch.object(f4f4bf16.ROUTING, "backend") as lookup:
            self.assertEqual(
                f4f4bf16.backend_for(self.XQ, self.WQ, global_scale=_meta(1)), MSLK
            )
            self.assertEqual(
                f4f4bf16.backend_for(self.XQ, self.WQ, mxfp4_block_size=16), MSLK
            )
        lookup.assert_not_called()

    def test_calls_the_chosen_backend(self) -> None:
        args = (self.XQ, self.WQ, _meta(5, 64), _meta(1280, 64))
        with (
            mock.patch("mslk.gemm.aiter.f4f4bf16.f4f4bf16") as aiter,
            mock.patch("mslk.gemm.triton.f4f4bf16.f4f4bf16") as triton,
        ):
            with force_backend(f4f4bf16.OP, f4f4bf16.AITER):
                f4f4bf16.f4f4bf16(*args)
            aiter.assert_called_once_with(*args, None)
            triton.assert_not_called()
            with force_backend(f4f4bf16.OP, MSLK):
                f4f4bf16.f4f4bf16(*args)
            triton.assert_called_once()
            self.assertEqual(aiter.call_count, 1)


if __name__ == "__main__":
    unittest.main()
