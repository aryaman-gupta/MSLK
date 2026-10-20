# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""When MSLK may use aiter: installed, the pinned build, and not switched off.

Package metadata, the ROCm check and the environment are mocked, so this needs
neither aiter nor a GPU.
"""

import importlib.metadata
import os
import unittest
from unittest import mock

from mslk.aiter import common

PINNED: str = common.pinned_version()


class AiterCommonTest(unittest.TestCase):
    def setUp(self) -> None:
        common.is_aiter_available.cache_clear()
        self.addCleanup(common.is_aiter_available.cache_clear)
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                common.DISABLE_ENV,
                common.DISABLE_OPS_ENV,
                common.ALLOW_UNPINNED_ENV,
            )
        }
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _available(self, installed: str | None, hip: str | None = "7.2") -> bool:
        def fake_version(dist: str) -> str:
            if installed is None:
                raise importlib.metadata.PackageNotFoundError(dist)
            return installed

        with (
            mock.patch.object(common.torch.version, "hip", hip),
            mock.patch.object(
                common.importlib.metadata, "version", side_effect=fake_version
            ),
            mock.patch.object(
                common.importlib.util, "find_spec", return_value=object()
            ),
        ):
            common.is_aiter_available.cache_clear()
            return common.is_aiter_available()

    def test_pin_is_an_exact_noprebuild_build(self) -> None:
        self.assertRegex(PINNED, r"^\d+\.\d+\.\d+\.dev\d+\+g[0-9a-f]+\.noprebuild$")

    def test_available_when_the_pinned_build_is_installed(self) -> None:
        self.assertTrue(self._available(PINNED))

    def test_unavailable_when_not_installed(self) -> None:
        self.assertFalse(self._available(None))

    def test_unavailable_without_rocm(self) -> None:
        self.assertFalse(self._available(PINNED, hip=None))

    def test_other_version_is_not_used(self) -> None:
        with self.assertLogs(common.logger, level="WARNING"):
            self.assertFalse(self._available("0.1.1.dev1+g0000000.noprebuild"))

    def test_other_version_is_used_when_allowed(self) -> None:
        os.environ[common.ALLOW_UNPINNED_ENV] = "1"
        with self.assertLogs(common.logger, level="WARNING"):
            self.assertTrue(self._available("0.1.1.dev1+g0000000.noprebuild"))

    def test_switches(self) -> None:
        with mock.patch.object(common, "is_aiter_available", return_value=True):
            self.assertTrue(common.is_aiter_enabled("f4f4bf16"))
            os.environ[common.DISABLE_OPS_ENV] = "other_op, f4f4bf16"
            self.assertFalse(common.is_aiter_enabled("f4f4bf16"))
            self.assertTrue(common.is_aiter_enabled("other_op_2"))
            del os.environ[common.DISABLE_OPS_ENV]
            os.environ[common.DISABLE_ENV] = "1"
            self.assertFalse(common.is_aiter_enabled("f4f4bf16"))

    def test_switches_cannot_enable_an_unavailable_aiter(self) -> None:
        with mock.patch.object(common, "is_aiter_available", return_value=False):
            self.assertFalse(common.is_aiter_enabled("f4f4bf16"))

    def test_import_refuses_an_unavailable_aiter(self) -> None:
        with mock.patch.object(common, "is_aiter_available", return_value=False):
            with self.assertRaises(RuntimeError):
                common.import_aiter_module("aiter")


if __name__ == "__main__":
    unittest.main()
