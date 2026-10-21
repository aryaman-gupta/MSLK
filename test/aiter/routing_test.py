# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Per-shape backend choice: table lookup, switches and forced backends.

The device and aiter's availability are mocked, so this needs neither aiter
nor a GPU.
"""

import unittest
from unittest import mock

from mslk.aiter import routing
from mslk.aiter.routing import choose_backend, force_backend, MSLK, ShapeTable

DEVICE = ("gfx950", 256)


def _table() -> ShapeTable:
    rows = [
        {
            "gfx": "gfx950",
            "cu_num": "256",
            "N": "8192",
            "K": "8192",
            "M_max": m,
            "backend": b,
        }
        for m, b in (("64", "aiter"), ("1024", "mslk"), ("16", "aiter"))
    ]
    rows.append(
        {
            "gfx": "gfx942",
            "cu_num": "304",
            "N": "8192",
            "K": "8192",
            "M_max": "16384",
            "backend": "aiter",
        }
    )
    return ShapeTable(rows, key_columns=("N", "K"), bucket_column="M_max")


class ShapeTableTest(unittest.TestCase):
    def test_bucket_is_the_smallest_bound_at_or_above_m(self) -> None:
        table = _table()
        for m, expected in (
            (1, "aiter"),
            (16, "aiter"),
            (17, "aiter"),
            (64, "aiter"),
            (65, "mslk"),
            (1024, "mslk"),
        ):
            self.assertEqual(table.lookup(DEVICE, (8192, 8192), m), expected, m)

    def test_uncovered_calls(self) -> None:
        table = _table()
        # Beyond the last bucket, an empty call, an untuned key, another device.
        self.assertIsNone(table.lookup(DEVICE, (8192, 8192), 1025))
        self.assertIsNone(table.lookup(DEVICE, (8192, 8192), 0))
        self.assertIsNone(table.lookup(DEVICE, (4096, 8192), 16))
        self.assertIsNone(table.lookup(("gfx950", 304), (8192, 8192), 16))
        self.assertEqual(table.lookup(("gfx942", 304), (8192, 8192), 16), "aiter")


class StringKeyTest(unittest.TestCase):
    def test_key_columns_can_hold_dtypes(self) -> None:
        rows = [
            {
                "gfx": "gfx950",
                "cu_num": "256",
                "out_dtype": d,
                "N": "1024",
                "M_max": "64",
                "backend": b,
            }
            for d, b in (("bfloat16", "aiter"), ("float16", "mslk"))
        ]
        table = ShapeTable(rows, key_columns=("out_dtype", "N"), bucket_column="M_max")
        self.assertEqual(table.lookup(DEVICE, ("bfloat16", 1024), 8), "aiter")
        self.assertEqual(table.lookup(DEVICE, ("float16", 1024), 8), "mslk")


class ChooseBackendTest(unittest.TestCase):
    def _choose(self, m: int, enabled: bool = True) -> str:
        with (
            mock.patch.object(routing, "is_aiter_enabled", return_value=enabled),
            mock.patch.object(routing, "device_key", return_value=DEVICE),
        ):
            return choose_backend("op", _table, 0, (8192, 8192), m)

    def test_uses_the_table_when_enabled(self) -> None:
        self.assertEqual(self._choose(16), "aiter")
        self.assertEqual(self._choose(512), MSLK)
        self.assertEqual(self._choose(4096), MSLK)  # not covered

    def test_mslk_when_disabled(self) -> None:
        self.assertEqual(self._choose(16, enabled=False), MSLK)

    def test_forced_backend_wins_and_is_restored(self) -> None:
        with force_backend("op", MSLK):
            self.assertEqual(self._choose(16), MSLK)
            with force_backend("op", "aiter"):
                self.assertEqual(self._choose(512, enabled=False), "aiter")
            self.assertEqual(self._choose(16), MSLK)
        self.assertEqual(self._choose(16), "aiter")


if __name__ == "__main__":
    unittest.main()
