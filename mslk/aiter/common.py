# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The single gate every aiter path in MSLK goes through.

aiter serves an op only when all of these hold:

- this is a ROCm build of PyTorch and aiter is installed;
- the installed aiter is exactly the build MSLK pins (``aiter_version.txt``),
  the only one its ops are tested against, unless
  ``MSLK_AITER_ALLOW_UNPINNED=1``;
- aiter is not switched off, globally (``MSLK_DISABLE_AITER=1``) or for the op
  (``MSLK_DISABLE_AITER_OPS``, a comma-separated list of op names).

Nothing here imports aiter: checking runs on package metadata, and aiter is
imported on the first call that uses it. Importing aiter is expensive (its
first import compiles a module with hipcc) and loads FlyDSL and Triton into the
process, so it must never happen as a side effect of ``import mslk``.
"""

import functools
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import logging
import os
from types import ModuleType
from typing import Optional

import torch

logger: logging.Logger = logging.getLogger(__name__)

AITER_DIST: str = "amd-aiter"
DISABLE_ENV: str = "MSLK_DISABLE_AITER"
DISABLE_OPS_ENV: str = "MSLK_DISABLE_AITER_OPS"
ALLOW_UNPINNED_ENV: str = "MSLK_AITER_ALLOW_UNPINNED"


@functools.lru_cache(maxsize=1)
def pinned_version() -> str:
    """The aiter version MSLK is tested against."""
    return (
        importlib.resources.files("mslk.aiter")
        .joinpath("aiter_version.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def installed_version() -> Optional[str]:
    """The installed aiter's version, or ``None`` if aiter is not installed."""
    try:
        return importlib.metadata.version(AITER_DIST)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def is_aiter_available() -> bool:
    """True when MSLK may use the installed aiter at all.

    Decided once per process. A different aiter than the pinned one is not
    used, with one warning, since MSLK's tables and tests only vouch for the
    pinned build.
    """
    if torch.version.hip is None:
        return False
    installed = installed_version()
    if installed is None or importlib.util.find_spec("aiter") is None:
        return False
    if installed != pinned_version():
        if os.environ.get(ALLOW_UNPINNED_ENV, "0") == "1":
            logger.warning(
                "Using aiter %s, which is not the version MSLK is tested against "
                "(%s), because %s=1.",
                installed,
                pinned_version(),
                ALLOW_UNPINNED_ENV,
            )
            return True
        logger.warning(
            "aiter %s is installed, but MSLK is tested against aiter %s only, so "
            "it will not use aiter. Install the pinned version, or set %s=1 to "
            "use this one anyway.",
            installed,
            pinned_version(),
            ALLOW_UNPINNED_ENV,
        )
        return False
    return True


def is_aiter_enabled(op: str) -> bool:
    """True when aiter may serve ``op``: available and not switched off.

    The switches are read on every call, so they can be changed at runtime.
    """
    if os.environ.get(DISABLE_ENV, "0") == "1":
        return False
    disabled_ops = os.environ.get(DISABLE_OPS_ENV, "")
    if op in (name.strip() for name in disabled_ops.split(",")):
        return False
    return is_aiter_available()


@functools.lru_cache(maxsize=None)
def import_aiter_module(name: str) -> ModuleType:
    """Import an aiter module, e.g. ``"aiter.ops.triton.gemm.basic.gemm_afp4wfp4"``.

    Callers check :func:`is_aiter_enabled` first; this only guards against
    importing an aiter that MSLK has not accepted.
    """
    if not is_aiter_available():
        raise RuntimeError(
            f"aiter is not available to MSLK (installed: {installed_version()}, "
            f"pinned: {pinned_version()})"
        )
    return importlib.import_module(name)
