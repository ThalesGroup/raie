# -*- coding: utf-8 -*-
# mypy: allow-untyped-defs

"""Top-level package for raie."""

__author__ = """Thales Digital Solutions"""

try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

__version__ = ""  # initialize as string

try:
    __version__ = str(version("ca-thalesgroup-trt-raie"))
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from raie import cli
from raie.cli import CLI_LOGGER, cli_main

__all__ = [
    "CLI_LOGGER",
    "cli",
    "cli_main",
]
