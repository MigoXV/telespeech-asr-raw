# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = []

# initialize hydra
from fairseq.dataclass.initialize import hydra_init

hydra_init()

import fairseq.models  # noqa
import fairseq.modules  # noqa
import fairseq.token_generation_constraints  # noqa
