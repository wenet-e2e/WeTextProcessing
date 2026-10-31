# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import byte, pynutil

from tn.utils import get_abs_path

_ALPHA_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖÜÉ"
_ALPHA_LOWER = "abcdefghijklmnopqrstuvwxyzåäöüé"

SV_LOWER = pynini.union(*_ALPHA_LOWER).optimize()
SV_UPPER = pynini.union(*_ALPHA_UPPER).optimize()
SV_ALPHA = (SV_LOWER | SV_UPPER).optimize()
TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(_ALPHA_UPPER, _ALPHA_LOWER)]).optimize()

# Byte-level sigma for single rewrites composed onto finite graphs.
BYTE_SIGMA = pynini.closure(byte.BYTE).optimize()

# Alternative readings rank after the canonical reading in nondeterministic mode.
ALT = 0.001


def alt(graph, weight=ALT):
    return pynutil.add_weight(graph, weight)


def tsv(name):
    return pynini.string_file(get_abs_path("swedish/data/" + name + ".tsv"))


def labels(name):
    """Reads TSV rows verbatim (no CSV quoting: units include a bare ``"``)."""

    with open(get_abs_path("swedish/data/" + name + ".tsv"), encoding="utf-8") as stream:
        return [line.rstrip("\r\n").split("\t") for line in stream if line.rstrip("\r\n")]


def rewrite(graph, rule, left="", right=""):
    """Applies one context-dependent rewrite to the output side of a finite graph."""

    return (graph @ pynini.cdrewrite(rule, left, right, BYTE_SIGMA)).optimize()


def squeeze_spaces():
    """Strips outer spaces and collapses inner runs of spaces in generated words."""

    char = pynini.difference(byte.BYTE, " ")
    body = char.plus + (pynini.cross(pynini.accep(" ").plus, " ") + char.plus).star
    strip = pynutil.delete(pynini.accep(" ").star)
    return (strip + body + strip).optimize()
