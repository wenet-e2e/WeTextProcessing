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
from pynini.lib import byte, utf8
from pynini.lib.pynutil import delete

from tn.utils import get_abs_path

SPACE = (byte.SPACE | "\u00a0").optimize()
DIGIT = byte.DIGIT
NONZERO = pynini.difference(DIGIT, "0").optimize()
VSIGMA = utf8.VALID_UTF8_CHAR.star.optimize()


def load_labels(rel_path):
    """Loads the ``written<TAB>spoken`` rows of a Hebrew data table in file order."""

    rows = []
    with open(get_abs_path("hebrew/data/" + rel_path), encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                rows.append(line.split("\t"))
    return rows


def string_file(rel_path):
    return pynini.string_map([(pynini.escape(row[0]), row[1]) for row in load_labels(rel_path)]).optimize()


PREFIX = pynini.union(*[row[0] for row in load_labels("prefix.tsv")]).optimize()
# Prefixes that carry (or contract) the definite article read days 1-10 as
# ordinals: ``ה-10.4`` -> ``העשירי באפריל``.
DEFINITE_PREFIX = pynini.union(*[row[0] for row in load_labels("prefix.tsv") if row[0][-1] in "הל"]).optimize()
INDEFINITE_PREFIX = pynini.difference(PREFIX, DEFINITE_PREFIX).optimize()


def first_field(processor, name, graph):
    """Verbalizes a field that is followed by another field."""

    return processor.verbalize_field(name, graph) + delete(" ")


def next_field(processor, name, graph):
    """Verbalizes a field that follows another field."""

    return processor.verbalize_field(name, graph, leading_space=True)
