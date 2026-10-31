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

from itn.utils import get_abs_path


def load_rows(rel_path):
    """Loads the tab-separated rows of an Italian ITN data table in file order."""

    rows = []
    with open(get_abs_path(rel_path), encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                rows.append(line.split("\t"))
    return rows


def string_map(pairs):
    """Maps each spoken form to its written form (``(written, spoken)`` pairs)."""

    arcs = [pynini.cross(pynini.escape(spoken), pynini.escape(written)) for written, spoken in pairs]
    return pynini.union(*arcs).optimize()


def table_map(rel_path):
    """Maps the spoken column of a ``written<TAB>spoken`` table to the written one."""

    return string_map([(row[0], row[1]) for row in load_rows(rel_path)])


def capitalized(pairs):
    """Adds capitalized variants, e.g. ``(dott., dottor)`` -> ``(Dott., Dottor)``."""

    extra = [(written[:1].upper() + written[1:], spoken[:1].upper() + spoken[1:]) for written, spoken in pairs]
    return list(dict.fromkeys(list(pairs) + extra))
