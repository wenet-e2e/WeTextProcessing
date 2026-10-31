# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def load_pairs(rel_path):
    """Loads ``key<TAB>value`` rows of a Kinyarwanda data table in file order."""

    pairs = []
    with open(get_abs_path(rel_path), encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if not line:
                continue
            key, value = line.split("\t")[:2]
            pairs.append((key, value))
    return pairs


def string_map(rel_path):
    """Builds a ``key -> value`` transducer from a table with unique keys."""

    arcs = [pynini.cross(pynini.escape(key), pynini.escape(value)) for key, value in load_pairs(rel_path)]
    return pynini.union(*arcs).optimize()
