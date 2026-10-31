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
from pynini.lib.pynutil import add_weight

from tn.utils import get_abs_path

# Penalty per rank among the readings of one written key.
RANK_WEIGHT = 0.01


def load_pairs(rel_path, invert=False):
    """Loads ``key<TAB>value`` rows of an Armenian data table in file order."""

    pairs = []
    with open(get_abs_path(rel_path), encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if not line:
                continue
            key, value = line.split("\t")[:2]
            pairs.append((value, key) if invert else (key, value))
    return pairs


def preferred_map(rel_path, invert=False, replace_space=None):
    """Builds a weighted map from a table whose keys may repeat.

    NeMo leaves the choice between repeated keys (e.g. ``$`` -> ``դոլար`` or
    ``ամերիկյան դոլար``) to arbitrary shortest-path tie breaking. Here the
    shortest reading wins (file order breaks ties); the others stay
    available as weighted alternatives for n-best output.
    """

    readings = {}
    for index, (key, value) in enumerate(load_pairs(rel_path, invert)):
        if replace_space is not None:
            value = value.replace(" ", replace_space)
        candidates = readings.setdefault(key, [])
        if value not in [candidate for _, _, candidate in candidates]:
            candidates.append((len(value), index, value))
    arcs = []
    for key, candidates in readings.items():
        for rank, (_, _, value) in enumerate(sorted(candidates)):
            arc = pynini.cross(pynini.escape(key), pynini.escape(value))
            arcs.append(add_weight(arc, rank * RANK_WEIGHT) if rank else arc)
    return pynini.union(*arcs).optimize()
