# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.russian.rules.utils import RU_ALPHA, RU_ALPHA_OR_SPACE, data_path, number_names, written_numbers
from tn.utils import load_labels


TOKEN_ORDER = {}


class Ordinal(Processor):
    """Raw ordinals such as "1996" or "2-ая"; the written ending selects the form."""

    def __init__(self):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        self.numbers = (written_numbers() @ number_names().ordinal).optimize()

        # Each row pairs a spoken ending with a written suffix, e.g. "ая-ая", "ого-го".
        endings = []
        for (row,) in load_labels(data_path("numbers/ordinal_endings.tsv")):
            spoken, written = row.split("-", 1)
            endings.append(pynini.cross(spoken + "-" + written, spoken))
        check_ending = pynini.closure(RU_ALPHA_OR_SPACE) + pynini.union(*endings)
        marked = (self.numbers + pynini.accep("-") + RU_ALPHA.plus) @ check_ending.optimize()
        # A bare number is read as a cardinal first; a written ending marks an ordinal.
        self.graph = (pynutil.add_weight(self.numbers, 0.1) | marked).optimize()

        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
