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
from tn.russian.rules.utils import RU_ALPHA, ordered_map


TOKEN_ORDER = {}


class Measure(Processor):
    """Numbers with units ("12 кг", "3 км/ч", "1/кг") or attached words ("тест-8")."""

    def __init__(self, cardinal, decimal):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        unit = ordered_map("measurements.tsv")
        per = pynini.cross("/", "в") | pynini.cross("/", "за")
        slash_unit = (per + pynutil.insert(" ") + unit).optimize()
        unit_slash_unit = pynutil.add_weight(unit + pynutil.insert(" ") + slash_unit, -0.1)
        default_units = (unit | unit_slash_unit).optimize()
        self.units_graph = (default_units | slash_unit).optimize()
        word = RU_ALPHA.plus
        # A written space before the unit is preferred, as in NeMo.
        delete_space = pynini.closure(pynutil.add_weight(pynutil.delete(pynini.union(" ", "\u00a0")), -1), 0, 1)
        dash = pynini.closure(pynutil.delete("-"), 0, 1)
        space = pynutil.insert(" ")

        cardinal_tagger = cardinal.sign_tagger.ques + self.tag_field("integer", cardinal.numbers)
        number_tagger = cardinal_tagger | decimal.fields_tagger
        units_tagger = (
            delete_space + space + self.tag_field("units", default_units)
            | space + self.tag_field("units", slash_unit)
        )
        nominative_tagger = self.tag_field("integer", cardinal.nominative_numbers)
        # Attached words rank below units and written ordinal endings ("8-го").
        suffix_tagger = self.tag_field("integer", cardinal.numbers) + dash + space + self.tag_field("suffix", word)
        tagger = (
            number_tagger + units_tagger
            | pynutil.add_weight(suffix_tagger, 0.5)
            | decimal.fields_tagger + pynutil.delete("-") + space + self.tag_field("suffix", word)
            | self.tag_field("prefix", word) + dash + space + nominative_tagger
            | self.tag_field("prefix", word) + pynutil.delete("-") + space + decimal.fields_tagger
        )

        cardinal_verbalizer = cardinal.sign_verbalizer.ques + self.verbalize_field("integer", cardinal.numbers)
        tail = pynini.accep(" ") + (self.verbalize_field("units", self.units_graph) | self.verbalize_field("suffix"))
        prefix = self.verbalize_field("prefix") + pynini.accep(" ")
        # One shared decimal copy; field combinations the tagger never emits are harmless.
        verbalizer = (
            cardinal_verbalizer + tail
            | prefix + self.verbalize_field("integer", cardinal.nominative_numbers)
            | prefix.ques + decimal.fields_verbalizer + tail.ques
        )
        self.tagger = self.add_tokens(tagger)
        self.verbalizer = self.delete_tokens(verbalizer)
