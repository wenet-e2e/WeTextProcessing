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

from itn.russian.rules.utils import data_path
from tn.processor import Processor


class Measure(Processor):
    """Spoken amounts with units, e.g. "двенадцать килограм" -> "12 кг", "пятнадцать за килограм" -> "15/кг"."""

    def __init__(self, cardinal, decimal):
        super().__init__(name="measure", ordertype="itn")
        unit = pynini.string_file(data_path("measurements.tsv"))
        per = pynini.cross("/", "в") | pynini.cross("/", "за")
        slash_unit = per + pynutil.insert(" ") + unit
        unit_slash_unit = pynutil.add_weight(unit + pynutil.insert(" ") + slash_unit, -0.1)
        # "12 кг" keeps a space before the unit; a bare "/кг" attaches to the number.
        self.units_graph = (
            pynutil.insert(" ") + (unit | unit_slash_unit).invert() | slash_unit.invert()
        ).optimize()

        space = pynutil.add_weight(pynutil.delete(" "), -1) + pynutil.insert(" ")
        number_tagger = cardinal.sign_tagger.ques + self.tag_field("integer", cardinal.numbers)
        number_tagger |= decimal.fields_tagger
        self.tagger = self.add_tokens(number_tagger + space + self.tag_field("units", self.units_graph))

        number_verbalizer = cardinal.sign_verbalizer.ques + self.verbalize_field("integer", cardinal.numbers)
        number_verbalizer |= decimal.fields_verbalizer
        self.verbalizer = self.delete_tokens(
            number_verbalizer + pynutil.delete(" ") + self.verbalize_field("units", self.units_graph))
