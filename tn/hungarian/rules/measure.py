# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from tn.hungarian.rules.utils import data_map, inflected_map
from tn.processor import Processor

TOKEN_ORDER = {}


class Measure(Processor):
    """Measures, e.g. ``3mm-ben`` -> ``három milliméterben``, ``5 km/h``.

    The number keeps the raw cardinal/decimal/fraction fields and the
    (possibly inflected) unit stays raw in ``units``.
    """

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        unit = (data_map("measures/measurements.tsv")
                | inflected_map("measures/measurements.tsv", lower=True)).optimize()
        per = pynini.cross("/", "per ") + unit
        self.units = (unit + (pynini.cross("", " ") + per).ques | per).optimize()
        times = pynini.union("x", "X")

        number_t = (
            self.tag_field("integer", cardinal.graph)
            | decimal.unsigned_tagger
            | fraction.unsigned_tagger
        )
        space_t = delete(" ").ques + insert(" ")
        units_t = self.tag_field("units", self.units)
        self.tagger = self.add_tokens(
            cardinal.sign_tagger + number_t + space_t + units_t
            | self.tag_field("integer", cardinal.graph) + insert(" ") + self.tag_field("units", times)
        )

        number_v = (
            self.verbalize_field("integer", cardinal.graph)
            | decimal.unsigned_verbalizer
            | fraction.unsigned_verbalizer
        )
        units_v = pynini.accep(" ") + self.verbalize_field("units", self.units | times)
        self.verbalizer = self.delete_tokens(cardinal.sign_verbalizer + number_v + units_v)
