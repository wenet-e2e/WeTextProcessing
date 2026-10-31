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

from pynini.lib.pynutil import delete, insert

from tn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Measure(Processor):
    """Quantities with units, e.g. ``52 կգ`` -> ``հիսուներկու կիլոգրամ``.

    Date units (``թ.``, ``դ.``) also accept ranges, e.g. ``5-10 դ.`` ->
    ``հինգից տասը դար``. The tagger keeps the written number(s) and unit.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("measure", token_orders={})
        integer = cardinal.graph
        units = preferred_map("armenian/data/measurements.tsv")
        date_units = preferred_map("armenian/data/measurement_dates.tsv")
        all_units = (units | date_units).optimize()

        separator = self.DELETE_SPACE + insert(" ")
        unit_tag = separator + self.tag_field("units", units)
        date_tag = separator + self.tag_field("units", date_units)
        cardinal_tag = self.tag_field("integer", integer) + separator + self.tag_field("units", all_units)
        range_tag = self.tag_field("from", integer) + delete("-") + insert(" ") + self.tag_field("to", integer)
        self.tagger = self.add_tokens(decimal.fields_tagger + unit_tag | cardinal_tag | range_tag + date_tag)

        unit = insert(" ") + self.verbalize_field("units", units, leading_space=True)
        date_unit = insert(" ") + self.verbalize_field("units", date_units, leading_space=True)
        cardinal_out = self.verbalize_field("integer", integer) + insert(" ")
        cardinal_out += self.verbalize_field("units", all_units, leading_space=True)
        range_out = self.verbalize_field("from", integer) + insert("ից ")
        range_out += self.verbalize_field("to", integer, leading_space=True)
        self.verbalizer = self.delete_tokens(decimal.fields_verbalizer + unit | cardinal_out | range_out + date_unit)
