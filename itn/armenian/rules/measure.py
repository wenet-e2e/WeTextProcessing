# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Measure(Processor):
    """Spoken quantities with units, e.g. ``իննսունյոթ հերց`` -> ``97 Հց``.

    Units may carry the case endings ``ում``/``ից`` (``կիլոմետր ժամում``);
    date units (``թվական``, ``դար``) also accept ranges such as
    ``հինգից տասը դար`` -> ``5-10 դ.``. All parts stay spoken in the tag.
    """

    def __init__(self, cardinal, decimal):
        super().__init__(name="measure", ordertype="itn", token_orders={})
        integer = (cardinal.graph + delete("ից").ques).optimize()
        units = preferred_map("armenian/data/measurements.tsv", invert=True)
        units = (units + (delete("ում") | delete("ից")).ques).optimize()
        date_units = preferred_map("armenian/data/measurement_dates.tsv", invert=True)
        all_units = (units | date_units).optimize()

        separator = delete(self.SPACE.plus) + insert(" ")
        decimal_tag = decimal.fields_tagger + separator + self.tag_field("units", units)
        cardinal_tag = self.tag_field("integer", integer) + separator + self.tag_field("units", all_units)
        range_tag = self.tag_field("from", integer) + separator
        range_tag += self.tag_field("to", integer) + separator + self.tag_field("units", date_units)
        self.tagger = self.add_tokens(decimal_tag | cardinal_tag | range_tag)

        unit = insert(" ") + self.verbalize_field("units", units, leading_space=True)
        decimal_out = decimal.fields_verbalizer + unit
        cardinal_out = self.verbalize_field("integer", integer) + insert(" ")
        cardinal_out += self.verbalize_field("units", all_units, leading_space=True)
        range_out = self.verbalize_field("from", integer) + insert("-")
        range_out += self.verbalize_field("to", integer, leading_space=True) + insert(" ")
        range_out += self.verbalize_field("units", date_units, leading_space=True)
        self.verbalizer = self.delete_tokens(decimal_out | cardinal_out | range_out)
