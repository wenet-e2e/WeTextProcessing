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

import pynini
from pynini.lib.pynutil import delete, insert

from itn.hebrew.rules.cardinal import GROUP_BY_THREES
from itn.hebrew.rules.utils import PREFIX, SPACE, first_field, inverse_file, next_field
from tn.processor import Processor

TOKEN_ORDER = {
    "measure": ["prefix", "negative", "integer", "integer_part", "fractional_part", "fraction", "units"],
}


class Measure(Processor):
    """Spoken measures, e.g. ``שתיים עשרה נקודה חמש מעלות`` -> ``12.5°``.

    Symbols from ``measurements.tsv`` attach to the number; abbreviations
    from ``spaced_measurements.tsv`` follow it after a space. ``סנטימטר אחד``
    puts the unit first; the token order restores number-unit order.
    """

    def __init__(self, cardinal, decimal):
        super().__init__(name="measure", ordertype="itn", token_orders=TOKEN_ORDER)
        sep = delete(SPACE.plus) + insert(" ")
        units = inverse_file("measurements.tsv") | insert(" ") + inverse_file("spaced_measurements.tsv")
        units = units.optimize()

        negative = self.tag_field("negative", pynini.accep("מינוס")) + sep
        number = decimal.number | self.tag_field("integer", cardinal.graph)
        tagged_units = self.tag_field("units", units)
        one = self.tag_field("integer", cardinal.graph @ "1")
        tagger = negative.ques + number + sep + tagged_units | tagged_units + sep + one
        self.tagger = self.add_tokens((self.tag_field("prefix", PREFIX) + insert(" ")).ques + tagger)

        number = decimal.verbalize_number | self.verbalize_field("integer", cardinal.graph @ GROUP_BY_THREES)
        self.verbalizer = self.delete_tokens(
            first_field(self, "prefix", PREFIX + insert("-")).ques
            + first_field(self, "negative", pynini.cross("מינוס", "-")).ques + number
            + next_field(self, "units", units))
