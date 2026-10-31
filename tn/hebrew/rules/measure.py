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

from tn.hebrew.rules import numbers
from tn.hebrew.rules.utils import SPACE, next_field, string_file
from tn.processor import Processor


class Measure(Processor):
    """Written measures, e.g. ``12.5°C`` -> ``שתיים עשרה וחצי מעלות צלסיוס``.

    Symbols such as ``%`` may attach to the number; abbreviations such as
    ``ס״מ`` follow it after a space. One unit is ``סנטימטר אחד`` and two is
    ``שני סנטימטר``.
    """

    def __init__(self, cardinal, decimal):
        super().__init__(name="measure")
        joined = string_file("measurements.tsv")
        spaced = string_file("spaced_measurements.tsv")
        units = (joined | spaced).optimize()
        tag_units = (insert(" ") + self.tag_field("units", joined)
                     | delete(SPACE.plus) + insert(" ") + self.tag_field("units", units))
        number = decimal.number | self.tag_field("integer", cardinal.number)
        self.tagger = self.add_tokens(cardinal.prefix.ques + cardinal.negative.ques + number + tag_units)

        integer = cardinal.number
        counted = pynini.cross("2", "שני") | (cardinal.digits - "1" - "2") @ integer
        unit = next_field(self, "units", insert(" ") + units)
        negative = cardinal.verbalize_negative + (decimal.verbalize_number | self.verbalize_field("integer", integer))
        positive = decimal.verbalize_number | self.verbalize_field("integer", counted)
        one = (self.verbalize_field("integer", pynini.cross("1", "")) + delete(" ")
               + self.verbalize_field("units", units + insert(" אחד")))
        self.verbalizer = self.delete_tokens(cardinal.verbalize_prefix.ques + ((negative | positive) + unit | one))

        people = (numbers.grouped_digits() @ ((cardinal.digits - "1" - "2") @ numbers.masculine_graph(999))).optimize()
        people_t = (self.tag_field("integer", people) + delete(SPACE.plus) + insert(" ")
                    + self.tag_field("units", pynini.accep("איש")))
        self.tagger |= self.add_tokens(cardinal.prefix.ques + cardinal.negative.ques + people_t)
        people_v = self.verbalize_field("integer", people) + next_field(self, "units", insert(" ") + pynini.accep("איש"))
        self.verbalizer |= self.delete_tokens(
            cardinal.verbalize_prefix.ques + cardinal.verbalize_negative.ques + people_v)
