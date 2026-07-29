# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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

from pynini import string_file
from pynini.lib.pynutil import delete, insert

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Date(Processor):

    def __init__(self, cardinal=None, input_normalizer=None, range_tagger=None):
        super().__init__(name="date")
        self.input_normalizer = input_normalizer
        self.range_tagger = range_tagger
        self.cardinal = cardinal or Cardinal(input_normalizer=input_normalizer)
        self.year = None
        self.month = None
        self.day = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        yyyy = self.cardinal.thousand
        m = string_file(get_abs_path("japanese/data/date/m.tsv"))
        mm = string_file(get_abs_path("japanese/data/date/mm.tsv"))
        d = string_file(get_abs_path("japanese/data/date/d.tsv"))
        dd = string_file(get_abs_path("japanese/data/date/dd.tsv"))
        rmsign = self.apply_input_processor(
            delete("/") | delete("-") | delete("."),
            self.input_normalizer,
        ) + insert(" ")

        self.year = (yyyy + insert("年")).optimize()
        self.month = self.apply_input_processor(m | mm, self.input_normalizer)
        self.day = self.apply_input_processor(d | dd, self.input_normalizer)

        year = self.tag_field("year", self.year)
        month = self.tag_field("month", self.month)
        day = self.tag_field("day", self.day)

        # yyyy/m/d | yyyy/mm/dd | dd/mm/yyyy
        # yyyy/0m | 0m/yyyy | 0m/dd
        month_two_digit = self.tag_field(
            "month",
            self.apply_input_processor(mm, self.input_normalizer),
        )
        date = ((year + rmsign + month + rmsign + day)
                | (day + rmsign + month + rmsign + year)
                | (year + rmsign + month_two_digit)
                | (month_two_digit + rmsign + year)
                | (month_two_digit + rmsign + day))
        # yyyy/0m | 0m/yyyy | 0m/dd
        simple_date = (year + rmsign + month) | (month + rmsign + year) | (month + rmsign + day)

        tagger = self.add_tokens(date)
        simple_tagger = self.add_tokens(simple_date)

        if self.range_tagger is None:
            self.tagger = tagger
        else:
            self.tagger = tagger + (self.range_tagger + tagger).ques | simple_tagger + self.range_tagger + simple_tagger

    def build_verbalizer(self):
        year = delete('year: "') + self.year + delete('" ')
        month = delete('month: "') + self.month + delete('"')
        day = delete(' day: "') + self.day + delete('"')
        verbalizer = year.ques + month + day.ques
        self.verbalizer = self.delete_tokens(verbalizer)
