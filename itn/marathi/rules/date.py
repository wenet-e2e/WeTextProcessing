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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.marathi.rules.utils import load_pairs, number_words, spoken_map, to_devanagari
from tn.processor import Processor

# A lone number after a month reads as the day unless it cannot be one.
YEAR_ONLY_WEIGHT = 0.01


class Date(Processor):
    """Spoken dates in either day-month or month-day order.

    ``पंधरा फेब्रुवारी अठराशे सतरा`` -> ``१५ फेब्रुवारी १८१७``, ``जानेवारी
    तेरा दोन हजार तेवीस`` -> ``जानेवारी १३ २०२३`` and ``इसवी सन पूर्व
    चारशे`` -> ``इ.स.पू. ४००``. The tagger keeps the spoken day, month,
    year and era in spoken order.
    """

    def __init__(self, cardinal):
        super().__init__(name="date", ordertype="itn", token_orders={})
        words = number_words()
        self.day = spoken_map([(to_devanagari(day), words[day]) for day in range(1, 32)])
        self.month = pynini.union(*[spoken for _, spoken in load_pairs("date/months.tsv")]).optimize()
        self.year = cardinal.graph
        self.era = spoken_map(load_pairs("date/eras.tsv"))
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        sep = delete(self.SPACE.plus) + insert(" ")
        day = self.tag_field("day", self.day)
        month = self.tag_field("month", self.month)
        year = self.tag_field("year", self.year)
        era = self.tag_field("era", self.era)
        mdy = month + sep + (day | add_weight(year, YEAR_ONLY_WEIGHT) | day + sep + year)
        dmy = day + sep + month + (sep + year).ques
        self.tagger = self.add_tokens(mdy | dmy | era + sep + year)

    def build_verbalizer(self):
        sep = insert(" ")
        day = self.verbalize_field("day", self.day, True)
        month = self.verbalize_field("month", self.month, True)
        year = self.verbalize_field("year", self.year, True)
        mdy = self.verbalize_field("month", self.month) + sep + (day | year | day + sep + year)
        dmy = self.verbalize_field("day", self.day) + sep + month + (sep + year).ques
        era = self.verbalize_field("era", self.era) + sep + year
        self.verbalizer = self.delete_tokens(mdy | dmy | era)
