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

from itn.marathi.rules.utils import number_words, spoken_map, to_devanagari
from tn.processor import Processor

# Quarter words before the hour and the minutes they stand for.
QUARTERS = {"सव्वा": "१५", "साडे": "३०", "पावणे": "४५"}
# ``दीड`` (half past one) and ``अडीच`` (half past two) name a whole time.
HALVES = {"दीड": "०१:३०", "अडीच": "०२:३०"}


class Time(Processor):
    """Spoken clock times on the 12-hour dial.

    ``अकरा वाजून दोन मिनिटे`` -> ``११:०२`` (minutes past), ``अकराला पाच
    मिनिटे`` -> ``१०:५५`` (minutes to; the ``-ला`` ending stays in the
    hours field), ``साडे चार`` -> ``०४:३०`` and ``दीड`` -> ``०१:३०``. The
    tagger drops ``वाजून`` and ``मिनिट(े)``.
    """

    def __init__(self):
        super().__init__(name="time", ordertype="itn", token_orders={})
        words = number_words()
        hours = spoken_map([(to_devanagari(hour, 2), words[hour]) for hour in range(1, 13)])
        to_hours = spoken_map([(to_devanagari((hour + 10) % 12 + 1, 2), words[hour] + "ला") for hour in range(1, 13)])
        before = spoken_map([(to_devanagari((hour + 10) % 12 + 1, 2), words[hour]) for hour in range(1, 13)])
        minutes = spoken_map([(to_devanagari(minute, 2), words[minute]) for minute in range(1, 60)])
        to_minutes = spoken_map([(to_devanagari(60 - minute, 2), words[minute]) for minute in range(1, 60)])
        halves = pynini.string_map(HALVES.items()).optimize()

        spaces = delete(self.SPACE.plus)
        separator = spaces + insert(" ")
        unit = spaces + delete(pynini.union("मिनिटे", "मिनिट"))
        past = self.tag_field("hours", hours) + spaces + delete("वाजून") + separator
        past += self.tag_field("minutes", minutes) + unit
        to = self.tag_field("hours", to_hours) + separator + self.tag_field("minutes", to_minutes) + unit
        quarters = []
        for word, value in QUARTERS.items():
            graph = before if word == "पावणे" else hours
            quarters.append(self.tag_field("relation", pynini.accep(word)) + separator + self.tag_field("hours", graph))
        half = self.tag_field("hours", halves)
        self.tagger = self.add_tokens(past | to | pynini.union(*quarters) | half)

        colon = insert(":")
        past = self.verbalize_field("hours", hours) + colon + self.verbalize_field("minutes", minutes, True)
        to = self.verbalize_field("hours", to_hours) + colon + self.verbalize_field("minutes", to_minutes, True)
        quarters = []
        for word, value in QUARTERS.items():
            graph = before if word == "पावणे" else hours
            relation = self.verbalize_field("relation", pynini.cross(word, ""))
            quarters.append(relation + self.verbalize_field("hours", graph, True) + insert(":" + value))
        half = self.verbalize_field("hours", halves)
        self.verbalizer = self.delete_tokens(past | to | pynini.union(*quarters) | half)
