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

from tn.marathi.rules.utils import canonical, number_words, to_devanagari, written_map
from tn.processor import Processor

# Minutes read as "minutes to the next hour" (``अकराला पाच मिनिटे``).
TO_MINUTES = range(51, 60)
# Readings with a quarter word before the hour.
QUARTERS = {15: "सव्वा ", 30: "साडे ", 45: "पावणे "}
# Half past one and half past two have their own words.
HALVES = {1: "दीड", 2: "अडीच"}
# The plain "hours past" reading stays available for n-best output.
ALTERNATIVE_WEIGHT = 0.5


class Time(Processor):
    """Written ``H:MM`` clock times read on the 12-hour dial.

    ``११:०२`` -> ``अकरा वाजून दोन मिनिटे``, ``१०:५५`` -> ``अकराला पाच
    मिनिटे``, ``०४:३०`` -> ``साडे चार``, ``१२:४५`` -> ``पावणे एक`` and
    ``०१:३०`` -> ``दीड``. The tagger keeps the written hours and minutes and
    drops the colon.
    """

    def __init__(self):
        super().__init__("time", token_orders={})
        words = number_words()

        def hours_map(reading, hours=range(24)):
            pairs = set()
            for hour in hours:
                for width in (1, 2):
                    pairs.add((to_devanagari(hour, width), reading(hour % 12 or 12)))
            return canonical(written_map(sorted(pairs)))

        def minutes_map(reading, minutes):
            return canonical(written_map([(to_devanagari(minute, 2), reading(minute)) for minute in minutes]))

        def unit(count):
            return words[count] + (" मिनिट" if count == 1 else " मिनिटे")

        def next_hour(hour):
            return words[hour % 12 + 1]

        hour = hours_map(lambda hour: words[hour])
        paths = [
            (hour, minutes_map(lambda minute: " वाजता", [0])),
            (hours_map(lambda hour: next_hour(hour) + "ला"), minutes_map(lambda minute: " " + unit(60 - minute), TO_MINUTES)),
        ]
        named = [hour for hour in range(24) if (hour % 12 or 12) in HALVES]
        for minute, word in QUARTERS.items():
            if minute == 45:
                hours = hours_map(lambda hour, word=word: word + next_hour(hour))
            elif minute == 30:
                hours = hours_map(lambda hour, word=word: word + words[hour], set(range(24)) - set(named))
                hours |= hours_map(lambda hour: HALVES[hour], named)
            else:
                hours = hours_map(lambda hour, word=word: word + words[hour])
            paths.append((hours, minutes_map(lambda minute: "", [minute])))
        special = set(QUARTERS) | set(TO_MINUTES)
        past = minutes_map(lambda minute: " वाजून " + unit(minute), [m for m in range(1, 60) if m not in special])
        alternative = minutes_map(lambda minute: " वाजून " + unit(minute), sorted(special))
        paths.append((hour, past))
        paths.append((hour, add_weight(alternative, ALTERNATIVE_WEIGHT)))

        hours = pynini.union(*[hours for hours, _ in paths]).optimize()
        minutes = pynini.union(*[minutes for _, minutes in paths]).optimize()
        tagger = self.tag_field("hours", hours) + delete(":") + insert(" ") + self.tag_field("minutes", minutes)
        self.tagger = self.add_tokens(tagger)
        verbalizer = [self.verbalize_field("hours", h) + self.verbalize_field("minutes", m, True) for h, m in paths]
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizer))
