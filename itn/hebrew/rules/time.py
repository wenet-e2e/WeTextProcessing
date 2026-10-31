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

from itn.hebrew.rules.utils import DIGIT, PREFIX, SPACE, accep_file, first_field, next_field, string_file
from tn.processor import Processor

TOKEN_ORDER = {"time": ["prefix", "hours", "minutes", "suffix"]}

FEMININE_UNITS = ["", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
TENS = ["", "", "עשרים", "שלושים", "ארבעים", "חמישים"]


def feminine(number):
    """The feminine reading NeMo uses for hours and minutes (1-59)."""

    if number < 10:
        return FEMININE_UNITS[number]
    if number == 10:
        return "עשר"
    if number < 20:
        return FEMININE_UNITS[number - 10] + " עשרה"
    tens, units = divmod(number, 10)
    return TENS[tens] + (" ו" + FEMININE_UNITS[units] if units else "")


class Time(Processor):
    """Spoken times, e.g. ``רבע לשש בערב`` -> ``17:45 בערב``.

    ``hours`` keeps the hour word; in the "minutes to hour" form it keeps its
    ``ל`` (``לשש``) and ``minutes`` precedes it in the input. ``minutes`` keeps
    its ``ו`` in the "hour and minutes" form (``וחצי``) and an optional
    ``דקות``. The verbalizer converts the hour to 24-hour time by ``suffix``.
    """

    def __init__(self, cardinal):
        super().__init__(name="time", ordertype="itn", token_orders=TOKEN_ORDER)
        sep = delete(SPACE.plus) + insert(" ")
        number = add_weight(cardinal.graph, -0.7)
        two_digits = pynini.closure(DIGIT, 2, 2)
        midnight = string_file("time/midnight_to_hour.tsv")
        hour = (pynini.union(*[feminine(n) for n in range(1, 13)]) @ number | midnight).optimize()
        to_hour = (delete("ל") + string_file("time/to_hour.tsv")).optimize()
        single = pynini.union(*[feminine(n) for n in range(2, 10)]) @ number @ (insert("0") + DIGIT)
        double = pynini.union(*[feminine(n) for n in range(10, 60)]) @ number @ two_digits
        numeric = (single | double).optimize()
        minutes_word = (delete(SPACE.plus) + delete("דקות")).ques
        and_minutes = delete("ו") + (numeric | string_file("time/minute_verbose.tsv")) + minutes_word
        to_minutes = (string_file("time/minute_to_verbose.tsv")
                      | numeric @ string_file("time/minute_to.tsv") + minutes_word).optimize()

        suffixes = {
            "day": (accep_file("time/day_suffix.tsv"), None),
            "noon": (accep_file("time/noon_suffix.tsv"), string_file("time/hour_to_noon.tsv")),
            "evening": (accep_file("time/evening_suffix.tsv"), string_file("time/hour_to_evening.tsv")),
            "night": (accep_file("time/night_suffix.tsv"), string_file("time/hour_to_night.tsv")),
        }
        prefix = (self.tag_field("prefix", PREFIX) + insert(" ")).ques
        tag_and = self.tag_field("minutes", and_minutes)
        tag_to = self.tag_field("minutes", to_minutes)
        tag_plain = self.tag_field("minutes", numeric)
        taggers = [
            # Without a suffix only midnight and "minutes to hour" are times.
            self.tag_field("hours", midnight),
            self.tag_field("hours", midnight) + sep + tag_and,
            tag_to + sep + self.tag_field("hours", to_hour),
        ]
        verbalize_minutes = (next_field(self, "minutes", insert(":") + and_minutes)
                             | next_field(self, "minutes", insert(":") + numeric) | insert(":00"))
        verbalize_to = next_field(self, "minutes", insert(":") + to_minutes)
        verbalizers = [
            self.verbalize_field("hours", hour) + verbalize_minutes,
            self.verbalize_field("hours", to_hour) + verbalize_to,
        ]
        for words, hour_map in suffixes.values():
            plain = hour if hour_map is None else hour @ hour_map
            to = to_hour if hour_map is None else to_hour @ hour_map
            tag_suffix = sep + self.tag_field("suffix", words)
            taggers.append((self.tag_field("hours", plain) + sep + tag_and
                            | tag_to + sep + self.tag_field("hours", to)
                            | self.tag_field("hours", plain) + (sep + tag_plain).ques) + tag_suffix)
            verbalize_suffix = next_field(self, "suffix", insert(" ") + words)
            verbalizers.append(self.verbalize_field("hours", plain) + verbalize_minutes + verbalize_suffix)
            verbalizers.append(self.verbalize_field("hours", to) + verbalize_to + verbalize_suffix)
        self.tagger = self.add_tokens(prefix + pynini.union(*taggers))
        self.verbalizer = self.delete_tokens(
            first_field(self, "prefix", PREFIX + insert("-")).ques + pynini.union(*verbalizers))
