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

from tn.hebrew.rules.numbers import FEMININE, group
from tn.hebrew.rules.utils import SPACE, next_field
from tn.processor import Processor

# Day-part words and the 24-hour hours each one can describe.
SUFFIXES = {
    "day": (("בבוקר", "לפנות בוקר"), range(0, 13)),
    "noon": (("בצהריים", "אחרי הצהריים", "אחר הצהריים"), range(12, 19)),
    "evening": (("בערב", "לפנות ערב"), range(17, 24)),
    "night": (("בלילה", ), (20, 21, 22, 23, 0, 1, 2, 3, 4)),
}
# Minutes read before the next hour: ``רבע לשש``.
MINUTES_TO = {35: "עשרים וחמישה", 40: "עשרים", 45: "רבע", 50: "עשרה", 55: "חמישה", 59: "דקה"}
# Minutes read after the hour: ``שש וחצי``.
MINUTES_PAST = {1: "דקה", 2: "שתי דקות", 5: "חמישה", 10: "עשרה", 15: "רבע", 30: "חצי"}


def hour_words(hour, twelve_hour):
    if hour % 24 == 0:
        return "חצות"
    if twelve_hour:
        hour = (hour - 1) % 12 + 1
    return group(hour)


def time_words(hour, minute, twelve_hour):
    if minute in MINUTES_TO or minute > 50:
        if minute in MINUTES_TO:
            before = MINUTES_TO[minute]
        else:
            before = ("שתי" if minute == 58 else FEMININE[60 - minute]) + " דקות"
        return before + " ל" + hour_words(hour + 1, twelve_hour)
    words = hour_words(hour, twelve_hour)
    if minute in MINUTES_PAST:
        return words + " ו" + MINUTES_PAST[minute]
    if minute:
        return words + " ו" + group(minute) + " דקות"
    return words


def time_map(hours, twelve_hour):
    pairs = []
    for hour in hours:
        for minute in range(60):
            words = time_words(hour, minute, twelve_hour)
            for written in {str(hour), "%02d" % hour}:
                pairs.append(('hours: "{}" minutes: "{:02d}"'.format(written, minute), words))
    return pynini.string_map(pairs).optimize()


def hour_field(hours):
    return pynini.union(*{form for hour in hours for form in (str(hour), "%02d" % hour)}).optimize()


class Time(Processor):
    """Written times, e.g. ``17:45 בערב`` -> ``רבע לשש בערב``.

    The tagger keeps the hour and minute digits and the day-part word; the
    verbalizer reads a 12-hour clock when a day-part word fits the hour and a
    24-hour clock otherwise, with minutes 35-59 before the next hour.
    """

    def __init__(self, cardinal):
        super().__init__(name="time")
        minutes = pynini.union(*["%02d" % minute for minute in range(60)]).optimize()
        clock = (self.tag_field("hours", hour_field(range(24))) + delete(":") + insert(" ")
                 + self.tag_field("minutes", minutes))
        taggers = [clock]
        verbalizers = [time_map(range(24), twelve_hour=False)]
        for words, hours in SUFFIXES.values():
            words = pynini.union(*words)
            clock = (self.tag_field("hours", hour_field(hours)) + delete(":") + insert(" ")
                     + self.tag_field("minutes", minutes))
            taggers.append(clock + delete(SPACE.plus) + insert(" ") + self.tag_field("suffix", words))
            verbalizers.append(time_map(hours, twelve_hour=True) + next_field(self, "suffix", insert(" ") + words))
        self.tagger = self.add_tokens(cardinal.prefix.ques + pynini.union(*taggers))
        self.verbalizer = self.delete_tokens(cardinal.verbalize_prefix.ques + pynini.union(*verbalizers))
