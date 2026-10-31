# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.hungarian.rules.utils import DIGIT, NONZERO, data_labels, inflect_abbreviation, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {"time": ["hours", "minutes", "seconds", "suffix", "zone"]}


def unit_forms(abbr, word):
    """A time unit with its abbreviation and inflected forms, e.g. ``p-cel`` -> ``perccel``."""

    forms = pynini.string_map(naive_inflector(abbr, word, True) + [(abbr, word)])
    return (forms | pynini.project(forms, "output")).optimize()


class Time(Processor):
    """Times, e.g. ``08:05`` -> ``nyolc óra öt perc``, ``8 ó 5 p-cel`` -> ``nyolc óra öt perccel``.

    Raw ``hours``/``minutes``/``seconds``; a trailing (inflected) unit is kept
    in ``suffix`` and a time zone in ``zone``. ``00`` minutes or seconds are
    kept raw and silenced by the verbalizer.
    """

    def __init__(self, cardinal):
        super().__init__("time", token_orders=TOKEN_ORDER)
        hour_forms = unit_forms("ó", "óra")
        minute_forms = unit_forms("p", "perc")
        second_forms = unit_forms("mp", "másodperc")
        hour_word = pynini.union("ó", "óra")
        minute_word = pynini.union("p", "perc")

        zone = pynini.string_map([(row[0], row[1]) for row in data_labels("time/time_zone.tsv")])
        for _, spelled in data_labels("time/time_zone.tsv"):
            zone |= pynini.string_map([(form.replace(" ", ""), form)
                                       for form, _ in inflect_abbreviation(spelled, "a", True)])
        self.zone = zone.optimize()

        def numbers(first, last):
            return pynini.union(*[str(x) for x in range(first, last + 1)])

        hour = numbers(0, 24) @ cardinal.plain
        minute = numbers(1, 59) @ cardinal.plain
        self.hour = ((delete("0").ques + DIGIT | NONZERO + DIGIT) @ hour).optimize()
        self.minute = ((delete("0") + NONZERO | NONZERO + DIGIT) @ minute).optimize()
        self.loose_minute = ((delete("0").ques + DIGIT | NONZERO + DIGIT) @ minute).optimize()
        zero = pynini.accep("00")

        colon = delete(":")
        space = insert(" ")
        optional_space = delete(" ").ques + space
        hours_t = self.tag_field("hours", self.hour)
        minutes_t = self.tag_field("minutes", self.minute | zero)
        seconds_t = self.tag_field("seconds", self.minute | zero)
        loose_minutes_t = self.tag_field("minutes", self.loose_minute)
        loose_seconds_t = self.tag_field("seconds", self.loose_minute)
        zone_t = pynini.accep(" ") + self.tag_field("zone", self.zone)

        def suffix_t(forms):
            return self.tag_field("suffix", forms)

        # 08:05, 08:05:05, optionally with a time zone.
        clock = hours_t + colon + space + minutes_t + (colon + space + seconds_t).ques + zone_t.ques
        # 8 ó-ban, 8 óra CET
        spoken = hours_t + pynini.accep(" ") + (suffix_t(hour_forms) | suffix_t(hour_word) + zone_t)
        # 8 ó 5 p-cel, 8 ó 5 perc CET
        units = hours_t + optional_space + delete(hour_word + " ") + loose_minutes_t + optional_space
        spoken |= units + (suffix_t(minute_forms) | suffix_t(minute_word) + zone_t)
        # 8 ó 5 p 3 mp-kor
        spoken |= (units + delete(minute_word + " ") + loose_seconds_t + optional_space
                   + suffix_t(second_forms))
        self.tagger = self.add_tokens(clock | spoken)

        hours_v = self.verbalize_field("hours", self.hour)
        minute_value = self.minute | self.loose_minute

        def unit_v(name, word):
            value = pynini.accep(" ") + self.verbalize_field(name, minute_value) + insert(" " + word)
            return value | self.verbalize_field(name, delete(zero), leading_space=True)

        def value_v(name):
            return pynini.accep(" ") + self.verbalize_field(name, minute_value)

        suffix_v = pynini.accep(" ") + self.verbalize_field("suffix", hour_forms | minute_forms | second_forms)
        zone_v = (pynini.accep(" ") + self.verbalize_field("zone", self.zone)).ques
        hour_v = hours_v + insert(" óra")
        graph = hour_v + unit_v("minutes", "perc") + unit_v("seconds", "masodperc").ques
        graph |= hours_v + suffix_v
        graph |= hour_v + value_v("minutes") + suffix_v
        graph |= hour_v + value_v("minutes") + insert(" perc") + value_v("seconds") + suffix_v
        self.verbalizer = self.delete_tokens(graph + zone_v)
