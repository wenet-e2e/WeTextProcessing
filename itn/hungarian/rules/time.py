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

from itn.hungarian.rules.utils import DIGIT, data_labels, inflect_abbreviation, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {}


def inflected_units(abbr, word):
    """Inflected spoken unit -> abbreviation, e.g. ``perccel`` -> ``p-cel``."""

    written = {}
    for form, spoken in naive_inflector(abbr, word, True):
        written.setdefault(spoken, form)
    return pynini.string_map([(spoken, form) for spoken, form in written.items()]).optimize()


class Time(Processor):
    """Times, e.g. ``nyolc óra öt perc`` -> ``08:05``, ``nyolc órában`` -> ``8 ó-ban``.

    Raw spoken ``hours``/``minutes``/``seconds``; an inflected trailing unit
    stays in ``suffix`` and a spelled time zone in ``zone``.
    """

    def __init__(self, cardinal):
        super().__init__("time", ordertype="itn", token_orders=TOKEN_ORDER)
        digits = pynini.invert(cardinal.spoken)

        def numbers(first, last):
            return digits @ pynini.union(*[str(x) for x in range(first, last + 1)])

        pad = insert("0") + DIGIT | DIGIT ** 2
        self.hour = numbers(0, 24).optimize()
        self.minute = numbers(1, 59).optimize()
        hour_padded = (self.hour @ pad).optimize()
        minute_padded = (self.minute @ pad).optimize()

        zone = {}
        for key, spelled in data_labels("time/time_zone.tsv"):
            zone.setdefault(spelled, key)
            for form, _ in inflect_abbreviation(spelled, "a", True):
                zone.setdefault(form, form.replace(" ", ""))
        self.zone = pynini.string_map(list(zone.items())).optimize()
        hour_suffix = inflected_units("ó", "óra")
        minute_suffix = inflected_units("p", "perc")
        second_suffix = inflected_units("mp", "másodperc")
        self.suffix = (hour_suffix | minute_suffix | second_suffix).optimize()

        space = pynini.accep(" ")
        hours_t = self.tag_field("hours", self.hour)
        minutes_t = space + self.tag_field("minutes", self.minute)
        seconds_t = space + self.tag_field("seconds", self.minute)
        zone_t = (space + self.tag_field("zone", self.zone)).ques
        hour_word = delete(" óra")
        minute_word = delete(" perc")
        second_word = delete(pynini.union(" másodperc", " masodperc"))

        def suffix_t(graph):
            return space + self.tag_field("suffix", graph)

        clock = hours_t + hour_word + (minutes_t + minute_word).ques + (seconds_t + second_word).ques + zone_t
        spoken = hours_t + suffix_t(hour_suffix)
        spoken |= hours_t + hour_word + minutes_t + suffix_t(minute_suffix)
        spoken |= hours_t + hour_word + minutes_t + minute_word + seconds_t + suffix_t(second_suffix)
        self.tagger = self.add_tokens(clock | spoken)

        hours_v = self.verbalize_field("hours", hour_padded)
        zero = insert(":00")

        def clock_v(name, graph):
            return pynini.cross(" ", ":") + self.verbalize_field(name, graph)

        minutes_v = clock_v("minutes", minute_padded)
        seconds_v = clock_v("seconds", minute_padded)
        clock_graph = hours_v + (minutes_v | zero) | hours_v + (minutes_v | zero) + seconds_v
        zone_v = (space + self.verbalize_field("zone", self.zone)).ques
        suffix_v = space + self.verbalize_field("suffix", self.suffix)
        bare_hours = self.verbalize_field("hours", self.hour)
        bare = space + self.verbalize_field("minutes", self.minute)
        bare_seconds = space + self.verbalize_field("seconds", self.minute)
        spoken_v = bare_hours + suffix_v
        spoken_v |= bare_hours + insert(" ó") + bare + suffix_v
        spoken_v |= bare_hours + insert(" ó") + bare + insert(" p") + bare_seconds + suffix_v
        self.verbalizer = self.delete_tokens(clock_graph + zone_v | spoken_v)
