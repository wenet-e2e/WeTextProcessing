# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from itn.spanish.rules.date import numbers_1_to_100
from itn.spanish.rules.utils import DIGIT, MINUS, PLUS, cased, data_file
from tn.processor import Processor

TOKEN_ORDER = {"time": ["hours", "minutes", "suffix", "zone"]}


def _numbers(graph, low, high):
    return (graph @ pynini.union(*[str(i) for i in range(low, high + 1)])).optimize()


class Time(Processor):
    """Raw spoken clock times; V writes "la 1:10 p.m.".

    As in Spanish usage, a full hour without a suffix ("las dos personas")
    and a time without an article and suffix are not converted.
    """

    def __init__(self, cased_input=False):
        super().__init__("time", ordertype="itn", token_orders=TOKEN_ORDER)
        numbers = cased(numbers_1_to_100(), cased_input)
        article = pynini.accep("las ")
        if cased_input:
            article |= pynini.accep("Las ")

        def words(*items):
            return cased(pynini.union(*items), cased_input)

        def drop(graph):
            return pynutil.delete(pynini.project(graph, "input"))

        # Hours: "la una", "las dieciséis".
        one = pynini.cross("la una", "la 1")
        if cased_input:
            one |= pynini.cross("la Una", "la 1") | pynini.cross(pynini.union("La Una", "La una"), "La 1")
        hours = (one | article + _numbers(numbers, 2, 23)).optimize()
        # Hours before a suffix may drop the article: "dos y media de la tarde".
        one_with_suffix = pynini.closure(pynini.union("la ", "La "), 0, 1) + pynini.cross(pynini.union("una", "Una"), "1")
        hours_with_suffix = (one_with_suffix | pynini.closure(article, 0, 1) + _numbers(numbers, 2, 23)).optimize()
        # Hours before which minutes are subtracted.
        hours_to = data_file("time/time_to.tsv")

        two_digits = DIGIT + DIGIT | pynutil.insert("0") + DIGIT
        verbose = pynini.cross(words("media"), "30") | pynini.cross(words("cuarto"), "15")
        minutes = (
            pynini.closure(drop(words("y", "con") + " "), 0, 1)
            + (_numbers(numbers, 1, 59) | verbose)
            + pynini.closure(drop(words(" minuto", " minutos")), 0, 1)
            + pynini.closure(drop(words(" horas")), 0, 1)
        )
        minutes = (minutes @ two_digits).optimize()
        minutes_to = data_file("time/minutes_to.tsv")
        minutes_before = drop(cased(MINUS, cased_input) + " ") + pynini.union(*[
            pynini.cross(words(word), number)
            for word, number in (
                ("cinco", "55"), ("diez", "50"), ("cuarto", "45"), ("veinte", "40"), ("veinticinco", "35"),
            )
        ])
        minutes_before = minutes_before.optimize()

        suffix = data_file("time/time_suffix.tsv")
        zones = pynini.invert(data_file("time/time_zone.tsv"))
        if cased_input:
            suffix |= pynutil.add_weight(data_file("time/time_suffix_cased.tsv"), 0.001)
            zones |= pynutil.add_weight(pynini.invert(data_file("time/time_zone_cased.tsv")), 0.001)
        self.suffix_graph = suffix.optimize()
        sign = pynini.cross(PLUS, "+") | pynini.cross(MINUS, "-")
        offset = pynutil.delete(" ") + sign + pynutil.delete(" ") + _numbers(numbers_1_to_100(), 1, 23)
        self.zone_graph = (zones + pynini.closure(offset, 0, 1)).optimize()

        space = pynini.accep(" ")
        suffix_t = space + self.tag_field("suffix", self.suffix_graph)
        zone_t = space + self.tag_field("zone", self.zone_graph)
        graph = pynini.union(
            self.tag_field("hours", hours) + space + self.tag_field("minutes", minutes),
            self.tag_field("minutes", minutes_to) + space + self.tag_field("hours", hours_to),
            self.tag_field("hours", hours_to) + space + self.tag_field("minutes", minutes_before),
        ) + pynini.closure(suffix_t, 0, 1)
        graph |= self.tag_field("hours", hours_with_suffix) + suffix_t + pynini.closure(zone_t, 0, 1)
        graph |= self.tag_field("hours", hours_with_suffix) + space + self.tag_field("minutes", minutes) + suffix_t
        self.tagger = self.add_tokens(graph)

        def minutes_v(graph):
            return pynutil.insert(":") + self.verbalize_field("minutes", graph, leading_space=True)

        graph = pynini.union(
            self.verbalize_field("hours", (hours | hours_with_suffix).optimize()) + minutes_v(minutes),
            self.verbalize_field("hours", hours_to) + minutes_v((minutes_to | minutes_before).optimize()),
            self.verbalize_field("hours", hours_with_suffix) + pynutil.insert(":00"),
        )
        insert_space = pynutil.insert(" ")
        graph += pynini.closure(insert_space + self.verbalize_field("suffix", self.suffix_graph, True), 0, 1)
        graph += pynini.closure(insert_space + self.verbalize_field("zone", self.zone_graph, True), 0, 1)
        self.verbalizer = self.delete_tokens(graph)
