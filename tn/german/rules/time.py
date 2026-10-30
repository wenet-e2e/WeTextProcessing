# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {"time": ["hours", "minutes", "seconds", "zone"]}


class Time(Processor):
    """Raw clock fields with bounded, hour-first colloquial verbalization."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("time", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        canonical = self.DIGIT | ((pynutil.delete("0") | (self.DIGIT - "0")) + self.DIGIT)
        numbers = cardinal.digits @ pynini.accep("null") | cardinal.two_digits
        self.number_graph = (canonical @ numbers).optimize()
        hours = pynini.union(*[str(hour) for hour in range(25)])
        minute_values = pynini.union(*[str(minute) for minute in range(60)])
        minutes = pynini.union(*["{:02d}".format(minute) for minute in range(60)])
        raw_hours = self.input_projection(canonical @ hours)
        self.zone_graph = pynini.string_file(get_abs_path("german/data/time/time_zone.tsv")).optimize()

        def inverse_table(name):
            return pynini.string_file(get_abs_path("german/data/time/" + name + ".tsv")).invert().optimize()

        night = pynutil.add_weight(inverse_table("hour_to_night"), 0.0001)
        self.night_hour_graph = (
            canonical @ hours @ pynini.cdrewrite(night, "[BOS]", "[EOS]", self.VSIGMA)
        ).optimize()
        self.next_hour_graph = (self.night_hour_graph @ inverse_table("hour_to")).optimize()
        self.minute_to_graph = (canonical @ inverse_table("minute_to")).optimize()
        quarter_number = numbers | pynini.cross("15", "viertel")
        self.past_minute_graph = (
            canonical @ pynini.union(*[str(minute) for minute in range(1, 31)]) @ quarter_number
        ).optimize()
        self.to_minute_graph = (
            self.minute_to_graph @ pynini.union(*[str(minute) for minute in range(1, 30)]) @ quarter_number
        ).optimize()
        self.hour_graph = (
            canonical @ hours @ numbers
            @ pynini.cdrewrite(pynini.cross("eins", "ein"), "[BOS]", "[EOS]", self.VSIGMA)
        ).optimize()

        hour = self.tag_field("hours", raw_hours)
        minute = self.tag_field("minutes", minutes)
        second = self.tag_field("seconds", minutes)
        suffix = pynutil.delete(" Uhr") | pynutil.delete("uhr")
        optional_zone = pynini.closure(pynini.accep(" ") + self.tag_field("zone", self.zone_graph), 0, 1)
        hour_only = self.tag_field("hours", hours) + suffix
        hour_minute = hour + pynutil.delete(pynini.union(":", ".")) + pynutil.insert(" ") + minute + suffix
        hour_minute_second = (
            hour + pynini.cross(":", " ") + minute + pynini.cross(":", " ") + second + suffix
        )
        self.fields_tagger = ((hour_only | hour_minute | hour_minute_second) + optional_zone).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        hour_v = self.verbalize_field("hours", self.hour_graph) + pynutil.insert(" uhr")
        silent_zero = canonical @ pynutil.delete("0")
        zero_minute_v = self.verbalize_field("minutes", silent_zero, leading_space=True)
        nonzero_minute_v = self.verbalize_field(
            "minutes", canonical @ pynini.difference(minute_values, "0") @ numbers,
            leading_space=True,
        )
        ordinary_hm = hour_v + (zero_minute_v | pynutil.insert(" ") + nonzero_minute_v)

        def counted_component(name, singular, plural):
            counted = (
                canonical @ pynini.cross("1", "eine " + singular)
                | canonical @ pynini.difference(minute_values, "1") @ numbers
                + pynutil.insert(" " + plural)
            )
            return self.verbalize_field(name, counted, leading_space=True)

        hms = (
            hour_v + pynutil.insert(" ") + counted_component("minutes", "minute", "minuten")
            + pynutil.insert(" ") + counted_component("seconds", "sekunde", "sekunden")
        )

        # Consume the raw hour first and remember its finite clock value in
        # graph state. Each branch then speaks the minute before that hour.
        # This preserves one flat TOKEN_ORDER without factorial field swapping
        # or precomputed spoken phrases in the tagger.
        colloquial = []
        for hour_value in range(25):
            value = str(hour_value)
            consume_hour = self.verbalize_field("hours", canonical @ pynutil.delete(value))
            past_hour = (pynini.accep(value) @ self.night_hour_graph @ numbers).project("output").optimize()
            next_hour = (pynini.accep(value) @ self.next_hour_graph @ numbers).project("output").optimize()
            past = (
                consume_hour + self.verbalize_field("minutes", self.past_minute_graph, leading_space=True)
                + pynutil.insert(" nach ") + pynutil.insert(past_hour)
            )
            half = (
                consume_hour + self.verbalize_field("minutes", canonical @ pynini.cross("30", "halb"),
                                                    leading_space=True)
                + pynutil.insert(" ") + pynutil.insert(next_hour)
            )
            to = (
                consume_hour + self.verbalize_field("minutes", self.to_minute_graph, leading_space=True)
                + pynutil.insert(" vor ") + pynutil.insert(next_hour)
            )
            colloquial.append(past | half | to)
        zone_v = pynini.closure(
            pynutil.insert(" ") + self.verbalize_field("zone", self.zone_graph, leading_space=True), 0, 1,
        )
        self.fields_verbalizer = (
            (hour_v | ordinary_hm | hms | pynutil.add_weight(pynini.union(*colloquial), 0.0001)) + zone_v
        ).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
