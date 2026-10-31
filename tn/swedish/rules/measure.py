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

from tn.processor import Processor
from tn.swedish.rules.utils import SV_ALPHA, TO_LOWER, alt, labels, tsv

TOKEN_ORDER = {
    "measure": ["prefix", "negative", "integer", "integer_part", "fractional_part", "quantity",
                "numerator", "denominator", "fraction", "units", "math"],
}


def ranked_map(name):
    """Loads a unit table; repeated symbols rank by row order."""

    seen = {}
    rows = []
    for row in labels(name):
        rank = seen.get(row[0], 0)
        seen[row[0]] = rank + 1
        rows.append((row[0], row[1], str(rank * 0.00001)))
    return pynini.string_map(rows)


class Measure(Processor):
    """Raw numbers and unit symbols; units agree with the spoken number."""

    def __init__(self, cardinal, decimal, fraction, deterministic=True):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        plurals = tsv("measure/unit_plural")
        cased = TO_LOWER.plus + (SV_ALPHA | TO_LOWER).plus

        def unit_table(name):
            table = ranked_map(name)
            return (table | alt(cased @ table, 0.0001)).optimize()

        unit_en = unit_table("measure/unit")
        unit_ett = unit_table("measure/unit_neuter")
        space = pynutil.delete(pynini.accep(" ").ques)
        per = pynini.cross("/", "per") + space + pynutil.insert(" ") + (unit_en | unit_ett)
        optional_per = (space + pynutil.insert(" ") + per).ques

        def units(table, plural):
            graph = (table @ plurals) if plural else table
            return (graph + optional_per | per).optimize()

        units_en_pl = units(unit_en, True)
        units_ett_pl = units(unit_ett, True)
        units_en_sg = units(unit_en, False)
        units_ett_sg = units(unit_ett, False)

        unit_space_t = pynutil.delete(pynini.accep(" ").star) + pynutil.insert(" ")

        def units_t(graph):
            return unit_space_t + self.tag_field("units", graph)

        def units_v(graph):
            return pynutil.insert(" ") + self.verbalize_field("units", graph, leading_space=True)

        sign_t = (self.tag_field("negative", cardinal.sign) + pynutil.insert(" ")).ques
        sign_v = (self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" ") + pynutil.insert(" ")).ques
        decimal_en_t, _ = decimal.fields(en=True)
        decimal_ett_t, decimal_ett_v = decimal.fields(en=False)
        _, quantity_v = decimal.quantity_fields()
        _, plain_en_v = decimal.plain_fields(en=True)
        _, plain_ett_v = decimal.plain_fields(en=False)
        alpha = SV_ALPHA.plus
        am_fm = pynini.union("AM", "FM")
        dash_alpha = pynutil.delete("-") + alpha
        times = pynini.cross(pynini.union("x", "X"), "x")
        prefix = alpha + "-"
        # "/kg" and "per kg" without a number.
        bare = (pynini.cross("/", "per") + space | pynini.accep("per") + pynutil.delete(" "))
        bare = bare + pynutil.insert(" ") + unit_en
        math = self.math_graph(cardinal)
        integers = [
            (cardinal.graph_no_one_en, units_en_pl),
            (cardinal.graph_no_one, units_ett_pl),
            (pynini.cross("1", "ett"), units_ett_sg),
            (pynini.cross("1", "en"), units_en_sg),
        ]

        taggers = [
            sign_t + decimal_en_t + units_t(units_en_pl),
            sign_t + decimal_ett_t + units_t(units_ett_pl),
            decimal_ett_t + units_t(am_fm),
            fraction.fields_tagger + units_t(units_en_pl),
            self.tag_field("units", bare),
            # "3,5-tums", "3,5x" and "abc-3,5".
            decimal_ett_t + pynutil.insert(" ") + self.tag_field("units", dash_alpha | times),
            self.tag_field("prefix", prefix) + pynutil.insert(" ") + decimal_ett_t,
            self.tag_field("math", math),
        ]
        taggers += [sign_t + self.tag_field("integer", number) + units_t(unit) for number, unit in integers]
        integer_v = pynini.union(*[
            self.verbalize_field("integer", number) + units_v(unit) for number, unit in integers])
        verbalizers = [
            sign_v + (
                quantity_v + units_v(units_en_pl | units_ett_pl | am_fm | dash_alpha | times)
                | plain_en_v + units_v(units_en_pl)
                | plain_ett_v + units_v(units_ett_pl | am_fm | dash_alpha | times)
                | integer_v
            ),
            fraction.fields_verbalizer + units_v(units_en_pl),
            self.verbalize_field("units", bare),
            self.verbalize_field("prefix", prefix) + pynutil.delete(" ") + pynutil.insert(" ") + decimal_ett_v,
            self.verbalize_field("math", math),
        ]
        self.tagger = self.add_tokens(pynini.union(*taggers))
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizers))

    def math_graph(self, cardinal):
        number = cardinal.graph
        operations = tsv("math_operations")
        greek = tsv("measure/greek_lower") | pynutil.insert("stort ") + tsv("measure/greek_upper")
        delimiter = pynini.accep(" ") | pynutil.insert(" ")
        equals = pynini.cross("=", "är")
        if not self.deterministic:
            equals |= alt(pynini.cross("=", "är lika med"))
        operand = number | SV_ALPHA | greek
        math = (
            operand + delimiter + operations + (delimiter | SV_ALPHA) + number
            + delimiter + equals + delimiter + operand
            | operand + delimiter + equals + delimiter + (number | SV_ALPHA)
            + delimiter + operations + delimiter + number
        )
        return math.optimize()
