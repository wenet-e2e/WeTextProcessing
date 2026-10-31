# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from itn.hindi.rules.utils import (CHAR, DEDH, DELETE_EXTRA_SPACE, DELETE_SPACE, SIGMA, WHITE_SPACE, spoken_to_written,
                                   string_file)
from tn.processor import Processor

TOKEN_ORDER = {}


def address_graph():
    """Addresses: digit words are joined, ordinals and connectors converted, other words kept.

    ``सात शून्य शून्य ओक स्ट्रीट`` -> ``७०० ओक स्ट्रीट``. A context keyword (``स्ट्रीट``, ``रोड``,
    ``फ्लैट`` ...) must appear as a complete word somewhere in the token.
    """

    digit_word = (spoken_to_written("numbers/digit.tsv") | spoken_to_written("numbers/zero.tsv")
                  | spoken_to_written("telephone/eng_digit.tsv") | spoken_to_written("telephone/eng_zero.tsv"))
    digit_word = digit_word.optimize()
    one_space = delete(" ")

    # Structured address: state/city names followed by a six digit pincode.
    names = (string_file("address/states.tsv") | string_file("address/cities.tsv")).optimize()
    pincode = digit_word + pynini.closure(one_space + digit_word, 5, 5)
    structured = add_weight(names + pynini.closure(pynini.accep(", ") + names, 0, 1) + " " + pincode, 1.0)

    special = string_file("address/special_characters.tsv")
    ordinal = string_file("address/ordinals.tsv")
    context = string_file("address/context_cues.tsv")
    digit_unit = (digit_word | string_file("address/digit_passthrough.tsv")).optimize()
    reserved = (digit_unit.copy().project("input") | ordinal.copy().project("input")).optimize()
    any_word = (CHAR - WHITE_SPACE - ",").plus
    text_word = pynini.difference(any_word, reserved).optimize()
    digit_block = digit_unit + add_weight(one_space + digit_unit, -1.0).star
    matchable = pynini.union(add_weight(digit_block, -0.1), add_weight(ordinal, -0.2), add_weight(text_word, 0.1))
    chain = matchable + add_weight(one_space + special + one_space + matchable, -0.5).star
    element = chain + pynini.accep(",").ques
    content = element + (pynini.accep(" ") + element).star
    separator = pynini.union(" ", ",")
    with_context = pynini.union(context + separator + SIGMA, SIGMA + " " + context,
                                SIGMA + " " + context + separator + SIGMA, context).optimize()
    contextual = add_weight(with_context @ content, 1.05)
    return (contextual | structured).optimize()


class Measure(Processor):
    """``ऋण बारह किलोग्राम`` -> ``measure { value: "ऋण बारह किलोग्राम" }`` -> ``-१२ kg``."""

    def __init__(self, cardinal, decimal):
        super().__init__("measure", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        negative = (pynini.cross("ऋण", "-") + DELETE_EXTRA_SPACE).ques
        units = DELETE_EXTRA_SPACE + insert(" ") + spoken_to_written("measure/measurements.tsv")

        def quarter(prefix, integer, fraction, weight):
            return add_weight(delete(prefix) + DELETE_SPACE + integer + DELETE_SPACE + insert(fraction), weight)

        quarters = (quarter("साढ़े", number, ".५", 0.1) | quarter("सवा", number, ".२५", 0.1)
                    | quarter("पौने", cardinal.paune, ".७५", 1) | quarter(DEDH, insert("१"), ".५", 0.1)
                    | quarter("ढाई", insert("२"), ".५", 1))
        by = number + DELETE_SPACE + pynini.cross("बाई", "x") + DELETE_SPACE + number
        graph = negative + (decimal.graph_wo_negative | number | quarters) + units
        graph |= negative + by + units.star
        graph |= address_graph()
        self.graph = graph.optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
