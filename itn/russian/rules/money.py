# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from itn.russian.rules.utils import DIGIT, data_path
from tn.processor import Processor


class Money(Processor):
    """Spoken amounts with currencies, e.g. "два рубля" -> money { integer_part: "два" currency: "рубля" } -> "2 руб."."""

    def __init__(self, cardinal, decimal):
        super().__init__(name="money", ordertype="itn")
        singular = pynini.string_file(data_path("currency/currency_singular.tsv")).invert().optimize()
        plural = pynini.string_file(data_path("currency/currency_plural.tsv")).invert().optimize()
        # As in NeMo, a currency word strongly outranks the same word read as a unit.
        space = pynutil.add_weight(pynutil.delete(" "), -100) + pynutil.insert(" ")
        one = cardinal.numbers @ "1"
        other = cardinal.numbers @ pynini.difference(DIGIT.plus, "1")
        # Compound amounts ending in "one" (21, 31, 101, but not 11) agree with the
        # singular noun ("тридцать один рубль"); NeMo only knows the bare "1".
        ends_one = cardinal.numbers @ (DIGIT.star + pynini.difference(DIGIT, "1") + "1")
        # As in NeMo, "1" takes the singular table and every other amount the plural one.
        self.tagger = self.add_tokens(
            (self.tag_field("integer_part", one) | self.tag_field("integer_part", ends_one))
            + space + self.tag_field("currency", singular)
            | (self.tag_field("integer_part", other) | decimal.fields_tagger)
            + space + self.tag_field("currency", plural)
        )

        def currency(graph):
            return pynini.accep(" ") + self.verbalize_field("currency", graph)

        self.verbalizer = self.delete_tokens(
            (self.verbalize_field("integer_part", one) | self.verbalize_field("integer_part", ends_one))
            + currency(singular)
            | self.verbalize_field("integer_part", other) + currency(plural)
            | decimal.fields_verbalizer + currency(plural)
        )
