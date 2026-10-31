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

from tn.processor import Processor
from tn.russian.rules.utils import (
    DIGIT,
    agreement_classes,
    data_path,
    insertion_labels,
    number_names,
    written_numbers,
)
from tn.utils import load_labels


TOKEN_ORDER = {}


class Decimal(Processor):
    """Comma decimals, e.g. "1,08" -> decimal { integer_part: "1" fractional_part: "08" }."""

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        delimiter = insertion_labels("numbers/decimal_delimiter.tsv")["@@decimal_delimiter@@"]
        endings = insertion_labels("numbers/decimal_endings.tsv")
        # Free NeMo readings combine any case of the number with any delimiter
        # or ending form; the agreeing nominative reading is preferred.
        free_integer = cardinal.numbers + delimiter + pynini.closure(
            pynutil.add_weight(pynutil.insert(" и"), 0.5), 0, 1)
        free_fractional = pynini.union(*[
            (DIGIT ** length) @ cardinal.with_leading_zeros + endings[str(10 ** length)]
            for length in range(1, 5)
        ])

        one, few, many = agreement_classes()
        feminine = number_names().feminine
        zeros = pynini.closure(pynutil.delete("0"))

        def agreeing(number, singular, plural):
            return (
                number @ one @ feminine + pynutil.insert(" " + singular)
                | number @ (few | many) @ feminine + pynutil.insert(" " + plural)
            )

        integer = agreeing(written_numbers(), "целая", "целых")
        forms = {}
        for key, form, _ in load_labels(data_path("numbers/decimal_endings.tsv")):
            forms.setdefault(key, []).append(form)
        fractional = pynini.union(*[
            agreeing((DIGIT ** length) @ (zeros + pynini.closure(DIGIT, 1)),
                     *[[form for form in forms[str(10 ** length)] if form.endswith(suffix)][0]
                       for suffix in ("ая", "ых")])
            for length in range(1, 5)
        ])
        self.integer_graph = (free_integer | pynutil.add_weight(integer, -0.5)).optimize()
        self.fractional_graph = (free_fractional | pynutil.add_weight(fractional, -0.5)).optimize()

        self.fields_tagger = (
            cardinal.sign_tagger.ques
            + self.tag_field("integer_part", self.integer_graph)
            + pynutil.delete(",") + pynutil.insert(" ")
            + self.tag_field("fractional_part", self.fractional_graph)
            + cardinal.quantity_tagger.ques
        ).optimize()
        self.fields_verbalizer = (
            cardinal.sign_verbalizer.ques
            + self.verbalize_field("integer_part", self.integer_graph)
            + pynini.accep(" ")
            + self.verbalize_field("fractional_part", self.fractional_graph)
            + cardinal.quantity_verbalizer.ques
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
