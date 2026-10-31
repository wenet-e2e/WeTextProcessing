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
from tn.russian.rules.utils import agreement_classes, data_path, number_names, ordered_map, written_numbers
from tn.utils import load_labels


TOKEN_ORDER = {}


class Money(Processor):
    """Amounts with a following currency, e.g. "31 руб." -> money { integer_part: "31" currency: "руб." }.

    NeMo reads any case of the amount with any currency form; the agreeing
    nominative reading ("тридцать один рубль", "две копейки") is preferred.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        singular = ordered_map("currency/currency_singular.tsv")
        plural = ordered_map("currency/currency_plural.tsv")
        # NeMo strongly prefers a written space before the currency.
        delimiter = pynini.closure(pynutil.add_weight(pynutil.delete(" "), -100), 0, 1)
        currency_tagger = delimiter + pynutil.insert(" ") + self.tag_field("currency", singular | plural)
        amount_tagger = self.tag_field("integer_part", cardinal.numbers) | decimal.fields_tagger
        self.tagger = self.add_tokens(amount_tagger + currency_tagger)

        def currency(graph):
            return pynini.accep(" ") + self.verbalize_field("currency", graph)

        one = pynini.accep("1") @ cardinal.numbers
        other = pynini.difference(self.VSIGMA, "1") @ cardinal.numbers
        free = (
            self.verbalize_field("integer_part", one) + currency(singular)
            | self.verbalize_field("integer_part", other) + currency(plural)
        )

        names = number_names()
        written = written_numbers()
        classes = agreement_classes()
        agreeing = []
        genitive = []
        forms = {}
        for name, one_form, few_form, many_form, gender in load_labels(
                data_path("currency/currency_agreement.tsv")):
            for index, form in enumerate((one_form, few_form, many_form)):
                forms.setdefault((gender == "f", index), []).append(pynini.cross(name, form))
            genitive.append(pynini.cross(name, few_form))
        for (feminine, index), currencies in sorted(forms.items()):
            number = written @ classes[index] @ (names.feminine if feminine else names.canonical)
            agreeing.append(self.verbalize_field("integer_part", number) + currency(pynini.union(*currencies)))
        # The currency tables weight forms by up to -1, so agreement needs a larger bonus.
        agreeing = pynutil.add_weight(pynini.union(*agreeing), -1.5)
        # Fractions take the genitive singular: "две целых пять десятых рубля".
        fraction = decimal.fields_verbalizer + (
            currency(plural) | pynutil.add_weight(currency(pynini.union(*genitive)), -0.15))
        self.verbalizer = self.delete_tokens(free | agreeing | fraction)
