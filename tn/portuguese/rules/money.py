# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {"money": ["integer_part", "fractional_part", "quantity", "currency_maj"]}


class Money(Processor):
    """Raw amounts and currency symbols, e.g. R$ 12,05 -> doze reais e cinco centavos.

    Without a quantity, one- or two-digit non-zero cents are read as minor
    units and every other fraction as a decimal. Amounts ending in a scale
    word take "de": R$ 1000000 -> um milhão de reais.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders=TOKEN_ORDER)

        def table(name):
            return load_labels(get_abs_path("portuguese/data/money/" + name))

        majors = table("currency_major.tsv")
        major_plural = dict(table("currency_major_plural.tsv"))
        minors = dict(table("currency_minor.tsv"))
        minor_plural = dict(table("currency_minor_plural.tsv"))
        symbols = pynini.union(*[symbol for symbol, _ in majors]).optimize()

        def spell(value):
            return pynini.shortestpath(pynini.accep(str(value)) @ cardinal.graph).string()

        cents = [(str(d), spell(10 * d)) for d in range(1, 10)]
        cents += [("{:02d}".format(value), spell(value)) for value in range(1, 100)]
        cents = pynini.string_map(cents).optimize()
        cent_one = pynini.accep("01") @ cents
        cent_other = pynini.difference(self.VSIGMA, "01") @ cents
        fractional = decimal.fractional_graph
        fraction_decimal = pynini.difference(self.VSIGMA, self.input_projection(cents)) @ fractional

        number = cardinal.graph
        scale_words = pynini.union(*cardinal.scale_words)
        ends_scale = self.VSIGMA + " " + scale_words
        one = pynini.accep("1") @ number
        other = pynini.difference(self.VSIGMA, "1") @ number
        other_de = (other @ ends_scale) + insert(" de")
        other_plain = other @ pynini.difference(self.VSIGMA, ends_scale)
        other = (other_de | other_plain).optimize()
        positive = (pynini.difference(self.VSIGMA, "0") @ other).optimize()
        quantity = decimal.quantity_graph
        quantity = (quantity @ scale_words + insert(" de") | quantity @ pynini.difference(self.VSIGMA, scale_words))

        amount_t = (
            self.tag_field("integer_part", number)
            + (delete(",") + insert(" ") + self.tag_field("fractional_part", fractional)).ques
            + (self.DELETE_SPACE + insert(" ") + self.tag_field("quantity", quantity)).ques
        )
        currency_t = self.tag_field("currency_maj", symbols)
        self.tagger = self.add_tokens(
            currency_t + self.DELETE_SPACE + insert(" ") + amount_t
            | amount_t + self.DELETE_SPACE + insert(" ") + currency_t
        )

        def integer(graph):
            return self.verbalize_field("integer_part", graph)

        def fraction(graph):
            return self.verbalize_field("fractional_part", graph, leading_space=True)

        def currency(graph):
            return self.verbalize_field("currency_maj", graph, leading_space=True)

        singular = pynini.string_map(majors)
        plural = pynini.string_map([(symbol, major_plural[word]) for symbol, word in majors])
        comma = insert(" " + decimal.separator + " ")
        quantity_v = insert(" ") + self.verbalize_field("quantity", quantity, leading_space=True)
        graphs = [
            # R$ 1 -> um real; R$ 1000000 -> um milhão de reais
            integer(one) + insert(" ") + currency(singular),
            integer(other) + insert(" ") + currency(plural),
            # R$ 1,234 -> um vírgula dois três quatro real
            integer(one) + comma + fraction(fraction_decimal) + insert(" ") + currency(singular),
            integer(other) + comma + fraction(fraction_decimal) + insert(" ") + currency(plural),
            # R$ 1,5 milhões -> um vírgula cinco milhões de reais
            integer(number) + (comma + fraction(fractional)).ques + quantity_v + insert(" ") + currency(plural),
        ]
        for symbol, word in majors:
            drop = currency(delete(symbol))
            minor = minors[symbol]

            def minor_reading(prefix):
                return (
                    prefix + fraction(cent_one) + insert(" " + minor)
                    | prefix + fraction(cent_other) + insert(" " + minor_plural[minor])
                ) + drop

            # R$ 0,20 -> vinte centavos; R$ 12,05 -> doze reais e cinco centavos
            graphs.append(minor_reading(integer(delete("0"))))
            graphs.append(minor_reading(integer(one) + insert(" " + word + " e ")))
            graphs.append(minor_reading(integer(positive) + insert(" " + major_plural[word] + " e ")))
        self.verbalizer = self.delete_tokens(pynini.union(*graphs))
