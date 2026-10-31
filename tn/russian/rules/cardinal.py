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
    RU_ALPHA,
    RU_ALPHA_OR_SPACE,
    data_path,
    latin_to_cyrillic,
    number_names,
    written_numbers,
)


TOKEN_ORDER = {}


class Cardinal(Processor):
    """Raw written integers, e.g. "2 300" -> cardinal { integer: "2 300" }.

    The verbalizer produces every case reading ("две тысячи триста",
    "двух тысяч трехсот", ...), serial spellings ("2-б-34") and the "2-х"
    genitive-plural form, following NeMo's non-deterministic Russian grammar.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        names = number_names()
        written = written_numbers()
        self.numbers = (written @ names.cardinal).optimize()
        self.nominative_numbers = (written @ names.nominative).optimize()
        self.with_leading_zeros = (pynini.closure(pynutil.delete("0")) + self.numbers).optimize()

        digit = (DIGIT @ pynini.string_file(data_path("numbers/cardinals_nominative_case.tsv"))).optimize()
        self.single_digits = (digit + pynini.closure(pynutil.insert(" ") + digit)).optimize()
        self.sign = pynini.cross("-", "минус")
        self.quantity = pynini.string_file(data_path("numbers/quantity.tsv")).optimize()

        x_form = pynini.closure(DIGIT, 1) + pynutil.delete("-х")
        x_form = x_form @ self.with_leading_zeros @ (pynini.closure(RU_ALPHA_OR_SPACE) + "х")
        spelled = (pynutil.add_weight(self._serial(), 10) | x_form).optimize()
        self.integer_graph = (self.with_leading_zeros | spelled).optimize()

        self.sign_tagger = self.tag_field("negative", self.sign) + pynutil.insert(" ")
        self.sign_verbalizer = self.verbalize_field("negative", self.sign) + pynini.accep(" ")
        self.quantity_tagger = (
            pynini.closure(pynutil.add_weight(pynutil.delete(" "), -0.1), 0, 1)
            + pynutil.insert(" ") + self.tag_field("quantity", self.quantity)
        )
        self.quantity_verbalizer = pynini.accep(" ") + self.verbalize_field("quantity", self.quantity)

        number = self.sign_tagger.ques + self.tag_field("integer", self.with_leading_zeros)
        number = pynutil.add_weight(number + self.quantity_tagger.ques, -0.1)
        other = self.tag_field("integer", spelled)
        self.tagger = self.add_tokens(number | other)
        self.verbalizer = self.delete_tokens(
            self.sign_verbalizer.ques + self.verbalize_field("integer", self.integer_graph)
            + self.quantity_verbalizer.ques
        )

    def _serial(self):
        """Letters and digits read one by one, e.g. "c325-b" or "8-913-983"."""
        to_cyrillic = latin_to_cyrillic()
        alpha = to_cyrillic | RU_ALPHA
        num = self.single_digits
        delimiter = pynutil.insert(" ") | pynini.cross("-", " ") | pynini.cross("/", " ")
        letter_num = (alpha + delimiter).plus + num
        num_letter = (num + delimiter).plus + alpha
        num_num = (num + delimiter).plus + num
        serial = (letter_num | num_letter | num_num) + pynini.closure(delimiter + (alpha | num))
        letters = RU_ALPHA | self.input_projection(to_cyrillic)
        chars = pynini.closure(letters | DIGIT | "-" | "/")
        mixed = chars + letters + chars + DIGIT + chars | chars + DIGIT + chars + letters + chars
        serial = mixed.optimize() @ serial.optimize()
        serial |= num + delimiter + num + delimiter + num + pynini.closure(delimiter + num)
        return serial.optimize()
