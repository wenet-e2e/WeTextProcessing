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

from pynini import closure, cross, string_file
from pynini.lib.pynutil import delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Telephone(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="telephone", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")

        # Single digit: spoken word -> digit character
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        single_digit = digit | zero | cross("o", "0") | cross("oh", "0")

        # 10 digits formatted as XXX-XXX-XXXX
        ten_digits = (
            single_digit + ds + single_digit + ds + single_digit
            + insert("-")
            + ds + single_digit + ds + single_digit + ds + single_digit
            + insert("-")
            + ds + single_digit + ds + single_digit + ds + single_digit + ds + single_digit
        )

        # Optional country code: "plus X" or just digits before the main number
        country_code_digits = (
            closure(single_digit + ds, 0, 2) + single_digit
        )
        country_code = (
            closure(cross("plus ", "+"), 0, 1) + country_code_digits
        )
        optional_country_code = closure(
            country_code + insert(" ") + ds, 0, 1
        )

        graph = optional_country_code + ten_digits
        final_graph = insert('value: "') + graph + insert('"')
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        value = (
            delete("value:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        self.verbalizer = self.delete_tokens(value)
