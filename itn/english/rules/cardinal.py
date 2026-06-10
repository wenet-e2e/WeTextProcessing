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

from pynini import closure, cross, difference, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        teen = string_file(get_abs_path("../itn/english/data/numbers/teen.tsv"))
        ties = string_file(get_abs_path("../itn/english/data/numbers/ties.tsv"))
        ds = delete(" ")

        # 1~9, 10~19, 20~99
        one_digit = digit
        two_digit = teen | (ties + (ds + digit | add_weight(insert("0"), 0.1)))
        self.graph_two_digit = two_digit
        up_to_99 = one_digit | two_digit

        # one hundred, one hundred twenty three, one hundred one
        hundreds = digit + ds + delete("hundred") + (ds + two_digit | ds + insert("0") + one_digit | insert("00"))
        # eleven hundred => 1100, twenty one hundred => 2100
        hundreds_as_thousand = (teen | ties + ds + digit) + ds + delete("hundred") + (
            ds + two_digit | ds + insert("0") + one_digit | insert("00")
        )

        # 1~999
        up_to_999 = up_to_99 | hundreds
        self.up_to_999 = up_to_999
        # 1~999 with zero-padding to 3 digits
        up_to_999_padded = hundreds | insert("0") + two_digit | insert("00") + one_digit

        def _with_mag(name):
            return up_to_999 + ds + delete(name)

        def _with_mag_padded(name):
            return up_to_999_padded + ds + delete(name)

        graph = zero | up_to_999 | hundreds_as_thousand
        graph |= _with_mag("thousand") + (ds + up_to_999_padded | insert("000"))
        graph |= (
            _with_mag("million")
            + (ds + _with_mag_padded("thousand") | insert("000"))
            + (ds + up_to_999_padded | insert("000"))
        )
        graph |= (
            _with_mag("billion")
            + (ds + _with_mag_padded("million") | insert("000"))
            + (ds + _with_mag_padded("thousand") | insert("000"))
            + (ds + up_to_999_padded | insert("000"))
        )
        graph |= (
            _with_mag("trillion")
            + (ds + _with_mag_padded("billion") | insert("000"))
            + (ds + _with_mag_padded("million") | insert("000"))
            + (ds + _with_mag_padded("thousand") | insert("000"))
            + (ds + up_to_999_padded | insert("000"))
        )
        graph |= (
            _with_mag("quadrillion")
            + (ds + _with_mag_padded("trillion") | insert("000"))
            + (ds + _with_mag_padded("billion") | insert("000"))
            + (ds + _with_mag_padded("million") | insert("000"))
            + (ds + _with_mag_padded("thousand") | insert("000"))
            + (ds + up_to_999_padded | insert("000"))
        )
        graph |= (
            _with_mag("quintillion")
            + (ds + _with_mag_padded("quadrillion") | insert("000"))
            + (ds + _with_mag_padded("trillion") | insert("000"))
            + (ds + _with_mag_padded("billion") | insert("000"))
            + (ds + _with_mag_padded("million") | insert("000"))
            + (ds + _with_mag_padded("thousand") | insert("000"))
            + (ds + up_to_999_padded | insert("000"))
        )
        graph |= (
            _with_mag("sextillion")
            + (ds + _with_mag_padded("quintillion") | insert("000"))
            + (ds + _with_mag_padded("quadrillion") | insert("000"))
            + (ds + _with_mag_padded("trillion") | insert("000"))
            + (ds + _with_mag_padded("billion") | insert("000"))
            + (ds + _with_mag_padded("million") | insert("000"))
            + (ds + _with_mag_padded("thousand") | insert("000"))
            + (ds + up_to_999_padded | insert("000"))
        )

        # strip leading zeros
        graph = graph @ union(delete(closure("0")) + (self.DIGIT - "0") + closure(self.DIGIT), "0")
        # delete optional "and"
        delete_and = self.build_rule(delete("and "), " ", self.ALPHA)
        graph = (delete_and @ graph).optimize()

        self.graph = graph
        self.graph_no_exception = graph

        # exclude 0-12 from cardinal tagger (they stay as words)
        from itn.english.rules.time import _num_to_word
        exception_labels = ["zero"] + [_num_to_word(x) for x in range(1, 13)]
        exception = union(*exception_labels).optimize()
        graph_with_exception = (difference(self.VSIGMA, exception) @ graph).optimize()

        minus = delete("minus") | delete("negative")
        optional_minus = closure(insert('negative: "-" ') + minus + ds, 0, 1)
        final_graph = optional_minus + insert('integer: "') + graph_with_exception + insert('"')
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        optional_sign = closure(
            delete('negative:') + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE + delete('"') + self.DELETE_SPACE,
            0, 1,
        )
        integer = delete("integer:") + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE.plus + delete('"')
        self.numbers = integer
        self.verbalizer = self.delete_tokens(optional_sign + integer)
