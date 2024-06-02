# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, WENET COMMUNITY.  Xingchen Song (sxc19@tsinghua.org.cn).
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
from pynini.examples import plurals

from tn.processor import Processor
from tn.utils import get_abs_path


class Telephone(Processor):

    def __init__(self, deterministic: bool = True):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('telephone', ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying telephone, and IP, and SSN which includes country code, number part and extension
        country code optional: +***
        number part: ***-***-****, or (***) ***-****
        extension optional: 1-9999
        E.g
        +1 123-123-5678-1 -> telephone { country_code: "one" number_part: "one two three, one two three, five six seven eight" extension: "one" }
        1-800-GO-U-HAUL -> telephone { country_code: "one" number_part: "one, eight hundred GO U HAUL" }
        """
        add_separator = pynutil.insert(", ")  # between components
        zero = pynini.cross("0", "zero")
        if not self.deterministic:
            zero |= pynini.cross("0", pynini.union("o", "oh"))
        digit = pynini.invert(
            pynini.string_file(get_abs_path(
                "english/data/number/digit.tsv"))).optimize() | zero

        telephone_prompts = pynini.string_file(
            get_abs_path("english/data/telephone/telephone_prompt.tsv"))
        country_code = (
            pynini.closure(telephone_prompts + self.DELETE_EXTRA_SPACE, 0, 1) +
            pynini.closure(pynini.cross("+", "plus "), 0, 1) +
            pynini.closure(digit + self.INSERT_SPACE, 0, 2) + digit +
            pynutil.insert(","))
        country_code |= telephone_prompts
        country_code = pynutil.insert(
            "country_code: \"") + country_code + pynutil.insert("\"")
        country_code = country_code + pynini.closure(
            pynutil.delete("-"), 0, 1) + self.DELETE_SPACE + self.INSERT_SPACE

        area_part_default = pynini.closure(digit + self.INSERT_SPACE, 2,
                                           2) + digit
        area_part = pynini.cross("800", "eight hundred") | pynini.compose(
            pynini.difference(pynini.closure(self.VCHAR), "800"),
            area_part_default)

        area_part = (
            (area_part + (pynutil.delete("-") | pynutil.delete(".")))
            |
            (pynutil.delete("(") + area_part +
             ((pynutil.delete(")") + pynini.closure(pynutil.delete(" "), 0, 1))
              | pynutil.delete(")-")))) + add_separator

        del_separator = pynini.closure(pynini.union("-", " ", "."), 0, 1)
        number_length = ((self.DIGIT + del_separator) |
                         (self.ALPHA + del_separator))**7
        number_words = pynini.closure((self.DIGIT @ digit) +
                                      (self.INSERT_SPACE |
                                       (pynini.cross("-", ', ')))
                                      | self.ALPHA
                                      | (self.ALPHA + pynini.cross("-", ' ')))
        number_words |= pynini.closure((self.DIGIT @ digit) +
                                       (self.INSERT_SPACE |
                                        (pynini.cross(".", ', ')))
                                       | self.ALPHA
                                       | (self.ALPHA + pynini.cross(".", ' ')))
        number_words = pynini.compose(number_length, number_words)
        number_part = area_part + number_words
        number_part = pynutil.insert(
            "number_part: \"") + number_part + pynutil.insert("\"")
        extension = (pynutil.insert("extension: \"") +
                     pynini.closure(digit + self.INSERT_SPACE, 0, 3) + digit +
                     pynutil.insert("\""))
        extension = pynini.closure(self.INSERT_SPACE + extension, 0, 1)

        graph = plurals._priority_union(country_code + number_part,
                                        number_part,
                                        pynini.closure(self.VCHAR)).optimize()
        graph = plurals._priority_union(country_code + number_part + extension,
                                        graph,
                                        pynini.closure(self.VCHAR)).optimize()
        graph = plurals._priority_union(number_part + extension, graph,
                                        pynini.closure(self.VCHAR)).optimize()

        # ip
        ip_prompts = pynini.string_file(
            get_abs_path("english/data/telephone/ip_prompt.tsv"))
        digit_to_str_graph = digit + pynini.closure(
            pynutil.insert(" ") + digit, 0, 2)
        ip_graph = digit_to_str_graph + (pynini.cross(".", " dot ") +
                                         digit_to_str_graph)**3
        graph |= (
            pynini.closure(  # noqa
                pynutil.insert("country_code: \"") + ip_prompts +  # noqa
                pynutil.insert("\"") + self.DELETE_EXTRA_SPACE,
                0,
                1)  # noqa
            + pynutil.insert("number_part: \"")  # noqa
            + ip_graph.optimize() + pynutil.insert("\"")  # noqa
        )
        # ssn
        ssn_prompts = pynini.string_file(
            get_abs_path("english/data/telephone/ssn_prompt.tsv"))
        three_digit_part = digit + (pynutil.insert(" ") + digit)**2
        two_digit_part = digit + pynutil.insert(" ") + digit
        four_digit_part = digit + (pynutil.insert(" ") + digit)**3
        ssn_separator = pynini.cross("-", ", ")
        ssn_graph = three_digit_part + ssn_separator + two_digit_part + ssn_separator + four_digit_part

        graph |= (
            pynini.closure(  # noqa
                pynutil.insert("country_code: \"") + ssn_prompts +  # noqa
                pynutil.insert("\"") + self.DELETE_EXTRA_SPACE,  # noqa
                0,
                1) + pynutil.insert("number_part: \"")  # noqa
            + ssn_graph.optimize()  # noqa
            + pynutil.insert("\"")  # noqa
        )

        final_graph = self.add_tokens(graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing telephone numbers, e.g.
            telephone { country_code: "one" number_part: "one two three, one two three, five six seven eight" extension: "one" }
            -> one, one two three, one two three, five six seven eight, one
        """
        optional_country_code = pynini.closure(
            pynutil.delete("country_code: \"") +
            pynini.closure(self.NOT_QUOTE, 1) + pynutil.delete("\"") +
            self.DELETE_SPACE + self.INSERT_SPACE,
            0,
            1,
        )

        number_part = (
            pynutil.delete("number_part: \"") +
            pynini.closure(self.NOT_QUOTE, 1) + pynini.closure(
                pynutil.add_weight(pynutil.delete(" "), -0.0001), 0, 1) +
            pynutil.delete("\""))

        optional_extension = pynini.closure(
            self.DELETE_SPACE + self.INSERT_SPACE +
            pynutil.delete("extension: \"") +
            pynini.closure(self.NOT_QUOTE, 1) + pynutil.delete("\""),
            0,
            1,
        )

        graph = optional_country_code + number_part + optional_extension
        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
