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

from pynini import closure, cross, compose, difference, invert, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert
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
        super().__init__("telephone", ordertype="en_tn")
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
        add_separator = insert(", ")  # between components
        zero = cross("0", "zero")
        if not self.deterministic:
            zero |= cross("0", union("o", "oh"))
        digit = (
            invert(
                string_file(get_abs_path("english/data/number/digit.tsv"))
            ).optimize()
            | zero
        )

        telephone_prompts = string_file(
            get_abs_path("english/data/telephone/telephone_prompt.tsv")
        )
        country_code = (
            (telephone_prompts + self.DELETE_EXTRA_SPACE).ques
            + cross("+", "plus ").ques
            + closure(digit + insert(" "), 0, 2)
            + digit
            + insert(",")
        )
        country_code |= telephone_prompts
        country_code = insert('country_code: "') + country_code + insert('"')
        country_code = country_code + delete("-").ques + self.DELETE_SPACE + insert(" ")

        area_part_default = (digit + insert(" ")) ** 2 + digit
        area_part = cross("800", "eight hundred") | compose(
            difference(self.VSIGMA, "800"), area_part_default
        )

        area_part = (
            (area_part + (delete("-") | delete(".")))
            | (
                delete("(")
                + area_part
                + ((delete(")") + delete(" ").ques) | delete(")-"))
            )
        ) + add_separator

        del_separator = union("-", " ", ".").ques
        number_length = (
            (self.DIGIT + del_separator) | (self.ALPHA + del_separator)
        ) ** 7
        number_words = (
            (self.DIGIT @ digit) + (insert(" ") | (cross("-", ", ")))
            | self.ALPHA
            | (self.ALPHA + cross("-", " "))
        ).star
        number_words |= (
            (self.DIGIT @ digit) + (insert(" ") | (cross(".", ", ")))
            | self.ALPHA
            | (self.ALPHA + cross(".", " "))
        ).star
        number_words = compose(number_length, number_words)
        number_part = area_part + number_words
        number_part = insert('number_part: "') + number_part + insert('"')
        extension = (
            insert('extension: "')
            + closure(digit + insert(" "), 0, 3)
            + digit
            + insert('"')
        )
        extension = (insert(" ") + extension).ques

        graph = plurals._priority_union(
            country_code + number_part, number_part, self.VSIGMA
        ).optimize()
        graph = plurals._priority_union(
            country_code + number_part + extension, graph, self.VSIGMA
        ).optimize()
        graph = plurals._priority_union(
            number_part + extension, graph, self.VSIGMA
        ).optimize()

        # ip
        ip_prompts = string_file(get_abs_path("english/data/telephone/ip_prompt.tsv"))
        digit_to_str_graph = digit + closure(insert(" ") + digit, 0, 2)
        ip_graph = digit_to_str_graph + (cross(".", " dot ") + digit_to_str_graph) ** 3
        graph |= (
            (
                insert('country_code: "')
                + ip_prompts
                + insert('"')
                + self.DELETE_EXTRA_SPACE
            ).ques
            + insert('number_part: "')  # noqa
            + ip_graph.optimize()
            + insert('"')  # noqa
        )
        # ssn
        ssn_prompts = string_file(get_abs_path("english/data/telephone/ssn_prompt.tsv"))
        three_digit_part = digit + (insert(" ") + digit) ** 2
        two_digit_part = digit + insert(" ") + digit
        four_digit_part = digit + (insert(" ") + digit) ** 3
        ssn_separator = cross("-", ", ")
        ssn_graph = (
            three_digit_part
            + ssn_separator
            + two_digit_part
            + ssn_separator
            + four_digit_part
        )

        graph |= (
            (
                insert('country_code: "')
                + ssn_prompts
                + insert('"')
                + self.DELETE_EXTRA_SPACE
            ).ques
            + insert('number_part: "')  # noqa
            + ssn_graph.optimize()  # noqa
            + insert('"')  # noqa
        )

        final_graph = self.add_tokens(graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing telephone numbers, e.g.
            telephone { country_code: "one" number_part: "one two three, one two three, five six seven eight" extension: "one" }
            -> one, one two three, one two three, five six seven eight, one
        """
        optional_country_code = (
            delete('country_code: "')
            + self.NOT_QUOTE.plus
            + delete('"')
            + self.DELETE_SPACE
            + insert(" ")
        ).ques

        number_part = (
            delete('number_part: "')
            + self.NOT_QUOTE.plus
            + add_weight(delete(" "), -0.0001).ques
            + delete('"')
        )

        optional_extension = (
            self.DELETE_SPACE
            + insert(" ")
            + delete('extension: "')
            + self.NOT_QUOTE.plus
            + delete('"')
        ).ques

        graph = optional_country_code + number_part + optional_extension
        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
