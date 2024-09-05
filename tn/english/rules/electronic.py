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

from pynini import (
    accep,
    cdrewrite,
    cross,
    compose,
    difference,
    invert,
    project,
    string_file,
    union,
)
from pynini.lib.pynutil import add_weight, delete, insert
from pynini.examples import plurals

from tn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Electronic(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("electronic", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying electronic: as URLs, email addresses, etc.
            e.g. cdf1@abc.edu -> tokens { electronic { username: "cdf one" domain: "abc.edu" } }
        """
        cardinal = Cardinal(self.deterministic)
        if self.deterministic:
            numbers = self.DIGIT
        else:
            numbers = insert(" ") + cardinal.long_numbers + insert(" ")

        accepted_symbols = project(
            string_file(get_abs_path("english/data/electronic/symbol.tsv")),
            "input",
        )
        accepted_common_domains = project(
            string_file(get_abs_path("english/data/electronic/domain.tsv")),
            "input",
        )

        dict_words = add_weight(
            string_file(get_abs_path("english/data/electronic/words.tsv")),
            -0.0001,
        )

        dict_words_without_delimiter = (
            dict_words + add_weight(insert(" ") + dict_words, -0.0001).plus
        )
        dict_words_graph = dict_words_without_delimiter | dict_words

        all_accepted_symbols_start = (
            dict_words_graph | self.ALPHA.star | accepted_symbols
        ).optimize()

        all_accepted_symbols_end = (
            dict_words_graph | numbers | self.ALPHA.star | accepted_symbols
        ).optimize()

        graph_symbols = string_file(
            get_abs_path("english/data/electronic/symbol.tsv")
        ).optimize()
        username = (self.ALPHA | dict_words_graph) + (
            self.ALPHA | numbers | accepted_symbols | dict_words_graph
        ).star

        username = insert('username: "') + username + insert('"') + cross("@", " ")

        domain_graph = (
            all_accepted_symbols_start
            + (
                all_accepted_symbols_end | add_weight(accepted_common_domains, -0.0001)
            ).star
        )

        protocol_symbols = ((graph_symbols | cross(":", "colon")) + insert(" ")).star
        protocol_start = (cross("https", "HTTPS ") | cross("http", "HTTP ")) + (
            accep("://") @ protocol_symbols
        )
        protocol_file_start = (
            accep("file") + insert(" ") + (accep(":///") @ protocol_symbols)
        )

        protocol_end = add_weight(
            cross("www", "WWW ") + accep(".") @ protocol_symbols, -1000
        )
        protocol = (
            protocol_file_start
            | protocol_start
            | protocol_end
            | (protocol_start + protocol_end)
        )

        domain_graph_with_class_tags = (
            insert('domain: "')
            + compose(
                self.ALPHA
                + self.NOT_SPACE.star
                + (self.ALPHA | self.DIGIT | accep("/")),
                domain_graph,
            ).optimize()
            + insert('"')
        )

        protocol = insert('protocol: "') + add_weight(protocol, -0.0001) + insert('"')
        # email
        graph = compose(
            self.VSIGMA + accep("@") + self.VSIGMA + accep(".") + self.VSIGMA,
            username + domain_graph_with_class_tags,
        )

        # abc.com, abc.com/123-sm
        # when only domain, make sure it starts and end with self.ALPHA
        graph |= (
            insert('domain: "')
            + compose(
                self.ALPHA
                + self.NOT_SPACE.star
                + accepted_common_domains
                + self.NOT_SPACE.star,
                domain_graph,
            ).optimize()
            + insert('"')
        )
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + insert(" ") + domain_graph_with_class_tags

        final_graph = self.add_tokens(graph)

        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing electronic
            e.g. electronic { username: "cdf one" domain: "abc.edu" } -> cdf one at abc dot edu
        """
        graph_digit_no_zero = invert(
            string_file(get_abs_path("english/data/number/digit.tsv"))
        ).optimize()
        graph_zero = cross("0", "zero")
        long_numbers = add_weight(
            graph_digit_no_zero + cross("000", " thousand"), -0.0001
        )

        if not self.deterministic:
            graph_zero |= cross("0", "o") | cross("0", "oh")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = string_file(
            get_abs_path("english/data/electronic/symbol.tsv")
        ).optimize()

        NEMO_NOT_BRACKET = difference(self.VCHAR, union("{", "}")).optimize()
        dict_words = project(
            string_file(get_abs_path("english/data/electronic/words.tsv")),
            "output",
        )
        default_chars_symbols = cdrewrite(
            insert(" ") + (graph_symbols | graph_digit | long_numbers) + insert(" "),
            "",
            "",
            self.VSIGMA,
        )
        default_chars_symbols = compose(
            NEMO_NOT_BRACKET.star, default_chars_symbols.optimize()
        ).optimize()

        # this is far cases when user name was split by dictionary words, i.e. "sevicepart@ab.com" -> "service part"
        space_separated_dict_words = add_weight(
            self.ALPHA + (self.ALPHA | " ").star + " " + (self.ALPHA | " ").star,
            -0.0001,
        )

        user_name = (
            delete("username:")
            + self.DELETE_SPACE
            + delete('"')
            + (default_chars_symbols | space_separated_dict_words).optimize()
            + delete('"')
        )

        domain_common = string_file(get_abs_path("english/data/electronic/domain.tsv"))

        # this will be used for a safe fallback
        domain_all = compose(
            default_chars_symbols,
            (self.ALPHA | " " | add_weight(dict_words, -0.0001)).star,
        )

        domain = (
            domain_all
            + insert(" ")
            + plurals._priority_union(
                domain_common,
                add_weight(cross(".", "dot"), weight=0.0001),
                self.VSIGMA,
            )
            + (insert(" ") + default_chars_symbols).ques
        )

        domain = (
            delete("domain:")
            + self.DELETE_SPACE
            + delete('"')
            + (domain | add_weight(domain_all, weight=100)).optimize()
            + self.DELETE_SPACE
            + delete('"')
        ).optimize()

        protocol = delete('protocol: "') + self.NOT_QUOTE.plus + delete('"')
        graph = (
            (protocol + self.DELETE_SPACE).ques
            + (user_name + self.DELETE_SPACE + insert(" at ") + self.DELETE_SPACE).ques
            + domain
            + self.DELETE_SPACE
        ).optimize() @ cdrewrite(self.DELETE_EXTRA_SPACE, "", "", self.VSIGMA)

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
