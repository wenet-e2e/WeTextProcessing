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
from tn.english.rules.cardinal import Cardinal


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
            numbers = pynutil.insert(" ") + cardinal.long_numbers + pynutil.insert(" ")

        accepted_symbols = pynini.project(
            pynini.string_file(get_abs_path("english/data/electronic/symbol.tsv")),
            "input",
        )
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("english/data/electronic/domain.tsv")),
            "input",
        )

        dict_words = pynutil.add_weight(
            pynini.string_file(get_abs_path("english/data/electronic/words.tsv")),
            -0.0001,
        )

        dict_words_without_delimiter = (
            dict_words
            + pynutil.add_weight(pynutil.insert(" ") + dict_words, -0.0001).plus
        )
        dict_words_graph = dict_words_without_delimiter | dict_words

        all_accepted_symbols_start = (
            dict_words_graph | self.ALPHA.star | accepted_symbols
        ).optimize()

        all_accepted_symbols_end = (
            dict_words_graph | numbers | self.ALPHA.star | accepted_symbols
        ).optimize()

        graph_symbols = pynini.string_file(
            get_abs_path("english/data/electronic/symbol.tsv")
        ).optimize()
        username = (self.ALPHA | dict_words_graph) + (
            self.ALPHA | numbers | accepted_symbols | dict_words_graph
        ).star

        username = (
            pynutil.insert('username: "')
            + username
            + pynutil.insert('"')
            + pynini.cross("@", " ")
        )

        domain_graph = (
            all_accepted_symbols_start
            + (
                all_accepted_symbols_end
                | pynutil.add_weight(accepted_common_domains, -0.0001)
            ).star
        )

        protocol_symbols = (
            (graph_symbols | pynini.cross(":", "colon")) + pynutil.insert(" ")
        ).star
        protocol_start = (
            pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")
        ) + (pynini.accep("://") @ protocol_symbols)
        protocol_file_start = (
            pynini.accep("file")
            + self.INSERT_SPACE
            + (pynini.accep(":///") @ protocol_symbols)
        )

        protocol_end = pynutil.add_weight(
            pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols, -1000
        )
        protocol = (
            protocol_file_start
            | protocol_start
            | protocol_end
            | (protocol_start + protocol_end)
        )

        domain_graph_with_class_tags = (
            pynutil.insert('domain: "')
            + pynini.compose(
                self.ALPHA
                + self.NOT_SPACE.star
                + (self.ALPHA | self.DIGIT | pynini.accep("/")),
                domain_graph,
            ).optimize()
            + pynutil.insert('"')
        )

        protocol = (
            pynutil.insert('protocol: "')
            + pynutil.add_weight(protocol, -0.0001)
            + pynutil.insert('"')
        )
        # email
        graph = pynini.compose(
            self.VCHAR.star
            + pynini.accep("@")
            + self.VCHAR.star
            + pynini.accep(".")
            + self.VCHAR.star,
            username + domain_graph_with_class_tags,
        )

        # abc.com, abc.com/123-sm
        # when only domain, make sure it starts and end with self.ALPHA
        graph |= (
            pynutil.insert('domain: "')
            + pynini.compose(
                self.ALPHA
                + self.NOT_SPACE.star
                + accepted_common_domains
                + self.NOT_SPACE.star,
                domain_graph,
            ).optimize()
            + pynutil.insert('"')
        )
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + pynutil.insert(" ") + domain_graph_with_class_tags

        final_graph = self.add_tokens(graph)

        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing electronic
            e.g. electronic { username: "cdf one" domain: "abc.edu" } -> cdf one at abc dot edu
        """
        graph_digit_no_zero = pynini.invert(
            pynini.string_file(get_abs_path("english/data/number/digit.tsv"))
        ).optimize()
        graph_zero = pynini.cross("0", "zero")
        long_numbers = pynutil.add_weight(
            graph_digit_no_zero + pynini.cross("000", " thousand"), -0.0001
        )

        if not self.deterministic:
            graph_zero |= pynini.cross("0", "o") | pynini.cross("0", "oh")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(
            get_abs_path("english/data/electronic/symbol.tsv")
        ).optimize()

        NEMO_NOT_BRACKET = pynini.difference(
            self.VCHAR, pynini.union("{", "}")
        ).optimize()
        dict_words = pynini.project(
            pynini.string_file(get_abs_path("english/data/electronic/words.tsv")),
            "output",
        )
        default_chars_symbols = pynini.cdrewrite(
            pynutil.insert(" ")
            + (graph_symbols | graph_digit | long_numbers)
            + pynutil.insert(" "),
            "",
            "",
            self.VCHAR.star,
        )
        default_chars_symbols = pynini.compose(
            NEMO_NOT_BRACKET.star, default_chars_symbols.optimize()
        ).optimize()

        # this is far cases when user name was split by dictionary words, i.e. "sevicepart@ab.com" -> "service part"
        space_separated_dict_words = pynutil.add_weight(
            self.ALPHA + (self.ALPHA | " ").star + " " + (self.ALPHA | " ").star,
            -0.0001,
        )

        user_name = (
            pynutil.delete("username:")
            + self.DELETE_SPACE
            + pynutil.delete('"')
            + (default_chars_symbols | space_separated_dict_words).optimize()
            + pynutil.delete('"')
        )

        domain_common = pynini.string_file(
            get_abs_path("english/data/electronic/domain.tsv")
        )

        # this will be used for a safe fallback
        domain_all = pynini.compose(
            default_chars_symbols,
            (self.ALPHA | " " | pynutil.add_weight(dict_words, -0.0001)).star,
        )

        domain = (
            domain_all
            + self.INSERT_SPACE
            + plurals._priority_union(
                domain_common,
                pynutil.add_weight(pynini.cross(".", "dot"), weight=0.0001),
                self.VCHAR.star,
            )
            + (self.INSERT_SPACE + default_chars_symbols).ques
        )

        domain = (
            pynutil.delete("domain:")
            + self.DELETE_SPACE
            + pynutil.delete('"')
            + (domain | pynutil.add_weight(domain_all, weight=100)).optimize()
            + self.DELETE_SPACE
            + pynutil.delete('"')
        ).optimize()

        protocol = (
            pynutil.delete('protocol: "') + self.NOT_QUOTE.plus + pynutil.delete('"')
        )
        graph = (
            (protocol + self.DELETE_SPACE).ques
            + (
                user_name
                + self.DELETE_SPACE
                + pynutil.insert(" at ")
                + self.DELETE_SPACE
            ).ques
            + domain
            + self.DELETE_SPACE
        ).optimize() @ pynini.cdrewrite(
            self.DELETE_EXTRA_SPACE, "", "", self.VCHAR.star
        )

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
