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
from pynini.examples import plurals
from pynini.lib import pynutil

from tn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Electronic(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None):
        super().__init__("electronic", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying electronic: as URLs, email addresses, etc.
            e.g. cdf1@abc.edu -> electronic { username: "cdf1" domain: "abc.edu" }
        """
        cardinal = self.cardinal
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

        dict_words_without_delimiter = dict_words + pynutil.add_weight(pynutil.insert(" ") + dict_words, -0.0001).plus
        dict_words_graph = dict_words_without_delimiter | dict_words

        all_accepted_symbols_start = (dict_words_graph | self.ALPHA.star | accepted_symbols).optimize()

        all_accepted_symbols_end = (dict_words_graph | numbers | self.ALPHA.star | accepted_symbols).optimize()

        graph_symbols = pynini.string_file(get_abs_path("english/data/electronic/symbol.tsv")).optimize()
        username = (self.ALPHA | dict_words_graph) + (self.ALPHA | numbers | accepted_symbols | dict_words_graph).star

        self.username_graph = username
        username = self.tag_field("username", self.username_graph) + pynini.cross("@", " ")

        domain_graph = (all_accepted_symbols_start +
                        (all_accepted_symbols_end | pynutil.add_weight(accepted_common_domains, -0.0001)).star)

        protocol_symbols = ((graph_symbols | pynini.cross(":", "colon")) + pynutil.insert(" ")).star
        protocol_start = (pynini.cross("https", "HTTPS ")
                          | pynini.cross("http", "HTTP ")) + (pynini.accep("://") @ protocol_symbols)
        protocol_file_start = pynini.accep("file") + self.INSERT_SPACE + (pynini.accep(":///") @ protocol_symbols)

        protocol_end = pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols
        protocol_start_only = protocol_file_start | protocol_start
        protocol_start_and_end = protocol_start + protocol_end
        self.protocol_graph = protocol_start_only | protocol_end | protocol_start_and_end

        self.domain_graph = pynini.compose(
            self.ALPHA + self.NOT_SPACE.star + (self.ALPHA | self.DIGIT | pynini.accep("/")),
            domain_graph,
        ).optimize()
        domain_graph_with_class_tags = self.tag_field("domain", self.domain_graph)

        # These weights choose the protocol/domain structure, so they belong
        # outside tag_field. The field projection intentionally removes only
        # spoken-form weights.
        protocol = (pynutil.add_weight(self.tag_field("protocol", protocol_start_only), -0.0001)
                    | pynutil.add_weight(self.tag_field("protocol", protocol_end), -1000.0001)
                    | pynutil.add_weight(self.tag_field("protocol", protocol_start_and_end), -1000.0001))
        # email
        graph = pynini.compose(
            self.VCHAR.star + pynini.accep("@") + self.VCHAR.star + pynini.accep(".") + self.VCHAR.star,
            username + domain_graph_with_class_tags,
        )

        # abc.com, abc.com/123-sm
        # when only domain, make sure it starts and end with self.ALPHA
        graph |= (self.tag_field(
            "domain",
            pynini.compose(
                self.ALPHA + self.NOT_SPACE.star + accepted_common_domains + self.NOT_SPACE.star,
                domain_graph,
            ).optimize(),
        ))
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + pynutil.insert(" ") + domain_graph_with_class_tags

        final_graph = self.add_tokens(graph)

        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing electronic
            e.g. electronic { username: "cdf1" domain: "abc.edu" } -> cdf one at abc dot edu
        """
        graph_digit_no_zero = pynini.invert(pynini.string_file(get_abs_path("english/data/number/digit.tsv"))).optimize()
        graph_zero = pynini.cross("0", "zero")
        long_numbers = pynutil.add_weight(graph_digit_no_zero + pynini.cross("000", " thousand"), -0.0001)

        if not self.deterministic:
            graph_zero |= pynini.cross("0", "o") | pynini.cross("0", "oh")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("english/data/electronic/symbol.tsv")).optimize()

        NEMO_NOT_BRACKET = pynini.difference(self.VCHAR, pynini.union("{", "}")).optimize()
        dict_words = pynini.project(
            pynini.string_file(get_abs_path("english/data/electronic/words.tsv")),
            "output",
        )
        default_chars_symbols = pynini.cdrewrite(
            pynutil.insert(" ") + (graph_symbols | graph_digit | long_numbers) + pynutil.insert(" "),
            "",
            "",
            self.VCHAR.star,
        )
        default_chars_symbols = pynini.compose(NEMO_NOT_BRACKET.star, default_chars_symbols.optimize()).optimize()

        # this is far cases when user name was split by dictionary words, i.e. "sevicepart@ab.com" -> "service part"
        space_separated_dict_words = pynutil.add_weight(
            self.ALPHA + (self.ALPHA | " ").star + " " + (self.ALPHA | " ").star,
            -0.0001,
        )

        username_value = self.username_graph @ (default_chars_symbols | space_separated_dict_words).optimize()
        user_name = (pynutil.delete("username:") + self.DELETE_SPACE + pynutil.delete('"') + username_value +
                     pynutil.delete('"'))

        domain_common = pynini.string_file(get_abs_path("english/data/electronic/domain.tsv"))

        # Catch-all pronunciation for domains not covered by the common-domain lexicon.
        domain_all = pynini.compose(
            default_chars_symbols,
            (self.ALPHA | " " | pynutil.add_weight(dict_words, -0.0001)).star,
        )

        domain = (domain_all + self.INSERT_SPACE + plurals._priority_union(
            domain_common,
            pynutil.add_weight(pynini.cross(".", "dot"), weight=0.0001),
            self.VCHAR.star,
        ) + (self.INSERT_SPACE + default_chars_symbols).ques)

        domain_value = self.domain_graph @ (domain | pynutil.add_weight(domain_all, weight=100)).optimize()
        domain = (pynutil.delete("domain:") + self.DELETE_SPACE + pynutil.delete('"') + domain_value + self.DELETE_SPACE +
                  pynutil.delete('"')).optimize()

        protocol = pynutil.delete('protocol: "') + self.protocol_graph + pynutil.delete('"')
        graph = ((protocol + self.DELETE_SPACE).ques +
                 (user_name + self.DELETE_SPACE + pynutil.insert(" at ") + self.DELETE_SPACE).ques + domain +
                 self.DELETE_SPACE).optimize()

        # Canonicalize whitespace in the output lattice itself. A variable-
        # length cdrewrite can partition the same run in multiple ways, leaving
        # duplicate semantic candidates that differ only in whitespace.
        normalize_whitespace = (self.DELETE_SPACE + self.NOT_SPACE.plus +
                                (pynutil.delete(self.SPACE.plus) + self.INSERT_SPACE + self.NOT_SPACE.plus).star +
                                self.DELETE_SPACE)
        graph = (graph @ normalize_whitespace).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
