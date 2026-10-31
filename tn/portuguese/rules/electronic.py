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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


class Electronic(Processor):
    """Raw e-mail, URL and social handles; spelled letter by letter when verbalized.

    Known server names, domains and common words are read as whole words,
    e.g. test@gmail.com -> t e s t arroba gmail ponto com.
    """

    def __init__(self, cardinal):
        super().__init__("electronic", token_orders=TOKEN_ORDER)

        def data(name):
            return get_abs_path("portuguese/data/electronic/" + name)

        symbol_rows = load_labels(data("symbols.tsv"))
        symbols = pynini.string_map(symbol_rows)
        digit = pynini.string_map([(str(n), cardinal.digits[str(n)]) for n in range(1, 10)] + [("0", cardinal.zero)])
        servers = pynini.string_file(data("server_name.tsv"))
        domains = pynini.string_file(data("domain.tsv"))
        words = pynini.string_file(data("electronic_spoken_unit.tsv"))

        letters = self.ALPHA | self.DIGIT | pynini.union(*[row[0] for row in symbol_rows])
        name_chars = letters - "."
        domain_component = name_chars.plus + ("." + name_chars.plus).plus
        username = letters.plus
        handle = name_chars.plus + ("." + name_chars.plus).star
        protocol_start = pynini.union("https://", "http://")
        protocol = protocol_start | "www." | protocol_start + "www."

        # A spoken unit for one character: symbols and digits are read, other
        # characters are kept; whole known words win by having fewer units.
        char = add_weight(symbols | digit | (self.NOT_QUOTE - pynini.union(*[row[0] for row in symbol_rows]) - self.DIGIT), 0.0001)
        user_unit = char | servers | words
        domain_unit = char | domains | servers | words
        user_graph = (user_unit + (insert(" ") + user_unit).star).optimize()
        domain_graph = (domain_unit + (insert(" ") + domain_unit).star).optimize()
        protocol_graph = (char + (insert(" ") + char).star).optimize()

        user_graph = (username @ user_graph).optimize()
        handle_graph = (handle @ user_graph).optimize()
        domain_graph = (domain_component @ domain_graph).optimize()
        protocol_graph = (protocol @ protocol_graph).optimize()

        email_t = self.tag_field("username", user_graph) + delete("@") + insert(" ") + self.tag_field("domain", domain_graph)
        url_t = (self.tag_field("protocol", protocol_graph) + insert(" ")).ques + self.tag_field("domain", domain_graph)
        social_t = delete("@") + self.tag_field("handle", handle_graph)
        self.tagger = self.add_tokens(email_t | url_t | social_t)

        email_v = (
            self.verbalize_field("username", user_graph) + insert(" arroba ")
            + self.verbalize_field("domain", domain_graph, leading_space=True)
        )
        url_v = (
            (self.verbalize_field("protocol", protocol_graph) + delete(" ") + insert(" ")).ques
            + self.verbalize_field("domain", domain_graph)
        )
        social_v = insert("arroba ") + self.verbalize_field("handle", handle_graph)
        self.verbalizer = self.delete_tokens(email_v | url_v | social_v)
