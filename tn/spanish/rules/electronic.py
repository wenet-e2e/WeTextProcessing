# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.spanish.rules.utils import ALPHA, CHAR, DIGIT, ONES, SIGMA, data_file, rewrite

TOKEN_ORDER = {"electronic": ["protocol", "username", "domain"]}


class Electronic(Processor):
    """Raw e-mail, URL and handle parts; V spells them out."""

    def __init__(self, deterministic=True):
        super().__init__("electronic", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        symbols = data_file("electronic/symbols.tsv")
        symbol_chars = pynini.project(symbols, "input").optimize()
        accepted = pynini.closure(ALPHA | DIGIT | pynini.difference(symbol_chars, "."), 1)
        domain = accepted + pynini.closure("." + accepted, 1)
        username = pynini.closure(ALPHA | DIGIT | symbol_chars, 1)
        handle = accepted + pynini.closure("." + accepted)
        protocol_start = pynini.union("https://", "http://")
        protocol = protocol_start | "www." | protocol_start + "www."

        digits = rewrite(pynini.invert(data_file("numbers/digit.tsv")), pynini.cross(ONES, "uno"))
        digits |= pynini.invert(data_file("numbers/zero.tsv"))
        expand = pynini.cdrewrite(symbols | digits, "", "", SIGMA)
        char = pynini.difference(CHAR, pynini.union('"', " "))

        def spell(graph):
            return (graph @ (pynini.closure(char + pynutil.insert(" ")) + char) @ expand).optimize()

        self.username_graph = spell(username)
        self.protocol_graph = spell(protocol)
        # Known server names and top-level domains are read as words.
        default = pynutil.add_weight(char, 0.0001) | data_file("electronic/domain.tsv")
        default |= data_file("electronic/server_name.tsv")
        self.domain_graph = (domain @ (default + pynini.closure(pynutil.insert(" ") + default)) @ expand).optimize()

        space = pynutil.insert(" ")
        domain_t = self.tag_field("domain", self.domain_graph)
        email_t = self.tag_field("username", self.username_graph) + pynutil.delete("@") + space + domain_t
        handle_t = pynutil.delete("@") + self.tag_field("username", handle @ self.username_graph)
        url_t = self.tag_field("protocol", self.protocol_graph) + space + domain_t
        self.tagger = self.add_tokens(domain_t | email_t | handle_t | url_t)

        domain_v = self.verbalize_field("domain", self.domain_graph)
        username_v = self.verbalize_field("username", self.username_graph)
        email_v = username_v + pynini.cross(" ", " arroba ") + domain_v
        handle_v = pynutil.insert("arroba ") + username_v
        url_v = self.verbalize_field("protocol", self.protocol_graph) + pynini.accep(" ") + domain_v
        self.verbalizer = self.delete_tokens(domain_v | email_v | handle_v | url_v)
