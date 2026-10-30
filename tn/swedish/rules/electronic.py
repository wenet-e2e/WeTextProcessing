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
from tn.swedish.rules.utils import BYTE_SIGMA, labels, tsv

TOKEN_ORDER = {}


def table(name):
    return pynini.string_map([row for row in labels(name) if row])


class Electronic(Processor):
    """Raw e-mail addresses and URLs, spelled out by the verbalizer."""

    def __init__(self, deterministic=True):
        super().__init__("electronic", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        symbols = table("electronic/symbols")
        domains = table("electronic/domain")
        servers = table("electronic/server_name")
        digits = tsv("numbers/digit").invert() | tsv("numbers/zero").invert()

        dot = pynini.accep(".")
        symbol_chars = self.input_projection(symbols) - dot
        chars = (self.ALPHA | self.DIGIT | symbol_chars).star
        chars_with_dot = (self.ALPHA | self.DIGIT | symbol_chars | dot).star
        common = self.input_projection(domains)
        username = chars_with_dot
        domain = chars + dot + chars
        common_domain = chars + common + ((symbol_chars | dot) + chars.plus).ques
        protocol_start = pynini.union("https://", "http://")
        protocol = protocol_start | "www." | protocol_start + "www."

        expand = pynini.cdrewrite(symbols | digits, "", "", BYTE_SIGMA)
        char = pynini.difference(self.VCHAR, pynini.union('"', " "))
        spell = (char + pynutil.insert(" ")).star + char
        self.username_graph = (username @ spell @ expand).optimize()
        double_slash = pynini.cdrewrite(
            pynini.cross("snedstreck snedstreck", "dubbla snedstreck"),
            pynini.union("[BOS]", " "), pynini.union("[EOS]", " "), BYTE_SIGMA)
        spell_symbols = pynini.cdrewrite(symbols, "", "", BYTE_SIGMA)
        self.protocol_graph = (protocol @ spell @ spell_symbols @ double_slash).optimize()
        default = pynutil.add_weight(char, 0.0001) | domains | servers
        self.domain_graph = ((domain | common_domain) @ (default + (pynutil.insert(" ") + default).star)
                             @ expand).optimize()

        domain_t = self.tag_field("domain", self.domain_graph)
        domain_v = self.verbalize_field("domain", self.domain_graph)
        email_t = self.tag_field("username", self.username_graph) + pynutil.delete("@")
        email_t += pynutil.insert(" ") + self.tag_field("domain", (domain @ self.domain_graph).optimize())
        email_v = self.verbalize_field("username", self.username_graph) + pynutil.insert(" snabel-a")
        email_v += pynutil.delete(" ") + pynutil.insert(" ") + domain_v
        url_t = self.tag_field("protocol", self.protocol_graph) + pynutil.insert(" ") + domain_t
        url_v = self.verbalize_field("protocol", self.protocol_graph) + pynutil.delete(" ") + pynutil.insert(" ")
        url_v += domain_v
        common_t = self.tag_field("domain", (common_domain @ self.domain_graph).optimize())
        self.tagger = self.add_tokens(email_t | common_t | url_t)
        self.verbalizer = self.delete_tokens(email_v | domain_v | url_v)
