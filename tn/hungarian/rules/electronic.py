# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import byte
from pynini.lib.pynutil import add_weight, delete, insert

from tn.hungarian.rules.utils import data_keys, data_map
from tn.processor import Processor

TOKEN_ORDER = {"electronic": ["protocol", "username", "domain"]}


class Electronic(Processor):
    """E-mails, URLs and handles, e.g. ``abc@def.hu`` -> ``a b c kukac d e f pont hu``.

    Raw ``protocol``/``username``/``domain`` fields; the verbalizer spells
    them, reading symbols, digits, known servers and domains as words.
    """

    def __init__(self):
        super().__init__("electronic", token_orders=TOKEN_ORDER)
        symbols = data_map("electronic/symbols.tsv")
        symbol_chars = data_keys("electronic/symbols.tsv")
        alnum = byte.ALPHA | byte.DIGIT
        accepted = (alnum | pynini.difference(symbol_chars, ".")).plus
        all_chars = (alnum | symbol_chars).plus
        domain = accepted + ("." + accepted).plus
        protocol_start = pynini.union("https://", "http://")
        protocol = protocol_start | "www." | protocol_start + "www."

        digits = data_map("number/digit.tsv", invert=True) | data_map("number/zero.tsv", invert=True)
        spelled_char = symbols | digits | byte.ALPHA

        def spell(graph):
            return (graph @ (spelled_char + (insert(" ") + spelled_char).star)).optimize()

        known = data_map("electronic/domain.tsv") | data_keys("electronic/server_name.tsv")
        piece = add_weight(spelled_char, 0.0001) | known
        self.domain = (domain @ (piece + (insert(" ") + piece).star)).optimize()
        self.username = spell(all_chars)
        self.handle = spell(accepted + ("." + accepted).star)
        self.protocol = spell(protocol)

        domain_t = self.tag_field("domain", self.domain)
        email_t = self.tag_field("username", self.username) + pynini.cross("@", " ") + domain_t
        handle_t = delete("@") + self.tag_field("username", self.handle)
        url_t = self.tag_field("protocol", self.protocol) + insert(" ") + domain_t
        self.tagger = self.add_tokens(domain_t | email_t | handle_t | url_t)

        domain_v = self.verbalize_field("domain", self.domain)
        username_v = self.verbalize_field("username", self.username)
        email_v = username_v + pynini.cross(" ", " kukac ") + domain_v
        handle_v = insert("kukac ") + username_v
        url_v = self.verbalize_field("protocol", self.protocol) + pynini.accep(" ") + domain_v
        self.verbalizer = self.delete_tokens(domain_v | email_v | handle_v | url_v)
