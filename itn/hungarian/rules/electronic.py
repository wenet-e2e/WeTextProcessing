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
from pynini.lib.pynutil import delete

from itn.hungarian.rules.utils import SIGMA, data_keys, data_labels, data_map
from tn.processor import Processor

TOKEN_ORDER = {}


class Electronic(Processor):
    """Spelled e-mails, URLs and handles, e.g. ``a b c kukac d e f pont hu`` -> ``abc@def.hu``.

    Raw spoken ``protocol``/``username``/``domain`` fields; the verbalizer
    joins letters and reads digit and symbol names (``pont`` -> ``.``).
    """

    def __init__(self):
        super().__init__("electronic", ordertype="itn", token_orders=TOKEN_ORDER)
        symbols = data_map("electronic/symbols.tsv", invert=True)
        digits = data_map("number/digit.tsv") | data_map("number/zero.tsv")
        letter = byte.ALPHA
        char = (letter | digits | symbols).optimize()
        servers = data_keys("electronic/server_name.tsv")
        tlds = pynini.union(*[row[0][1:] for row in data_labels("electronic/domain.tsv")])
        name = (char | servers | tlds).optimize()
        dot = pynini.cross("pont", ".")
        joint = delete(" ")

        def spell(piece, minimum=1):
            return (piece + pynini.closure(joint + piece, minimum - 1)).optimize()

        piece = pynini.difference(pynini.project(name, "input"), "pont") @ name
        domain = spell(piece) + (joint + dot + joint + spell(piece)).plus
        self.domain = domain.optimize()
        self.username = spell(char)
        self.handle = spell(char, 2)
        protocol_start = spell(letter) @ pynini.union("http", "https") + joint + pynini.cross("kettőspont", ":")
        protocol_start += (joint + pynini.cross("perjel", "/")) ** 2
        www = pynini.cross("w w w", "www") + joint + dot
        self.protocol = (protocol_start | www | protocol_start + joint + www).optimize()
        # A bare domain must end in a known top-level domain.
        site = (SIGMA + " pont " + tlds) @ self.domain

        space = pynini.accep(" ")
        domain_t = self.tag_field("domain", self.domain)
        email_t = self.tag_field("username", self.username) + delete(" kukac") + space + domain_t
        handle_t = delete("kukac ") + self.tag_field("username", self.handle)
        url_t = self.tag_field("protocol", self.protocol) + space + domain_t
        self.tagger = self.add_tokens(self.tag_field("domain", site) | email_t | handle_t | url_t)

        domain_v = self.verbalize_field("domain", self.domain)
        username_v = self.verbalize_field("username", self.username)
        email_v = username_v + pynini.cross(" ", "@") + domain_v
        handle_v = pynini.cross("", "@") + username_v
        url_v = self.verbalize_field("protocol", self.protocol) + delete(" ") + domain_v
        self.verbalizer = self.delete_tokens(domain_v | email_v | handle_v | url_v)
