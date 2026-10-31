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

from itn.swedish.rules.utils import rows, spoken_table, table
from tn.processor import Processor

TOKEN_ORDER = {}


class Electronic(Processor):
    """Raw spelled-out e-mail addresses and URLs to their written form."""

    def __init__(self):
        super().__init__(name="electronic", ordertype="itn", token_orders=TOKEN_ORDER)
        symbols = table("electronic/symbols")
        symbol_chars = pynini.union(*[row[0] for row in rows("electronic/symbols")])
        digits = spoken_table("numbers/digit") | spoken_table("numbers/zero")
        tokens = (
            self.ALPHA | digits | symbols | table("electronic/server_name") | table("electronic/domain")
            | pynini.cross("dubbla snedstreck", "//")
        )
        spelled = (tokens + (pynutil.delete(" ") + tokens).star).optimize()

        dot = pynini.accep(".")
        chars = (self.ALPHA | self.DIGIT | (symbol_chars - dot)).star
        username = (self.ALPHA | self.DIGIT | symbol_chars).plus
        common = pynini.union(*[row[0] for row in rows("electronic/domain")])
        domain = chars + dot + chars
        common_domain = chars + common + ((symbol_chars | dot) + chars.plus).ques
        start = pynini.union("https://", "http://")
        protocol = start | "www." | start + "www."

        self.username = (spelled @ username).optimize()
        self.domain = (spelled @ (domain | common_domain)).optimize()
        self.protocol = (spelled @ protocol).optimize()
        email_domain = (spelled @ domain).optimize()

        space = pynini.accep(" ")
        email_t = (
            self.tag_field("username", self.username) + space + pynutil.delete("snabel-a ")
            + self.tag_field("domain", email_domain)
        )
        url_t = self.tag_field("protocol", self.protocol) + space + self.tag_field("domain", self.domain)
        common_t = self.tag_field("domain", spelled @ common_domain)
        self.tagger = self.add_tokens(email_t | url_t | common_t)

        join = pynutil.delete(" ")
        verbalizer = (
            self.verbalize_field("username", self.username) + join + pynutil.insert("@")
            + self.verbalize_field("domain", self.domain)
            | self.verbalize_field("protocol", self.protocol) + join + self.verbalize_field("domain", self.domain)
            | self.verbalize_field("domain", self.domain)
        )
        self.verbalizer = self.delete_tokens(verbalizer)
