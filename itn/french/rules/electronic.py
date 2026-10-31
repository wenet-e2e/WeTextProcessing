# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from itn.french.rules.cardinal import load
from tn.processor import Processor


TOKEN_ORDER = {"electronic": ["username", "domain", "protocol"]}


class Electronic(Processor):
    """Spelled e-mail addresses and URLs.

    ``a b c arobase g mail point com`` -> ``electronic { username: "a b c" domain: "g mail point com" }``
    -> ``abc@gmail.com``.
    """

    def __init__(self):
        super().__init__("electronic", ordertype="itn", token_orders=TOKEN_ORDER)
        space = delete(" ")
        alpha_num = self.ALPHA | load("numbers/digit") | load("numbers/zero")
        symbols = load("electronic/symbols")
        at = pynini.union("arobase", "chez", "at", "à")
        accepted = alpha_num | symbols
        dot = pynini.cross("point", ".")
        username = (alpha_num + space + (accepted + space).star + alpha_num).optimize()
        letters = (alpha_num + space).star + alpha_num
        server = letters | load("electronic/server_name")
        domain = (server + space + dot + space + (letters | load("electronic/domain"))).optimize()

        www = pynini.cross(pynini.union("www", "w w w", "double vé double vé double vé"), "www")
        http = pynini.cross(pynini.union("http", "h t t p", "ache té té pé"), "http")
        http |= pynini.cross(pynini.union("https", "h t t p s", "ache té té pé esse"), "https")
        http += pynini.cross(
            pynini.union(
                " deux-points barre oblique barre oblique ",
                " deux-points barre barre ",
                " deux-points double barre ",
                " deux-points slash slash ",
            ),
            "://",
        )
        ending = space + symbols + space + (letters | load("electronic/domain") | (accepted + space).star + accepted)
        protocol = http.ques + www + space + dot + space + ((space + accepted).plus | server) + ending.plus
        protocol = protocol.optimize()

        email = self.tag_field("username", username) + space + delete(at) + space + insert(" ")
        email += self.tag_field("domain", domain)
        self.tagger = self.add_tokens(email | self.tag_field("protocol", protocol))
        email = self.verbalize_field("username", username) + delete(" ") + insert("@")
        email += self.verbalize_field("domain", domain)
        self.verbalizer = self.delete_tokens(email | self.verbalize_field("protocol", protocol))
