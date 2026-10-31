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

from itn.spanish.rules.utils import ALPHA, cased, data_file
from tn.processor import Processor

TOKEN_ORDER = {"electronic": ["username", "domain", "protocol"]}


class Electronic(Processor):
    """Raw spelled e-mail addresses and URLs; V writes them.

    "a b c arroba g mail punto com" is tagged as username "a b c" and domain
    "g mail punto com"; URLs ("doble ve doble ve doble ve punto ...") are one
    raw "protocol" field.
    """

    def __init__(self, cased_input=False):
        super().__init__("electronic", ordertype="itn", token_orders=TOKEN_ORDER)
        delete_space = pynutil.delete(" ")
        num = data_file("numbers/digit.tsv") | data_file("numbers/zero.tsv")
        alpha_num = (ALPHA | num).optimize()
        symbols = pynini.invert(data_file("electronic/symbols.tsv")).optimize()
        accepted = (alpha_num | symbols).optimize()
        dot = pynini.cross("punto", ".")
        alternative_dot = pynini.closure(delete_space, 0, 1) + "." + pynini.closure(delete_space, 0, 1)

        username = alpha_num + delete_space + pynini.closure(accepted + delete_space) + alpha_num
        single_alphanum = pynini.closure(alpha_num + delete_space) + alpha_num
        servers = pynini.invert(data_file("electronic/server_name.tsv"))
        server = single_alphanum | servers | pynini.closure(ALPHA, 2)
        domains = pynini.invert(data_file("electronic/domain.tsv"))
        domain = pynutil.add_weight(single_alphanum, -0.0001) | domains | pynini.closure(ALPHA, 2)
        domain_graph = server + (delete_space + dot + delete_space | alternative_dot) + domain

        www = pynini.union("www", "w w w", "doble ve doble ve doble ve")
        http = pynini.union("http", "h t t p", "hache te te pe")
        https = pynini.union("https", "h t t p s", "hache te te pe ese")
        protocol_end = pynutil.add_weight(pynini.cross(www, "www"), 0.0001)
        protocol_start = pynutil.add_weight(pynini.cross(http, "http") | pynini.cross(https, "https"), 0.0001)
        protocol_start += pynini.cross(" dos puntos barra barra ", "://")
        ending = (
            delete_space + symbols + delete_space
            + (domain | pynini.closure(accepted + delete_space) + accepted)
        )
        protocol_default = (
            (
                pynini.closure(delete_space + accepted, 1)
                | server
                | pynutil.add_weight(pynini.closure(ALPHA, 1), 0.001)
            )
            + pynini.closure(ending, 1)
        )
        protocol = (
            pynini.closure(protocol_start, 0, 1) + protocol_end + delete_space + dot + delete_space
            + protocol_default
        )
        if cased_input:
            protocol |= pynini.closure(protocol_start, 0, 1) + protocol_end + alternative_dot + protocol_default
        protocol |= pynini.closure(protocol_end + delete_space + dot, 0, 1) + protocol_default

        self.username_graph = cased(username.optimize(), cased_input)
        self.domain_graph = cased(domain_graph.optimize(), cased_input)
        self.protocol_graph = cased(protocol.optimize(), cased_input)
        at = cased(pynini.accep("arroba"), cased_input)

        email_t = (
            self.tag_field("username", self.username_graph)
            + pynutil.delete(" ") + pynutil.delete(at) + pynini.accep(" ")
            + self.tag_field("domain", self.domain_graph)
        )
        self.tagger = self.add_tokens(email_t | self.tag_field("protocol", self.protocol_graph))

        email_v = (
            self.verbalize_field("username", self.username_graph)
            + pynutil.insert("@")
            + self.verbalize_field("domain", self.domain_graph, leading_space=True)
        )
        self.verbalizer = self.delete_tokens(email_v | self.verbalize_field("protocol", self.protocol_graph))
