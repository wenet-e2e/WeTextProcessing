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
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {}


class Electronic(Processor):
    """Raw spelled e-mail addresses and URLs.

    "c d f arroba a b c ponto e d u" -> username "c d f", domain "a b c ponto e d u"
    -> cdf@abc.edu; "dáblio dáblio dáblio ponto nvidia ponto com" -> www.nvidia.com.
    """

    def __init__(self, cardinal):
        super().__init__("electronic", ordertype="itn", token_orders=TOKEN_ORDER)

        def data(name):
            return pynini.string_file(get_abs_path("portuguese/data/electronic/" + name))

        sep = delete(" ")
        alpha_num = (self.ALPHA | cardinal.digit | cardinal.zero).optimize()
        symbols = data("symbols.tsv").invert()
        accepted = (alpha_num | symbols).optimize()
        dot = pynini.cross("ponto", ".")
        username = alpha_num + sep + (accepted + sep).star + alpha_num
        spelled = (alpha_num + sep).star + alpha_num
        server = spelled | data("server_name.tsv").invert()
        domain = spelled | data("domain.tsv").invert()
        self.username_graph = username.optimize()
        self.domain_graph = (server + sep + dot + sep + domain).optimize()

        www = pynini.cross(pynini.union("www", "w w w", "dáblio dáblio dáblio"), "www")
        http = pynini.cross(pynini.union("http", "h t t p", "agá tê tê pê"), "http")
        https = pynini.cross(pynini.union("https", "h t t p s", "agá tê tê pê ésse"), "https")
        start = (http | https) + pynini.cross(" dois pontos barra barra ", "://")
        ending = sep + symbols + sep + (domain | (accepted + sep).star + accepted)
        self.protocol_graph = (
            start.ques + www + sep + dot + sep + ((sep + accepted).plus | server) + ending.plus
        ).optimize()

        email_t = (
            self.tag_field("username", self.username_graph) + sep + delete("arroba") + sep + insert(" ")
            + self.tag_field("domain", self.domain_graph)
        )
        self.tagger = self.add_tokens(email_t | self.tag_field("protocol", self.protocol_graph))
        email_v = (
            self.verbalize_field("username", self.username_graph) + insert("@")
            + self.verbalize_field("domain", self.domain_graph, leading_space=True)
        )
        self.verbalizer = self.delete_tokens(email_v | self.verbalize_field("protocol", self.protocol_graph))
