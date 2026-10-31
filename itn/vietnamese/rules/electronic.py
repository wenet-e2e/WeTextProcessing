# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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


def electronic_file(name):
    return pynini.string_file(get_abs_path("vietnamese/data/electronic/" + name + ".tsv")).optimize()


class Electronic(Processor):
    """Spelled e-mail addresses and URLs keep their words raw: ``a b c a móc g mail chấm com`` -> ``abc@gmail.com``."""

    def __init__(self, cardinal):
        super().__init__(name="electronic", ordertype="itn", token_orders={})
        space = delete(" ")
        alpha_num = (self.ALPHA | cardinal.digit | cardinal.zero).optimize()
        symbols = electronic_file("symbols").invert()
        url_symbols = electronic_file("url_symbols").invert()
        url_chars = alpha_num | url_symbols
        dot = pynini.cross("chấm", ".")
        letters = pynini.closure(alpha_num + space) + alpha_num
        server = letters | electronic_file("server_name") | pynini.closure(self.ALPHA, 2)
        domain = letters | electronic_file("domain") | pynini.closure(self.ALPHA, 2)

        self.username_graph = (alpha_num + pynini.closure(space + (alpha_num | symbols))).optimize()
        multi_domain = pynini.closure(dot + space + domain + space) + dot + space + domain
        self.domain_graph = (server + space + multi_domain).optimize()
        protocol_start = pynini.cross("h t t p", "http") | pynini.cross("h t t p s", "https")
        protocol_start += pynini.cross(" hai chấm sẹc sẹc ", "://")
        www = pynini.cross(pynini.union("w w w", "www"), "www") + space + dot + space
        ending = space + url_symbols + space + (domain | pynini.closure(url_chars + space) + url_chars)
        self.protocol_graph = (
            protocol_start.ques + www.ques + server + space + dot + space + domain + ending.star
        ).optimize()

        at = space + delete(pynini.union("a còng", "a móc", "a vòng")) + insert(" ") + space
        email = self.tag_field("username", self.username_graph) + at + self.tag_field("domain", self.domain_graph)
        self.tagger = self.add_tokens(email | self.tag_field("protocol", self.protocol_graph))

        email_v = self.verbalize_field("username", self.username_graph) + delete(" ") + insert("@")
        email_v += self.verbalize_field("domain", self.domain_graph)
        self.verbalizer = self.delete_tokens(email_v | self.verbalize_field("protocol", self.protocol_graph))
