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

from pynini import closure, cross, invert, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Electronic(Processor):

    def __init__(self):
        super().__init__(name="electronic", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        symbols = invert(string_file(get_abs_path("../itn/english/data/electronic/symbols.tsv")))

        char = self.ALPHA | digit | zero
        word = add_weight(closure(self.ALPHA, 2), 0.1)
        token = char | symbols | word
        component = token + closure(ds + token)

        dot = cross("dot", ".")
        domain = component + (ds + dot + ds + component).plus

        username = insert('username: "') + component + insert('"')
        domain_field = insert('domain: "') + domain + insert('"')

        # Email: X at Y dot Z (requires "at" keyword)
        graph_email = username + ds + delete("at") + ds + insert(" ") + domain_field

        # URL: requires protocol or www prefix
        http = cross("h t t p", "http")
        https = cross("h t t p s", "https")
        protocol = (http | https) + cross(" colon slash slash ", "://")
        www = cross("w w w", "www")

        # protocol + [www.] + domain
        url_with_protocol = protocol + closure(www + ds + dot + ds, 0, 1) + domain
        # www. + domain (no protocol)
        url_with_www = www + ds + dot + ds + domain
        # domain only (must have dot): nvidia dot com
        url_domain_only = domain

        graph_url = insert('protocol: "') + (url_with_protocol | url_with_www | url_domain_only) + insert('"')

        final_graph = graph_email | graph_url
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        username = delete("username:") + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE.plus + delete('"')
        domain = delete("domain:") + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE.plus + delete('"')
        protocol = delete("protocol:") + self.DELETE_SPACE + delete('"') + self.NOT_QUOTE.plus + delete('"')

        graph_email = username + self.DELETE_SPACE + insert("@") + domain
        graph_url = protocol

        self.verbalizer = self.delete_tokens(graph_email | graph_url)
