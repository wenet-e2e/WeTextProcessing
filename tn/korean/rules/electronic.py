# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Electronic(Processor):

    def __init__(self, cardinal):
        super().__init__("electronic", token_orders={})
        self.cardinal = cardinal
        symbols = pynini.string_file(get_abs_path("korean/data/electronic/symbol.tsv"))
        domains = pynini.string_file(get_abs_path("korean/data/electronic/domain.tsv"))
        extensions = pynini.string_file(get_abs_path("korean/data/electronic/extensions.tsv"))
        cues = pynini.string_file(get_abs_path("korean/data/electronic/cc_cues.tsv"))
        digit = (pynini.string_file(get_abs_path("korean/data/number/digit.tsv"))
                 | pynini.cross("0", "영")).optimize()
        digit_rewrite = self.build_rule(digit)
        # A known ending must end at a label/path boundary, not inside .com.
        domain_rewrite = self.build_rule(domains | extensions, r=pynini.union(".", "/", "[EOS]"))
        dot_spacing = self.build_rule(pynini.cross("닷", " 닷"), self.NOT_SPACE)

        alnum = self.ALPHA | self.DIGIT
        username_symbols = pynini.difference(self.input_projection(symbols), "@")
        username_raw = alnum + (alnum | username_symbols).star + "@"
        self.username = (self.input_projection(username_raw) @ (digit_rewrite
                         @ self.build_rule(pynini.cross("@", " 골뱅이 ")))).optimize()
        label = (alnum | "-").plus
        tld = pynini.accep(".") + pynini.closure(alnum, 2)
        domain_core = label + tld.plus | tld
        path = pynini.accep("/") + self.NOT_SPACE.plus
        filename = pynini.difference(self.NOT_SPACE, pynini.union("/", ".")).plus
        filename += self.input_projection(extensions)
        domain_raw = (domain_core + path.ques | filename).optimize()
        self.domain = (domain_raw @ digit_rewrite @ domain_rewrite @ dot_spacing).optimize()

        # The source's deterministic URL reading uses English "colon".
        protocol_symbols = pynini.difference(self.input_projection(symbols), ":") @ symbols
        protocol_symbols = (protocol_symbols | pynini.cross(":", "colon")) + insert(" ")
        http = (pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP "))
        http += pynini.accep("://") @ protocol_symbols.star
        file_protocol = pynini.accep("file") + insert(" ") + (pynini.accep(":///") @ protocol_symbols.star)
        www = pynini.cross("www", "WWW ") + (pynini.accep(".") @ protocol_symbols.star)
        url_protocol = http | file_protocol | add_weight(www, -1000) | http + add_weight(www, -1000)
        # URL readings already end in a separator space; cues need one.
        self.protocol = (url_protocol | cues + insert(" ")).optimize()

        email = self.tag_field("username", self.username) + insert(" ") + self.tag_field("domain", self.domain)
        domain_only = self.tag_field("domain", domain_core | filename)
        url = self.tag_field("protocol", url_protocol) + insert(" ") + self.tag_field("domain", self.domain)
        cue = self.tag_field("protocol", cues) + self.DELETE_SPACE + insert(" ")
        cue += self.tag_field("domain", pynini.closure(self.DIGIT, 4, 16))
        four = digit**4
        grouped = (four + pynini.cross(pynini.union("-", " "), " "))**3 + four
        self.card_number = grouped.optimize()
        card = insert('kind: "credit_card" ') + self.tag_field("card_number", self.card_number)
        card = add_weight(card + self.DELETE_SPACE, -1.0)
        self.tagger = self.add_tokens(email | domain_only | url | cue | card)

        # Domain semantics also accept cue-bound digits without a TLD.
        cue_domain = pynini.closure(self.DIGIT, 4, 16) @ digit_rewrite
        protocol = self.verbalize_field("protocol", self.protocol) + delete(" ")
        username = self.verbalize_field("username", self.username) + delete(" ")
        domain = self.verbalize_field("domain", self.domain | cue_domain)
        default = protocol.ques + username.ques + domain
        card = delete('kind: "credit_card" ') + insert("신용카드 ")
        card += self.verbalize_field("card_number", self.card_number)
        self.verbalizer = self.delete_tokens(default | card)
