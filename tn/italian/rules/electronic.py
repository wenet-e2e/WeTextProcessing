# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.italian.rules.utils import load_pairs, load_words, preferred_map
from tn.processor import Processor

# Penalty for spelling a run that also has a whole-word reading.
SPELL_WEIGHT = 0.001


class Electronic(Processor):
    """URLs, e-mail addresses and handles, e.g. ``@jensen`` -> ``chiocciola j e n s e n``.

    The tagger keeps the written protocol, user name and domain; the
    verbalizer spells them, naming symbols and digits, while known server
    names (``nvidia``) and top-level domains (``.com`` -> ``punto com``) are
    read as words when they cover a whole letter/digit run.
    """

    def __init__(self, cardinal):
        super().__init__("electronic", token_orders={})
        letter = self.ALPHA
        digit_words = pynini.cross("0", "zero") | cardinal.digits
        symbol_pairs = load_pairs("italian/data/electronic/symbols.tsv")
        symbol_words = preferred_map("italian/data/electronic/symbols.tsv")
        symbols = pynini.union(*[pynini.escape(key) for key, _ in symbol_pairs]).optimize()
        symbols_no_dot = pynini.difference(symbols, ".").optimize()
        servers = pynini.union(*load_words("italian/data/electronic/server_name.tsv")).optimize()
        domains = preferred_map("italian/data/electronic/domain.tsv")

        alnum = letter | self.DIGIT
        accepted = (alnum | symbols_no_dot).plus
        every = (alnum | symbols).plus
        segment = "." + accepted
        # A bare domain needs a letter in its last segment (``1.000`` is a number).
        last = accepted @ (self.VSIGMA + letter + self.VSIGMA)
        domain = accepted + segment.star + "." + last
        url_domain = accepted + segment.plus
        protocol = pynini.union("https://", "http://")
        protocol = protocol | "www." | protocol + "www."

        space = insert(" ")
        char = letter | digit_words | symbol_words
        spelled = char + (space + char).star
        word = add_weight(alnum.plus @ spelled, SPELL_WEIGHT) | servers
        # Runs are maximal: a run is always followed by a symbol or the end.
        unit = (word | domains) + (space + domains).star
        item = symbol_words | unit + space + symbol_words
        domain_verb = ((item + space).star + (item | unit)).optimize()

        domain_tag = self.tag_field("domain", url_domain @ domain_verb)
        url = self.tag_field("protocol", protocol) + insert(" ") + domain_tag
        email = self.tag_field("username", every @ spelled) + delete("@") + insert(" ") + domain_tag
        handle = delete("@") + self.tag_field("username", (accepted + segment.star) @ spelled)
        bare = self.tag_field("domain", domain @ domain_verb)
        self.tagger = self.add_tokens(url | email | handle | bare)

        protocol_verb = self.verbalize_field("protocol", protocol @ spelled) + delete(" ") + insert(" ")
        domain_field = self.verbalize_field("domain", domain_verb)
        username = self.verbalize_field("username", spelled)
        verbalizer = protocol_verb.ques + domain_field
        verbalizer |= username + delete(" ") + insert(" chiocciola ") + domain_field
        verbalizer |= insert("chiocciola ") + username
        self.verbalizer = self.delete_tokens(verbalizer)
