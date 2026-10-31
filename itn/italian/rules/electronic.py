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

from itn.italian.rules.utils import load_rows, string_map, table_map
from tn.processor import Processor


class Electronic(Processor):
    """Spelled URLs, e-mail addresses and handles, e.g. ``chiocciola j e n s e n`` -> ``@jensen``.

    The tagger keeps the spoken protocol, user name and domain; the
    verbalizer joins letters, digit words (``uno`` -> ``1``), symbol names
    (``trattino basso`` -> ``_``), server names (``nvidia``) and top-level
    domains (``punto com`` -> ``.com``). A bare domain needs a ``punto`` and
    must end in a known top-level domain or a run of at least two letters.
    """

    def __init__(self):
        super().__init__("electronic", token_orders={})
        sep = delete(" ")
        digits = table_map("italian/data/numbers/digit.tsv") | pynini.cross("zero", "0")
        content = (self.ALPHA | digits).optimize()
        names = pynini.union(*[row[0] for row in load_rows("italian/data/electronic/server_name.tsv")])
        symbols = table_map("italian/data/electronic/symbols.tsv")
        dot = pynini.cross("punto", ".")
        others = (symbols @ (self.VSIGMA - ".")).optimize()
        tlds = string_map([(row[0], "punto " + row[1]) for row in load_rows("italian/data/electronic/domain.tsv")])

        start = content | names
        segment = start + (sep + (start | others)).star
        long_segment = start + (sep + (start | others)).plus | names
        domain = segment + (sep + (dot + sep + segment | tlds)).star + sep + (tlds | dot + sep + long_segment)
        domain = domain.optimize()
        user = (content + (sep + (start | others | dot)).star).optimize()
        handle_user = (content + (sep + (start | others | dot)).plus).optimize()

        slashes = " due punti slash slash"
        scheme = pynini.cross("h t t p s" + slashes, "https://") | pynini.cross("h t t p" + slashes, "http://")
        www = pynini.cross("w w w punto", "www.")
        protocol = (scheme | www | scheme + sep + www).optimize()
        path_item = start | others | dot | tlds
        path = sep + (others @ pynini.accep("/")) + (sep + path_item).plus
        # Prefer the ``www.`` reading of the protocol over spelling it in the domain.
        url_domain = (self.VSIGMA - ("w w w punto" + self.VSIGMA).optimize()) @ (domain + path.ques)

        space = pynini.accep(" ")
        url = self.tag_field("protocol", protocol) + space + self.tag_field("domain", url_domain)
        email = self.tag_field("username", user) + delete(" chiocciola") + space + self.tag_field("domain", domain)
        handle = delete("chiocciola ") + self.tag_field("username", handle_user)
        self.tagger = self.add_tokens(url | email | handle | self.tag_field("domain", domain))

        url_verb = self.verbalize_field("protocol", protocol) + sep + self.verbalize_field("domain", url_domain)
        email_verb = self.verbalize_field("username", user) + sep + insert("@") + self.verbalize_field("domain", domain)
        handle_verb = insert("@") + self.verbalize_field("username", handle_user)
        bare_verb = self.verbalize_field("domain", domain)
        self.verbalizer = self.delete_tokens(url_verb | email_verb | handle_verb | bare_verb)
