# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

import heapq
import logging
import string
from dataclasses import dataclass
from typing import Optional

from pynini import accep, cdrewrite, cross, difference, escape, invert, shortestpath, union
from pynini.lib import byte, utf8
from pynini.lib.pynutil import add_weight, delete, insert

from tn.alignment import NormalizationMapping, NormalizationResult, trace_input_spans, transduce_with_spans
from tn.cache import CacheBundle, default_cache_dir, production_source_fingerprint
from tn.token_parser import TokenParser

logger = logging.getLogger("wetext")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s WETEXT %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class RuleSpec:
    """One pipeline rule and its top-level classifier weight."""

    rule: object
    tagger_weight: Optional[float] = None
    verbalize: bool = True


@dataclass(frozen=True)
class _WeightedOutput:
    text: str
    weight: float
    rank: int


@dataclass(frozen=True)
class _NormalizationCandidate:
    tagged: str
    output: str
    tagger_weight: float
    verbalizer_weight: float
    best_verbalizer_weight: float
    tagger_rank: int
    verbalizer_rank: int

    @property
    def weight(self):
        return self.tagger_weight + self.verbalizer_weight - self.best_verbalizer_weight


class _UniqueOutputPathStream:
    """Lazily expands exact unique-output shortest paths without a fixed beam."""

    def __init__(self, lattice):
        self._lattice = lattice.copy()
        self._lattice.project("output").rmepsilon()
        self._items = []
        self._yielded = set()
        self._next_rank = 0
        self._limit = 0
        self._exhausted = False

    def _expand(self):
        if self._exhausted:
            return
        limit = 1 if self._limit == 0 else self._limit * 2
        shortest = shortestpath(self._lattice, nshortest=limit, unique=True)
        paths = shortest.paths()
        items = []
        seen = set()
        path_count = 0
        while not paths.done():
            path_count += 1
            output = paths.ostring()
            weight = float(paths.weight())
            if output not in seen and output not in self._yielded:
                seen.add(output)
                items.append(_WeightedOutput(output, weight, -1))
            paths.next()
        self._items = items
        self._limit = limit
        if path_count < limit:
            self._exhausted = True

    def peek(self):
        while not self._items and not self._exhausted:
            self._expand()
        if not self._items:
            return None
        item = self._items[0]
        return _WeightedOutput(item.text, item.weight, self._next_rank)

    def pop(self):
        item = self.peek()
        if item is not None:
            self._yielded.add(item.text)
            self._items.pop(0)
            self._next_rank += 1
        return item


class Processor:

    def __init__(self, name, ordertype="tn", token_orders=None):
        self.ALPHA = byte.ALPHA
        self.DIGIT = byte.DIGIT
        self.PUNCT = byte.PUNCT
        self.SPACE = byte.SPACE | "\u00A0"
        self.VCHAR = utf8.VALID_UTF8_CHAR
        self.VSIGMA = self.VCHAR.star
        self.LOWER = byte.LOWER
        self.UPPER = byte.UPPER

        CHAR = difference(self.VCHAR, union("\\", '"'))
        self.CHAR = CHAR | cross("\\", "\\\\\\") | cross('"', '\\"')
        self.SIGMA = (CHAR | cross("\\\\\\", "\\") | cross('\\"', '"')).star
        self.NOT_QUOTE = difference(self.VCHAR, r'"').optimize()
        self.NOT_SPACE = difference(self.VCHAR, self.SPACE).optimize()
        self.INSERT_SPACE = insert(" ")
        self.DELETE_SPACE = delete(self.SPACE).star
        self.DELETE_EXTRA_SPACE = cross(self.SPACE.plus, " ")
        self.DELETE_ZERO_OR_ONE_SPACE = delete(self.SPACE.ques)
        self.MIN_NEG_WEIGHT = -0.0001
        self.TO_LOWER = union(*[cross(x, y) for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)])
        self.TO_UPPER = invert(self.TO_LOWER)

        self.name = name
        self.ordertype = ordertype
        self.token_orders = token_orders
        self.tagger = None
        self.verbalizer = None

    @staticmethod
    def tagger_union(rule_specs):
        """Builds the classifier union from one ordered rule inventory."""

        taggers = [add_weight(spec.rule.tagger, spec.tagger_weight) for spec in rule_specs if spec.tagger_weight is not None]
        if not taggers:
            raise ValueError("rule inventory must contain at least one tagger")
        return union(*taggers).optimize()

    @staticmethod
    def verbalizer_union(rule_specs):
        """Builds the verbalizer union from the same rule inventory."""

        verbalizers = [spec.rule.verbalizer for spec in rule_specs if spec.verbalize]
        if not verbalizers:
            raise ValueError("rule inventory must contain at least one verbalizer")
        return union(*verbalizers).optimize()

    def token_parser(self):
        """Returns a parser configured by the owning pipeline."""

        return TokenParser(self.token_orders if self.token_orders is not None else self.ordertype)

    def build_rule(self, fst, l="", r=""):
        rule = cdrewrite(fst, l, r, self.VSIGMA)
        return rule

    def add_tokens(self, tagger):
        tagger = insert(f"{self.name} {{ ") + tagger + insert(" } ")
        return tagger.optimize()

    def delete_tokens(self, verbalizer):
        verbalizer = delete(f"{self.name}") + delete(" { ") + verbalizer + delete(" }") + delete(" ").ques
        return verbalizer.optimize()

    @staticmethod
    def input_projection(graph):
        """Returns the minimum-weight acceptor for a field's source side.

        Joint normalization subtracts the best conditional verbalizer weight,
        so retaining the source-side minimum here restores the original total
        path cost without double-counting output alternatives.
        """

        projected = graph.copy().project("input")
        return projected.optimize()

    def tag_field(self, name, graph):
        """Tags a weighted raw field using the quoted-field wire encoding."""

        serialized = self.input_projection(graph) @ self.CHAR.star
        return insert(f'{name}: "') + serialized + insert('"')

    def verbalize_field(self, name, graph=None, leading_space=False):
        """Decodes a quoted raw field, then applies its semantic transducer."""

        decoded = self.SIGMA if graph is None else self.SIGMA @ graph
        prefix = (" " if leading_space else "") + f'{name}: "'
        return delete(prefix) + decoded + delete('"')

    @staticmethod
    def apply_input_processor(graph, input_processor=None):
        """Composes optional input canonicalization into a semantic field graph."""

        if input_processor is None:
            return graph
        return (input_processor @ graph).optimize()

    def build_verbalizer(self):
        self.verbalizer = self.delete_tokens(self.verbalize_field("value"))

    @staticmethod
    def _source_fingerprint(prefix):
        del prefix
        return production_source_fingerprint()

    def build_fst(self, prefix, cache_dir, overwrite_cache, cache_config=None):
        """Loads or atomically builds a content-addressed graph bundle.

        ``cache_dir=None`` selects the platform user cache and ``False`` keeps
        graphs in memory only. Explicit directories remain supported as cache
        roots. Legacy flat v1 files are intentionally left untouched because
        they cannot prove that their tagger and verbalizer belong together.
        """

        cache_config = {} if cache_config is None else cache_config
        if cache_dir is False:
            logger.info("building fst for {} ...".format(self.name))
            if hasattr(self, 'build_tagger_and_verbalizer'):
                self.build_tagger_and_verbalizer()
            else:
                self.build_tagger()
                self.build_verbalizer()
            self.tagger.optimize()
            self.verbalizer.optimize()
            logger.info("done")
            return

        cache_root = default_cache_dir() if cache_dir is None else cache_dir
        bundle = CacheBundle(
            cache_root=cache_root,
            prefix=prefix,
            ordertype=self.ordertype,
            cache_config=cache_config,
            source_fingerprint=self._source_fingerprint(prefix),
        )

        if not overwrite_cache:
            graphs = bundle.load()
            if graphs is not None:
                logger.info("found existing fst bundle: {}".format(bundle.path))
                logger.info("skip building fst for {} ...".format(self.name))
                self.tagger, self.verbalizer = graphs
                return

        with bundle.lock():
            bundle.recover_residuals()
            if not overwrite_cache:
                graphs = bundle.load()
                if graphs is not None:
                    logger.info("found existing fst bundle: {}".format(bundle.path))
                    logger.info("skip building fst for {} ...".format(self.name))
                    self.tagger, self.verbalizer = graphs
                    return
                bundle.remove_invalid()

            logger.info("building fst for {} ...".format(self.name))
            if hasattr(self, 'build_tagger_and_verbalizer'):
                self.build_tagger_and_verbalizer()
            else:
                self.build_tagger()
                self.build_verbalizer()
            self.tagger.optimize()
            self.verbalizer.optimize()
            if self._source_fingerprint(prefix) != bundle.source_fingerprint:
                raise RuntimeError("grammar sources changed while building the cache bundle")
            bundle.publish(self.tagger, self.verbalizer)
            graphs = bundle.load()
            if graphs is None:
                raise RuntimeError("published cache bundle failed verification: {}".format(bundle.path))
            self.tagger, self.verbalizer = graphs
            logger.info("done")
            logger.info("fst bundle: {}".format(bundle.path))

    @staticmethod
    def _validate_nbest(nbest):
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer")

    def tag(self, input, nbest=1):
        self._validate_nbest(nbest)
        if len(input) == 0:
            return "" if nbest == 1 else [""]
        input = escape(input)
        lattice = input @ self.tagger
        if nbest == 1:
            return shortestpath(lattice, nshortest=1, unique=True).string()
        lattice = shortestpath(lattice.project("output").rmepsilon(), nshortest=nbest, unique=True)
        paths = lattice.paths()
        results = []
        while not paths.done():
            results.append(paths.ostring())
            paths.next()
        return results

    def verbalize(self, input):
        if len(input) == 0:
            return ""
        output, _, _ = self._verbalize_tagged(input)
        return output

    def _verbalize_tagged(self, tagged, trace_tokens=False, output_text=None):
        parser = self.token_parser()
        reordered, token_spans = parser.reorder_with_spans(tagged)
        output, output_spans = transduce_with_spans(
            reordered,
            self.verbalizer,
            token_spans if trace_tokens else (),
            output_text=output_text,
        )
        return output, parser, output_spans

    def normalize(self, input, nbest=1):
        self._validate_nbest(nbest)
        if not input:
            return "" if nbest == 1 else [""]
        candidates = self._normalization_candidates(input, nbest)
        outputs = [candidate.output for candidate in candidates]
        return outputs[0] if nbest == 1 else outputs

    def normalize_with_mapping(self, input, nbest=1, include_identity=False):
        """Normalizes text and traces each tagged token through the WFSTs.

        This requires the tagger to preserve written field values. It never
        falls back to surface-text diffing.
        """

        self._validate_nbest(nbest)
        if not input:
            result = NormalizationResult("", "", ())
            return result if nbest == 1 else [result]

        candidates = self._normalization_candidates(input, nbest)
        results = [self._normalize_candidate_with_mapping(input, candidate, include_identity) for candidate in candidates]
        return results[0] if nbest == 1 else results

    def _normalization_candidates(self, input, nbest):
        tagger_stream = _UniqueOutputPathStream(accep(escape(input)) @ self.tagger)
        frontier = []
        activated = 0
        candidates = []
        seen_outputs = set()

        def activate_next_tag():
            nonlocal activated
            tagged_path = tagger_stream.pop()
            if tagged_path is None:
                return False
            parser = self.token_parser()
            reordered = parser.reorder(tagged_path.text)
            verbalizer_stream = _UniqueOutputPathStream(accep(escape(reordered)) @ self.verbalizer)
            verbalized_path = verbalizer_stream.pop()
            tagger_rank = tagged_path.rank
            activated += 1
            if verbalized_path is None:
                return True
            candidate = _NormalizationCandidate(
                tagged=tagged_path.text,
                output=verbalized_path.text,
                tagger_weight=tagged_path.weight,
                verbalizer_weight=verbalized_path.weight,
                best_verbalizer_weight=verbalized_path.weight,
                tagger_rank=tagger_rank,
                verbalizer_rank=verbalized_path.rank,
            )
            heapq.heappush(
                frontier,
                (
                    candidate.weight,
                    candidate.tagger_rank,
                    candidate.verbalizer_rank,
                    activated,
                    candidate,
                    verbalizer_stream,
                ),
            )
            return True

        while len(candidates) < nbest:
            next_tag = tagger_stream.peek()
            if not frontier:
                if not activate_next_tag():
                    break
                continue

            frontier_key = frontier[0][:3]
            if next_tag is not None:
                # Verbalizer weights are conditional on a raw tag: subtracting
                # that tag's best verbalizer weight keeps all spoken variants'
                # relative costs while leaving cross-tag classification to the
                # tagger. Therefore every unseen tag has an exact lower bound
                # equal to its tagger weight, even when the full verbalizer has
                # negative-weight cycles across arbitrary token sequences.
                unseen_key = (next_tag.weight, next_tag.rank, 0)
                if unseen_key < frontier_key:
                    activate_next_tag()
                    continue

            _, _, _, serial, candidate, verbalizer_stream = heapq.heappop(frontier)
            next_verbalized = verbalizer_stream.pop()
            if next_verbalized is not None:
                next_candidate = _NormalizationCandidate(
                    tagged=candidate.tagged,
                    output=next_verbalized.text,
                    tagger_weight=candidate.tagger_weight,
                    verbalizer_weight=next_verbalized.weight,
                    best_verbalizer_weight=candidate.best_verbalizer_weight,
                    tagger_rank=candidate.tagger_rank,
                    verbalizer_rank=next_verbalized.rank,
                )
                heapq.heappush(
                    frontier,
                    (
                        next_candidate.weight,
                        next_candidate.tagger_rank,
                        next_candidate.verbalizer_rank,
                        serial,
                        next_candidate,
                        verbalizer_stream,
                    ),
                )
            if candidate.output in seen_outputs:
                continue
            seen_outputs.add(candidate.output)
            candidates.append(candidate)

        if not candidates:
            raise RuntimeError("no normalization path for the requested input")
        return candidates

    def _normalize_candidate_with_mapping(self, input, candidate, include_identity):
        tagged = candidate.tagged

        output, parser, output_spans = self._verbalize_tagged(
            tagged,
            trace_tokens=True,
            output_text=candidate.output,
        )
        input_spans = trace_input_spans(
            input,
            tagged,
            self.tagger,
            ((token.start, token.end) for token in parser.tokens),
        )

        mappings = []
        for token, (input_start, input_end), (output_start, output_end) in zip(parser.tokens, input_spans, output_spans):
            source = input[input_start:input_end]
            spoken = output[output_start:output_end]
            if source == spoken:
                kind = "equal"
            elif not spoken:
                kind = "delete"
            else:
                kind = "replace"
            if include_identity or kind != "equal":
                mappings.append(
                    NormalizationMapping(
                        kind=kind,
                        token_type=token.name,
                        input_start=input_start,
                        input_end=input_end,
                        output_start=output_start,
                        output_end=output_end,
                        input_text=source,
                        output_text=spoken,
                    ))

        return NormalizationResult(input, output, tuple(mappings))
