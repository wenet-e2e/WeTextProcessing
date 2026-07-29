# Copyright (c) 2026, WENET COMMUNITY.
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

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from pynini import accep, escape, shortestpath


class AlignmentError(RuntimeError):
    """Raised when a normalization path cannot be traced exactly."""


@dataclass(frozen=True)
class NormalizationMapping:
    """The written and spoken spans produced by one tagged token."""

    kind: str
    token_type: str
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    input_text: str
    output_text: str

    def as_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "token_type": self.token_type,
            "input": {
                "start": self.input_start,
                "end": self.input_end,
                "text": self.input_text,
            },
            "output": {
                "start": self.output_start,
                "end": self.output_end,
                "text": self.output_text,
            },
        }


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized text together with exact token-level WFST mappings."""

    input_text: str
    output_text: str
    mappings: Tuple[NormalizationMapping, ...]

    def as_dict(self) -> Dict:
        return {
            "input": self.input_text,
            "output": self.output_text,
            "mappings": [mapping.as_dict() for mapping in self.mappings],
        }


def _text_boundaries(text):
    character_to_byte = [0]
    byte_to_character = {0: 0}
    byte_is_whitespace = []
    offset = 0
    for index, char in enumerate(text, 1):
        encoded_length = len(char.encode("utf-8"))
        byte_is_whitespace.extend([char.isspace()] * encoded_length)
        offset += encoded_length
        character_to_byte.append(offset)
        byte_to_character[offset] = index
    return tuple(character_to_byte), byte_to_character, tuple(byte_is_whitespace)


def _character_spans_to_byte_spans(text, character_to_byte, spans, reject_empty=False):
    previous_end = 0
    normalized = []
    for start, end in spans:
        if start < previous_end or end < start or end > len(text):
            raise AlignmentError("token spans must be ordered, non-overlapping character ranges")
        if reject_empty and start == end:
            raise ValueError("input spans must be non-empty character ranges")
        normalized.append((character_to_byte[start], character_to_byte[end]))
        previous_end = end
    return tuple(normalized)


def _linear_arcs(path):
    state = path.start()
    if state == -1:
        raise AlignmentError("no WFST path for the requested input")

    visited = set()
    while state not in visited:
        visited.add(state)
        arcs = list(path.arcs(state))
        if not arcs:
            return
        if len(arcs) != 1:
            raise AlignmentError("shortest WFST path is not linear")
        arc = arcs[0]
        yield arc
        state = arc.nextstate
    raise AlignmentError("shortest WFST path contains a cycle")


def trace_input_spans(input_text: str, output_text: str, fst, output_spans: Iterable[Tuple[int, int]]):
    """Maps output character spans to input spans on one exact WFST path."""

    lattice = accep(escape(input_text)) @ fst @ accep(escape(output_text))
    path = shortestpath(lattice, nshortest=1, unique=True)
    input_character_to_byte, input_byte_to_character, input_byte_is_whitespace = _text_boundaries(input_text)
    output_character_to_byte, _, _ = _text_boundaries(output_text)
    output_byte_spans = _character_spans_to_byte_spans(output_text, output_character_to_byte, output_spans)
    consumed_starts = [None] * len(output_byte_spans)
    consumed_ends = [None] * len(output_byte_spans)
    input_offset = 0
    output_offset = 0
    token_index = 0

    for arc in _linear_arcs(path):
        next_input_offset = input_offset + (1 if arc.ilabel else 0)
        while token_index < len(output_byte_spans) and output_offset >= output_byte_spans[token_index][1]:
            token_index += 1
        owner = None
        if token_index < len(output_byte_spans):
            start, end = output_byte_spans[token_index]
            if arc.ilabel and start <= output_offset < end and (output_offset != start or arc.olabel):
                is_deleted_leading_space = (not arc.olabel and input_byte_is_whitespace[input_offset]
                                            and consumed_starts[token_index] is None)
                if not is_deleted_leading_space:
                    owner = token_index
            elif (arc.ilabel and arc.olabel and output_offset + 1 == start and not input_byte_is_whitespace[input_offset]):
                owner = token_index
        if owner is not None:
            if consumed_starts[owner] is None:
                consumed_starts[owner] = input_offset
            consumed_ends[owner] = next_input_offset
        input_offset = next_input_offset
        output_offset += 1 if arc.olabel else 0

    spans = []
    for start, end in zip(consumed_starts, consumed_ends):
        if start is None:
            raise AlignmentError("tagged token did not consume any input")
        try:
            spans.append((input_byte_to_character[start], input_byte_to_character[end]))
        except KeyError as error:
            raise AlignmentError("WFST path split a Unicode code point") from error
    return tuple(spans)


def transduce_with_spans(
        input_text: str,
        fst,
        input_spans: Iterable[Tuple[int, int]] = (),
        output_text: str = None,
):
    """Runs one exact WFST path and maps input token spans to its output.

    Output emitted while the path is inside a token's serialized input range
    belongs to that token. At a token boundary, whitespace output is an
    unowned stream separator; delayed non-whitespace output belongs to the
    preceding token. A fully deleted token gets a zero-width span after all
    output already emitted at its starting boundary. When ``output_text`` is
    provided, the traced path is constrained to that selected output instead
    of silently falling back to the shortest output.
    """

    character_spans = tuple(input_spans)
    input_character_to_byte, _, _ = _text_boundaries(input_text)
    input_byte_spans = _character_spans_to_byte_spans(
        input_text,
        input_character_to_byte,
        character_spans,
        reject_empty=True,
    )
    lattice = accep(escape(input_text)) @ fst
    if output_text is not None:
        lattice @= accep(escape(output_text))
    path = shortestpath(lattice, nshortest=1, unique=True)
    if path.start() == -1:
        raise AlignmentError("no WFST path for the requested input")
    selected_output = path.string()
    output_character_to_byte, output_byte_to_character, output_byte_is_whitespace = _text_boundaries(selected_output)
    emitted_starts = [None] * len(input_byte_spans)
    emitted_ends = [None] * len(input_byte_spans)
    boundary_offsets = {}
    input_offset = 0
    output_offset = 0
    token_index = 0

    for arc in _linear_arcs(path):
        boundary_offsets[input_offset] = output_offset
        while token_index < len(input_byte_spans) and input_offset >= input_byte_spans[token_index][1]:
            token_index += 1
        owner = None
        if arc.ilabel:
            if token_index < len(input_byte_spans):
                start, end = input_byte_spans[token_index]
                if start <= input_offset < end:
                    owner = token_index
        else:
            if token_index < len(input_byte_spans):
                start, end = input_byte_spans[token_index]
                if start < input_offset < end or (token_index == 0 and input_offset == start):
                    owner = token_index
            is_boundary_whitespace = arc.olabel and output_byte_is_whitespace[output_offset]
            if owner is None and arc.olabel and not is_boundary_whitespace:
                previous_index = token_index - 1
                if previous_index >= 0 and input_byte_spans[previous_index][1] <= input_offset:
                    owner = previous_index
        if arc.olabel and owner is not None:
            if emitted_starts[owner] is None:
                emitted_starts[owner] = output_offset
            emitted_ends[owner] = output_offset + 1
        input_offset += 1 if arc.ilabel else 0
        output_offset += 1 if arc.olabel else 0

    boundary_offsets[input_offset] = output_offset
    expected_input_bytes = input_character_to_byte[-1]
    expected_output_bytes = output_character_to_byte[-1]
    if input_offset != expected_input_bytes or output_offset != expected_output_bytes:
        raise AlignmentError("WFST path offsets do not match its input/output strings")

    output_spans = []
    previous_end = 0
    for (input_start, _), emitted_start, emitted_end in zip(input_byte_spans, emitted_starts, emitted_ends):
        if emitted_start is not None:
            output_start = emitted_start
            output_end = emitted_end
            if output_start < previous_end:
                raise AlignmentError("token output spans overlap or are out of order")
        else:
            output_start = output_end = max(boundary_offsets.get(input_start, output_offset), previous_end)
        try:
            output_spans.append((output_byte_to_character[output_start], output_byte_to_character[output_end]))
        except KeyError as error:
            raise AlignmentError("WFST path split a Unicode code point") from error
        previous_end = output_end
    return selected_output, tuple(output_spans)
