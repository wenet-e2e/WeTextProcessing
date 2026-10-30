import pytest
from pynini import cross, escape
from pynini import union
from pynini.lib.pynutil import add_weight, delete, insert

import tn.alignment as alignment
from tn.alignment import AlignmentError, NormalizationMapping, trace_input_spans, transduce_with_spans


def test_trace_input_spans_uses_the_wfst_path():
    tagged = 'char { value: "1" } math { value: "23" }'
    tagger = cross("1", 'char { value: "1" }') + cross("23", ' math { value: "23" }')

    spans = trace_input_spans("123", tagged, tagger, ((0, 19), (20, len(tagged))))

    assert spans == ((0, 1), (1, 3))


def test_mapping_exposes_token_type():
    mapping = NormalizationMapping(
        kind="replace",
        token_type="math",
        input_start=4,
        input_end=6,
        output_start=4,
        output_end=6,
        input_text="12",
        output_text="十二",
    )

    assert mapping.as_dict() == {
        "kind": "replace",
        "token_type": "math",
        "input": {
            "start": 4,
            "end": 6,
            "text": "12",
        },
        "output": {
            "start": 4,
            "end": 6,
            "text": "十二",
        },
    }


def test_transduce_with_spans_leaves_inserted_separator_unowned():
    output, spans = transduce_with_spans(
        "ab",
        cross("a", "A") + insert(" ") + cross("b", "B"),
        ((0, 1), (1, 2)),
    )

    assert output == "A B"
    assert spans == ((0, 1), (2, 3))


@pytest.mark.parametrize("separator", ["\u00a0", "\u3000"])
def test_transduce_with_spans_leaves_unicode_whitespace_unowned(separator):
    output, spans = transduce_with_spans(
        "ab",
        cross("a", "A") + insert(separator) + cross("b", "B"),
        ((0, 1), (1, 2)),
    )

    assert separator.isspace()
    assert output == "A{}B".format(separator)
    assert spans == ((0, 1), (2, 3))


def test_transduce_with_spans_covers_deletion_and_unicode():
    output, spans = transduce_with_spans(
        "蘋-果",
        cross("蘋", "苹") + delete("-") + cross("果", "果"),
        ((0, 1), (1, 2), (2, 3)),
    )

    assert output == "苹果"
    assert spans == ((0, 1), (1, 1), (1, 2))


def test_deleted_token_anchor_follows_delayed_boundary_output():
    output, spans = transduce_with_spans(
        "ab",
        cross("a", "A") + insert("X") + delete("b"),
        ((0, 1), (1, 2)),
    )

    assert output == "AX"
    assert spans == ((0, 2), (2, 2))
    assert spans[0][1] <= spans[1][0]


def test_transduce_with_spans_rejects_empty_input_spans():
    with pytest.raises(ValueError, match="input spans must be non-empty"):
        transduce_with_spans(
            "a",
            cross("a", "A"),
            ((0, 0), (0, 1)),
        )


def test_transduce_with_spans_traces_the_selected_non_shortest_output():
    graph = union(
        cross("a", "A"),
        add_weight(cross("a", "ALT"), 1.0),
    )

    output, spans = transduce_with_spans(
        "a",
        graph,
        ((0, 1), ),
        output_text="ALT",
    )

    assert output == "ALT"
    assert spans == ((0, 3), )


@pytest.mark.parametrize("helper", ["_linear_arcs", "_text_boundaries"])
@pytest.mark.parametrize("selected_output", [None, '别名 "[x]"\\\\'])
def test_transduce_without_spans_skips_tracking_helpers(monkeypatch, helper, selected_output):
    source = '原文 "[x]"\\\\'
    best = "规范化😀"
    alternate = '别名 "[x]"\\\\'
    graph = union(
        cross(escape(source), escape(best)),
        add_weight(cross(escape(source), escape(alternate)), 1.0),
    )
    calls = []
    original = getattr(alignment, helper)

    def counted_helper(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(alignment, helper, counted_helper)
    output, spans = transduce_with_spans(source, graph, output_text=selected_output)

    assert output == (best if selected_output is None else selected_output)
    assert spans == ()
    assert calls == []


def test_transduce_without_spans_preserves_empty_output():
    assert transduce_with_spans("a", delete("a")) == ("", ())


@pytest.mark.parametrize("selected_output", [None, "missing"])
def test_transduce_without_spans_rejects_missing_paths(selected_output):
    source = "missing" if selected_output is None else "a"

    with pytest.raises(AlignmentError, match="no WFST path"):
        transduce_with_spans(source, cross("a", "A"), output_text=selected_output)
