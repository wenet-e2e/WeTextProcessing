# Python Rule Architecture

This document defines the contract for Python TN and ITN grammars. Following
it keeps normalization explainable and allows exact input/output span mapping.

## Pipeline contract

The pipeline has two semantic stages:

1. The **tagger** classifies input and serializes the original field values.
2. The **verbalizer** converts those raw fields to their normalized form.

A tagger must not perform an irreversible semantic rewrite. For example, the
Chinese TN input `12` should be tagged with `value: "12"`; converting it to
`十二` is the verbalizer's responsibility. Character-width conversion,
traditional-to-simplified conversion, punctuation removal, and similar output
cleanup also belong after tagging.

This gives `normalize_with_mapping()` a real WFST path from the raw input span
to the normalized output span. The implementation intentionally has no
surface-text-diff fallback.

## Implementing a field

Use the helpers on `Processor` instead of manually encoding quoted fields:

```py
class Example(Processor):

    def __init__(self):
        super().__init__("example")
        self.graph = ...  # raw written/spoken form -> normalized form
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))

    def build_verbalizer(self):
        value = self.verbalize_field("value", self.graph)
        self.verbalizer = self.delete_tokens(value)
```

`tag_field()` projects the semantic graph onto its input side and safely
serializes the raw value. `verbalize_field()` decodes that value and applies
the semantic graph. These helpers also handle quotes and backslashes in the
tagged-token wire format.

When input canonicalization is genuinely required before matching, compose it
with the semantic graph through `apply_input_processor()`. Do not replace the
serialized field with the canonicalized or normalized value.

## Registering rules

Each top-level language/direction pipeline owns:

- one ordered `RuleSpec` inventory;
- the classifier weight for every taggable rule;
- its token field-order schema.

Build both unions from the same inventory:

```py
rules = (
    RuleSpec(date, 1.02),
    RuleSpec(cardinal, 1.06),
    RuleSpec(char, 100),
)
tagger = self.tagger_union(rules)
verbalizer = self.verbalizer_union(rules)
```

Use `RuleSpec(rule)` for a verbalizer-only rule. Use
`RuleSpec(rule, weight, verbalize=False)` only when a classifier rule
intentionally has no top-level verbalizer. Keeping a single inventory prevents
new rules from being added to only one side of the pipeline.

Token field order belongs to the owning pipeline and is passed through
`Processor(..., token_orders=TOKEN_ORDERS)`. The global maps in
`tn.token_parser` exist only for backward compatibility with directly
instantiated rules.

## API behavior

- `normalize(text)` returns the best normalized string.
- `normalize(text, nbest=N)` returns a list when `N > 1`.
- `normalize_with_mapping(text)` returns a `NormalizationResult`.
- `normalize_with_mapping(text, nbest=N)` returns a list when `N > 1`.
- `include_identity=True` includes tagged tokens whose surface text is
  unchanged.
- Mapping offsets are half-open Python Unicode character offsets, not UTF-8
  byte offsets.
- Untagged text does not produce mapping entries.

N-best ranking is joint across tagger and verbalizer paths. Do not implement
an independent tagger-only fallback or surface matching layer.

## Review checklist

- Does every tagger field preserve the corresponding raw input?
- Is every semantic conversion applied in the verbalizer or postprocessor?
- Are fields built with `tag_field()` and `verbalize_field()`?
- Is the rule registered once through the pipeline's `RuleSpec` inventory?
- Does the pipeline own the token field-order schema?
- Are tag output, verbalized output, mapping spans, escaping, and n-best
  behavior covered by tests where applicable?
