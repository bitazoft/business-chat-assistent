"""
Templates are edited by shop staff, so rendering must be safe against both
typos and deliberate abuse.
"""
from templates.template_store import (
    DEFAULT_TEMPLATES,
    TemplateKey,
    placeholders_in,
    render_template,
)


def test_substitutes_known_placeholders():
    assert render_template("Hello {name}!", {"name": "Nimal"}) == "Hello Nimal!"


def test_unknown_placeholder_renders_empty_not_an_error():
    """A seller's typo must not raise KeyError mid-send."""
    assert render_template("Hi {custmer_name}!", {"customer_name": "Nimal"}) == "Hi !"


def test_none_renders_empty():
    assert render_template("Ref: {reference}", {"reference": None}) == "Ref: "


def test_non_string_values_are_coerced():
    assert render_template("{count} items", {"count": 3}) == "3 items"


def test_attribute_access_is_not_evaluated():
    """`{x.__class__}` is valid str.format syntax and would leak internals."""
    out = render_template("{message.__class__.__mro__}", {"message": "hi"})
    assert "class" in out or out == "{message.__class__.__mro__}"
    assert "type" not in out.lower()
    assert "object" not in out.lower()


def test_indexing_and_format_specs_are_left_alone():
    assert render_template("{0}", {"0": "x"}) == "{0}"
    assert render_template("{price:.2f}", {"price": 3}) == "{price:.2f}"


def test_braces_in_copy_do_not_break_rendering():
    assert render_template("Use {{ and }} freely", {}) == "Use {{ and }} freely"


def test_placeholders_in_lists_unique_names_in_order():
    assert placeholders_in("{a} {b} {a} {c}") == ["a", "b", "c"]
    assert placeholders_in("") == []


def test_every_default_template_renders_with_its_documented_placeholders():
    """A shipped default must never leave an unfilled placeholder."""
    for key, definition in DEFAULT_TEMPLATES.items():
        context = {name: f"<{name}>" for name in placeholders_in(definition.body)}
        rendered = render_template(definition.body, context)
        assert "{" not in rendered.replace("{{", "").replace("}}", ""), key
        assert rendered.strip(), key


def test_outbound_wrapper_default_passes_the_message_through():
    body = DEFAULT_TEMPLATES[TemplateKey.OUTBOUND_WRAPPER].body
    assert "{message}" in body
    assert render_template(body, {"message": "the answer"}) == "the answer"


def test_default_templates_only_use_documented_placeholders():
    """Catches a default that references a name nothing ever supplies."""
    from routes.template_routes import ALWAYS_AVAILABLE as always_available
    for key, definition in DEFAULT_TEMPLATES.items():
        used = set(placeholders_in(definition.body))
        declared = set(definition.placeholders) | always_available
        assert used <= declared, f"{key} uses undeclared {used - declared}"
