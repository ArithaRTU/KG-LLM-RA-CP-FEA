"""Fractal 4EM Change Studio.

Single-file Streamlit implementation of a knowledge-graph-backed,
rationale-aware approach to federated enterprise architecture. Model changes
notify sibling units and the immediate parent. Adaptation guidance is generated
only when a notified user explicitly requests it.

Install dependencies with ``pip install streamlit neo4j openai`` and run the
application with ``streamlit run Upload/UI_Implementation.py``.

Runtime settings can be supplied through the environment variables declared
below. Organisational-unit selection controls view and edit scope only; it is
not an authentication boundary.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import textwrap
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from openai import OpenAI


# Runtime configuration

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "30000"))

APP_TITLE = "Fractal 4EM Change Studio"
MAX_GRAPH_ELEMENTS = int(os.getenv("MAX_GRAPH_ELEMENTS", "180"))
# Rank against the full result set while enforcing a predictable LLM context size.
MAX_TARGET_ELEMENTS_FETCH = int(os.getenv("MAX_TARGET_ELEMENTS_FETCH", "5000"))
MAX_TARGET_RELATIONSHIPS_FETCH = int(os.getenv("MAX_TARGET_RELATIONSHIPS_FETCH", "5000"))
MAX_LLM_ELEMENTS = int(os.getenv("MAX_LLM_ELEMENTS", "160"))
MAX_LLM_RELATIONSHIPS = int(os.getenv("MAX_LLM_RELATIONSHIPS", "240"))
CANDIDATE_SHORTLIST_SIZE = int(os.getenv("CANDIDATE_SHORTLIST_SIZE", "8"))


# 4EM domain model

SUBMODELS: dict[str, str] = {
    "Goals": "Goals Model",
    "Business Rules": "Business Rules Model",
    "Concepts": "Concepts Model",
    "Business Processes": "Business Process Model",
    "Actors & Resources": "Actors and Resources Model",
    "Technical Components & Requirements": "Technical Components and Requirements Model",
}

ELEMENT_TYPES: dict[str, list[str]] = {
    "Goals": [
        "Goal",
        "Problem",
        "Weakness",
        "Threat",
        "Cause",
        "Constraint",
        "Opportunity",
    ],
    "Business Rules": [
        "BusinessRule",
        "DerivationRule",
        "EventActionRule",
        "StaticConstraintRule",
        "TransitionConstraintRule",
    ],
    "Concepts": ["Concept", "Attribute"],
    "Business Processes": [
        "Process",
        "ExternalProcess",
        "InformationSet",
        "MaterialSet",
    ],
    "Actors & Resources": [
        "Individual",
        "OrganisationalUnit",
        "Role",
        "NonHumanResource",
    ],
    "Technical Components & Requirements": [
        "ISGoal",
        "ISProblem",
        "ISRequirement",
        "FunctionalRequirement",
        "NonFunctionalRequirement",
        "TechnicalComponent",
    ],
}

TYPE_PREFIX: dict[str, str] = {
    "Goal": "Goal",
    "Problem": "Problem",
    "Weakness": "Weakness",
    "Threat": "Threat",
    "Cause": "Cause",
    "Constraint": "Constraint",
    "Opportunity": "Opportunity",
    "BusinessRule": "Rule",
    "DerivationRule": "Derivation Rule",
    "EventActionRule": "Event-Action Rule",
    "StaticConstraintRule": "Static Rule",
    "TransitionConstraintRule": "Transition Rule",
    "Concept": "Concept",
    "Attribute": "Attribute",
    "Process": "Process",
    "ExternalProcess": "External Process",
    "InformationSet": "Information Set",
    "MaterialSet": "Material Set",
    "Individual": "Individual",
    "OrganisationalUnit": "OU",
    "Role": "Role",
    "NonHumanResource": "Resource",
    "ISGoal": "IS Goal",
    "ISProblem": "IS Problem",
    "ISRequirement": "IS Requirement",
    "FunctionalRequirement": "Functional Requirement",
    "NonFunctionalRequirement": "Nonfunctional Requirement",
    "TechnicalComponent": "TC",
}

# These fallback styles approximate the Chapter 8 notation in Sandkuhl et al.,
# Enterprise Modeling: Tackling Business Challenges with the 4EM Method. The
# interactive renderer uses the more precise, element-specific SVG definitions.
SUBMODEL_STYLE: dict[str, dict[str, str]] = {
    "Goals": {"shape": "box", "fill": "#98E58F"},
    "Business Rules": {"shape": "box", "fill": "#E8E1EF"},
    "Concepts": {"shape": "ellipse", "fill": "#E9EF00"},
    "Business Processes": {"shape": "ellipse", "fill": "#FFFFFF"},
    "Actors & Resources": {"shape": "box", "fill": "#E7C6CB"},
    "Technical Components & Requirements": {
        "shape": "ellipse",
        "fill": "#FFFFFF",
    },
}

BOOK_4EM_COLOURS: dict[str, str] = {
    "goal": "#98E58F",
    "problem": "#F2B400",
    "cause": "#4FA6CF",
    "constraint_border": "#EE8B7B",
    "opportunity": "#087F22",
    "rule": "#E8E1EF",
    "concept": "#E9EF00",
    "concept_border": "#4768D7",
    "individual": "#E6C4CA",
    "role": "#8669AA",
    "resource": "#D8D8D8",
    "org_unit_border": "#78A7EE",
    "line": "#5F6368",
}

RELATION_STRENGTHS = ["", "LOW", "MEDIUM", "HIGH"]
CONCEPT_CARDINALITIES = ["", "0:1", "1:1", "0:N", "1:N", "0:M", "1:M"]
RELATION_TOTALITIES = ["PARTIAL", "TOTAL"]

MODEL_STATUSES = ["AS_IS", "TO_BE", "PROPOSED", "ARCHIVED"]
PRIORITIES = ["", "LOW", "MEDIUM", "HIGH"]
CRITICALITIES = ["", "LOW", "MEDIUM", "HIGH"]

INTRA_MODEL_RELATIONSHIPS: dict[str, list[str]] = {
    "Goals": [
        "SUPPORTS",
        "HINDERS",
        "CONFLICTS",
        "CAUSES",
        "AND_REFINES",
        "OR_REFINES",
        "AND_OR_REFINES",
        "RELATES_TO",
    ],
    "Business Rules": [
        "SUPPORTS",
        "HINDERS",
        "AND_REFINES",
        "OR_REFINES",
        "RELATES_TO",
    ],
    "Concepts": ["RELATES_TO", "ISA", "PART_OF", "HAS_ATTRIBUTE"],
    "Business Processes": [
        "CONTROL_FLOW",
        "DECOMPOSES_TO",
        "INPUT_TO",
        "OUTPUT_FROM",
        "AND_SPLIT",
        "OR_SPLIT",
        "AND_JOIN",
        "OR_JOIN",
        "RELATES_TO",
    ],
    "Actors & Resources": [
        "RELATES_TO",
        "ISA",
        "PART_OF",
        "PLAYS_ROLE",
        "BELONGS_TO",
        "OWNS",
        "MONITORS",
        "DEPENDS_ON",
    ],
    "Technical Components & Requirements": [
        "SUPPORTS",
        "HINDERS",
        "AND_REFINES",
        "OR_REFINES",
        "AND_OR_REFINES",
        "PART_OF",
        "HAS_GOAL",
        "HAS_REQUIREMENT",
        "RELATES_TO",
    ],
}

CROSS_MODEL_RELATIONSHIPS: dict[tuple[str, str], list[str]] = {
    ("Goals", "Concepts"): ["REFERS_TO", "RELATES_TO"],
    ("Goals", "Business Processes"): ["MOTIVATES", "SUPPORTS", "HINDERS"],
    ("Goals", "Business Rules"): ["MOTIVATES", "SUPPORTS", "HINDERS"],
    ("Goals", "Actors & Resources"): ["MOTIVATES", "REQUIRES", "RELATES_TO"],
    ("Goals", "Technical Components & Requirements"): [
        "MOTIVATES",
        "SUPPORTS",
        "REQUIRES",
    ],
    ("Business Rules", "Business Processes"): [
        "TRIGGERS",
        "SUPPORTS",
        "HINDERS",
        "APPLIES_TO",
    ],
    ("Business Rules", "Concepts"): ["REFERS_TO", "RELATES_TO"],
    ("Business Rules", "Technical Components & Requirements"): [
        "MOTIVATES",
        "REQUIRES",
        "SUPPORTS",
    ],
    ("Business Processes", "Concepts"): [
        "REFERS_TO",
        "USES",
        "PRODUCES",
        "CONSUMES",
    ],
    ("Business Processes", "Technical Components & Requirements"): [
        "MOTIVATES",
        "REQUIRES",
        "USES",
        "SUPPORTS",
    ],
    ("Actors & Resources", "Goals"): ["DEFINES", "RESPONSIBLE_FOR", "SUPPORTS"],
    ("Actors & Resources", "Business Rules"): ["DEFINES", "RESPONSIBLE_FOR"],
    ("Actors & Resources", "Business Processes"): [
        "PERFORMS",
        "RESPONSIBLE_FOR",
        "SUPPORTS",
    ],
    ("Actors & Resources", "Technical Components & Requirements"): [
        "OWNS",
        "RESPONSIBLE_FOR",
        "USES",
        "INVOLVED_ACTOR",
    ],
    ("Technical Components & Requirements", "Goals"): [
        "SUPPORTS",
        "HINDERS",
        "RELATES_TO",
    ],
    ("Technical Components & Requirements", "Business Processes"): [
        "SUPPORTS",
        "APPLIES_TO",
        "USES",
    ],
    ("Technical Components & Requirements", "Concepts"): [
        "RELATES_TO",
        "USES",
    ],
    ("Business Processes", "Business Rules"): ["SUPPORTS", "APPLIES_TO"],
    ("Concepts", "Goals"): ["RELATES_TO"],
    ("Concepts", "Business Processes"): ["RELATES_TO"],
}

ALL_RELATIONSHIP_KINDS = sorted(
    {
        item
        for values in list(INTRA_MODEL_RELATIONSHIPS.values())
        + list(CROSS_MODEL_RELATIONSHIPS.values())
        for item in values
    }
    | {"RELATES_TO"}
)

ALL_ELEMENT_TYPES = sorted({t for values in ELEMENT_TYPES.values() for t in values})

PROPAGATION_INSTRUCTIONS = r"""
You are a 4EM change-propagation analyst. Evaluate exactly one source change
against exactly one target organisational unit and its current local 4EM model.
Your output must describe only the concrete local model effect. Select the target
correspondence before choosing an action, and return the smallest action set that
is actually necessary.

STEP 1 — BEST TARGET CORRESPONDENCE
Compare the source affected element, source after-state and driver with the
candidate shortlist and the complete supplied target model. Select one best
candidate_target_element_id when a credible existing counterpart exists.
Classify candidate_fit as:
- DIRECT_COUNTERPART: same 4EM function and same kind of operative obligation;
- FUNCTIONAL_ANALOGUE: corresponding local function with local terminology,
  subject domain, aggregation level, artefacts or implementation context;
- TOPIC_ONLY: wording/theme overlap without the governed function or dependency;
- NONE: no credible target function, ownership or dependency.

The shortlist is a navigation aid, not a forced answer. Same submodel and element
type are substantive evidence, but never sufficient by themselves. Organisational
proximity and notification receipt are also insufficient.

STEP 2 — APPLICABILITY
A change applies only when the target performs, owns, consumes, produces, depends
on, or is constrained by an operative clause in the source after-state. Do not
reject merely because local terminology differs when a functional analogue exists.
Do reject source-domain-specific obligations when the target has no corresponding
activity or dependency.

STEP 3 — SMALLEST ACTION SET
- When nothing applies, return propagation_applicability=NONE and exactly one
  NO_CHANGE action.
- When something applies, return propagation_applicability=APPLIES and one to five
  concrete actions, with no NO_CHANGE action.
- Prefer exactly one UPDATE_ELEMENT of the best existing counterpart.
- UPDATE_ELEMENT must use an ID supplied in target_4em_elements; normally it should
  equal candidate_target_element_id.
- Use CREATE_ELEMENT only when no existing element can represent the required state.
- Use CREATE_RELATION only when the source change specifically requires missing
  relationship topology. Do not add speculative supporting relations.
- Return multiple actions only when independent target elements truly require change.
- Preserve the source rationale in each action driver, localised to target evidence.

For unused action fields required by the schema, return an empty string, empty list,
or the nearest neutral enum value. For NO_CHANGE use submodel=Goals,
element_type=Goal, model_status=AS_IS, relationship_kind=RELATES_TO, and empty IDs.
Keep explanations concise and evidence-based.
""".strip()


# Shared utilities


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


def make_4em_code(element_type: str) -> str:
    number = uuid.uuid4().int % 1_000_000_000_000
    return f"{TYPE_PREFIX.get(element_type, element_type)} {number:012d}"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def is_placeholder(value: str) -> bool:
    upper = (value or "").upper()
    return not value or "PASTE_" in upper or "YOUR_AURA_INSTANCE" in upper


def clean_tags(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        parts = raw
    else:
        parts = raw.split(",")
    return sorted({p.strip() for p in parts if p and p.strip()})


def dot_escape(value: Any) -> str:
    text = str(value or "")
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def safe_dot_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def element_label(element: dict[str, Any], unit_names: dict[str, str] | None = None) -> str:
    unit_suffix = ""
    if unit_names:
        unit_suffix = f" — {unit_names.get(element.get('unit_id'), element.get('unit_id', ''))}"
    return f"[{element.get('code', element.get('id', ''))}] {element.get('title', '')}{unit_suffix}"


def relationship_label(rel: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]) -> str:
    source = elements_by_id.get(rel.get("source_id"), {})
    target = elements_by_id.get(rel.get("target_id"), {})
    return (
        f"{source.get('code', rel.get('source_id', ''))} —{rel.get('kind', '')}→ "
        f"{target.get('code', rel.get('target_id', ''))}"
    )


def allowed_relationship_kinds(
    source: dict[str, Any], target: dict[str, Any]
) -> list[str]:
    source_model = source.get("submodel", "")
    target_model = target.get("submodel", "")
    if source_model == target_model:
        return INTRA_MODEL_RELATIONSHIPS.get(source_model, ["RELATES_TO"])
    return CROSS_MODEL_RELATIONSHIPS.get(
        (source_model, target_model), ["RELATES_TO"]
    )


def render_relationship_notation_inputs(
    source: dict[str, Any],
    target: dict[str, Any],
    kind: str,
    key_prefix: str,
) -> dict[str, str]:
    """Render controls for relationship-specific 4EM notation metadata.

    The metadata affects presentation, not relationship semantics. It captures
    influence strength, Concept Model cardinality, and decomposition totality.
    """
    notation: dict[str, str] = {}

    if kind in {"SUPPORTS", "HINDERS", "CONFLICTS"}:
        notation["strength"] = st.selectbox(
            "Influence strength (optional)",
            RELATION_STRENGTHS,
            format_func=lambda value: "Not specified" if not value else value.title(),
            key=f"{key_prefix}_strength",
        )

    if (
        source.get("submodel") == "Concepts"
        and target.get("submodel") == "Concepts"
        and kind == "RELATES_TO"
    ):
        left, right = st.columns(2)
        with left:
            notation["source_cardinality"] = st.selectbox(
                "Source multiplicity",
                CONCEPT_CARDINALITIES,
                format_func=lambda value: "Not specified" if not value else value,
                key=f"{key_prefix}_source_cardinality",
            )
        with right:
            notation["target_cardinality"] = st.selectbox(
                "Target multiplicity",
                CONCEPT_CARDINALITIES,
                format_func=lambda value: "Not specified" if not value else value,
                key=f"{key_prefix}_target_cardinality",
            )

    if kind in {"ISA", "PART_OF"}:
        notation["totality"] = st.selectbox(
            "Decomposition coverage",
            RELATION_TOTALITIES,
            format_func=lambda value: value.title(),
            key=f"{key_prefix}_totality",
            help=(
                "Partial uses an open circle/square; total uses a filled circle/square, "
                "matching the 4EM Concepts and Actors/Resources notation."
            ),
        )

    return {key: value for key, value in notation.items() if value}


def _wrap_notation_text(value: Any, width: int, max_lines: int) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return [""]
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = textwrap.shorten(lines[-1], width=max(5, width), placeholder="…")
    return lines


def _svg_text(lines: list[str], start_y: int, line_height: int, fill: str = "#202124", size: int = 13) -> str:
    parts: list[str] = []
    for index, line in enumerate(lines):
        y = start_y + index * line_height
        parts.append(
            f'<text x="140" y="{y}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{fill}">'
            f'{html_lib.escape(line)}</text>'
        )
    return "".join(parts)


def _svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


def _notation_node_svg(element: dict[str, Any], decomposed: bool = False) -> str:
    """Build a data-URI SVG using the canonical 4EM element notation."""
    element_type = str(element.get("element_type", ""))
    code = str(element.get("code", element_type or "4EM component"))
    if decomposed and "(+)" not in code:
        code = f"{code} (+)"
    title = str(element.get("title", ""))
    code_lines = _wrap_notation_text(code, 31, 1)
    title_lines = _wrap_notation_text(title, 34, 3)
    line = BOOK_4EM_COLOURS["line"]
    shape = ""
    text_fill = "#202124"
    code_y = 42
    title_y = 66

    if element_type == "Goal":
        shape = f'<rect x="5" y="5" width="270" height="110" fill="{BOOK_4EM_COLOURS["goal"]}" stroke="{line}" stroke-width="1.4"/>'
    elif element_type in {"Problem", "Weakness", "Threat"}:
        shape = f'<rect x="5" y="5" width="270" height="110" fill="{BOOK_4EM_COLOURS["problem"]}" stroke="{line}" stroke-width="1.4"/>'
    elif element_type == "Cause":
        shape = f'<rect x="5" y="5" width="270" height="110" fill="{BOOK_4EM_COLOURS["cause"]}" stroke="{line}" stroke-width="1.4"/>'
    elif element_type == "Constraint":
        shape = f'<rect x="5" y="5" width="270" height="110" fill="#FFFFFF" stroke="{BOOK_4EM_COLOURS["constraint_border"]}" stroke-width="2"/>'
    elif element_type == "Opportunity":
        shape = f'<rect x="5" y="5" width="270" height="110" fill="{BOOK_4EM_COLOURS["opportunity"]}" stroke="{line}" stroke-width="1.4"/>'
        text_fill = "#FFFFFF"
    elif element_type in {
        "BusinessRule",
        "DerivationRule",
        "EventActionRule",
        "StaticConstraintRule",
        "TransitionConstraintRule",
    }:
        shape = f'<rect x="5" y="5" width="270" height="110" fill="{BOOK_4EM_COLOURS["rule"]}" stroke="#A7A0AD" stroke-width="1.3"/>'
    elif element_type == "Concept":
        shape = f'<rect x="5" y="10" width="270" height="100" rx="38" ry="38" fill="{BOOK_4EM_COLOURS["concept"]}" stroke="{BOOK_4EM_COLOURS["concept_border"]}" stroke-width="2.2"/>'
    elif element_type == "Attribute":
        shape = f'<rect x="18" y="28" width="244" height="64" rx="30" ry="30" fill="#FFFFFF" stroke="{BOOK_4EM_COLOURS["concept_border"]}" stroke-width="2.2"/>'
        code_lines = []
        title_lines = _wrap_notation_text(title or code, 35, 2)
        title_y = 62
    elif element_type == "Process":
        shape = (
            f'<rect x="5" y="8" width="270" height="104" rx="42" ry="42" fill="#FFFFFF" stroke="{line}" stroke-width="1.6"/>'
            f'<line x1="8" y1="50" x2="272" y2="50" stroke="{line}" stroke-width="1.2"/>'
        )
        code_y = 36
        title_y = 75
    elif element_type == "ExternalProcess":
        shape = (
            '<rect x="18" y="20" width="250" height="88" fill="#8C8C8C" opacity="0.65"/>'
            f'<rect x="5" y="7" width="250" height="88" fill="#FFFFFF" stroke="{line}" stroke-width="1.6"/>'
        )
        code_y = 38
        title_y = 62
    elif element_type in {"InformationSet", "MaterialSet"}:
        shape = f'<polygon points="30,12 274,12 248,108 5,108" fill="#FFFFFF" stroke="{line}" stroke-width="1.6"/>'
    elif element_type == "Individual":
        shape = f'<polygon points="30,5 275,5 275,115 5,115 5,30" fill="{BOOK_4EM_COLOURS["individual"]}" stroke="{line}" stroke-width="1.4"/>'
    elif element_type == "Role":
        shape = f'<polygon points="30,5 275,5 275,115 5,115 5,30" fill="{BOOK_4EM_COLOURS["role"]}" stroke="#5F4A78" stroke-width="1.4"/>'
        text_fill = "#FFFFFF"
    elif element_type == "NonHumanResource":
        shape = f'<rect x="5" y="5" width="270" height="110" fill="{BOOK_4EM_COLOURS["resource"]}" stroke="{line}" stroke-width="1.4"/>'
    elif element_type == "OrganisationalUnit":
        shape = f'<rect x="5" y="5" width="270" height="110" fill="#FFFFFF" stroke="{BOOK_4EM_COLOURS["org_unit_border"]}" stroke-width="1.8"/>'
    elif element_type == "TechnicalComponent":
        shape = f'<ellipse cx="140" cy="60" rx="132" ry="52" fill="#FFFFFF" stroke="{line}" stroke-width="1.6"/>'
    elif element_type in {"ISRequirement", "FunctionalRequirement", "NonFunctionalRequirement"}:
        shape = f'<ellipse cx="140" cy="60" rx="132" ry="52" fill="#FFFFFF" stroke="{line}" stroke-width="1.6" stroke-dasharray="6 5"/>'
    elif element_type == "ISGoal":
        shape = f'<ellipse cx="140" cy="60" rx="132" ry="52" fill="{BOOK_4EM_COLOURS["goal"]}" stroke="{line}" stroke-width="1.6"/>'
    elif element_type == "ISProblem":
        shape = f'<ellipse cx="140" cy="60" rx="132" ry="52" fill="{BOOK_4EM_COLOURS["problem"]}" stroke="{line}" stroke-width="1.6"/>'
    else:
        shape = f'<rect x="5" y="5" width="270" height="110" fill="#FFFFFF" stroke="{line}" stroke-width="1.4"/>'

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="280" height="120" viewBox="0 0 280 120">'
        f'{shape}'
        f'{_svg_text(code_lines, code_y, 16, text_fill, 12)}'
        f'{_svg_text(title_lines, title_y, 16, text_fill, 13)}'
        '</svg>'
    )
    return _svg_data_uri(svg)


def tokenise(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "with",
        "is",
        "are",
        "be",
        "by",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in stop
    }


def lexical_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_tokens = tokenise(f"{a.get('title', '')} {a.get('description', '')}")
    b_tokens = tokenise(f"{b.get('title', '')} {b.get('description', '')}")
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def parse_json_object(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, dict) else value


_CANDIDATE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "on",
    "by", "from", "as", "is", "are", "be", "must", "every", "this", "that",
    "its", "their", "through", "using", "used", "when", "before", "after",
    "into", "across", "any", "all", "each", "course", "programme", "program",
    "curriculum", "model", "unit", "local", "approved", "current", "ensure",
    "preserve", "maintain", "rtu", "study",
}


def compact_unit_for_llm(unit: dict[str, Any]) -> dict[str, Any]:
    compact = dict(unit)
    if "metadata_json" in compact:
        compact["metadata"] = parse_json_object(compact.pop("metadata_json"))
    return compact


def organisational_relation(
    source_unit: dict[str, Any], target_unit: dict[str, Any]
) -> str:
    source_id = str(source_unit.get("id", ""))
    target_id = str(target_unit.get("id", ""))
    source_parent = str(source_unit.get("parent_id", ""))
    target_parent = str(target_unit.get("parent_id", ""))
    if source_id and source_id == target_id:
        return "SAME_UNIT"
    if source_id and source_id == target_parent:
        return "SOURCE_IS_PARENT"
    if target_id and target_id == source_parent:
        return "TARGET_IS_PARENT"
    if source_parent and source_parent == target_parent:
        return "SIBLING_UNITS"
    return "OTHER"


def semantic_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if token in _CANDIDATE_STOPWORDS or len(token) < 3:
            continue
        if re.fullmatch(r"de\d+", token):
            continue
        tokens.add(token)
    return tokens


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def rank_target_candidates(
    source_change: dict[str, Any],
    target_elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank target elements and return the deterministic evaluator shortlist."""
    source_element = source_change.get("affected_element") or {}
    source_title_tokens = semantic_tokens(source_element.get("title"))
    source_semantic_tokens = (
        semantic_tokens(source_element.get("description"))
        | semantic_tokens(source_change.get("after_state"))
        | semantic_tokens(source_change.get("driver"))
    )
    source_tags = semantic_tokens(source_element.get("tags"))

    ranked: list[dict[str, Any]] = []
    for element in target_elements:
        score = 0.0
        signals: list[str] = []
        if element.get("submodel") == source_element.get("submodel"):
            score += 6.0
            signals.append("same_submodel")
        if element.get("element_type") == source_element.get("element_type"):
            score += 5.0
            signals.append("same_element_type")

        title_overlap = jaccard(
            source_title_tokens, semantic_tokens(element.get("title"))
        )
        semantic_overlap = jaccard(
            source_semantic_tokens,
            semantic_tokens(element.get("title"))
            | semantic_tokens(element.get("description"))
            | semantic_tokens(element.get("driver")),
        )
        tag_overlap = jaccard(source_tags, semantic_tokens(element.get("tags")))
        score += 10.0 * title_overlap + 5.0 * semantic_overlap + tag_overlap
        if title_overlap:
            signals.append(f"title_overlap={title_overlap:.3f}")
        if semantic_overlap:
            signals.append(f"semantic_overlap={semantic_overlap:.3f}")
        if tag_overlap:
            signals.append(f"tag_overlap={tag_overlap:.3f}")

        ranked.append(
            {
                "id": element.get("id"),
                "submodel": element.get("submodel"),
                "element_type": element.get("element_type"),
                "title": element.get("title"),
                "description": element.get("description"),
                "heuristic_score": round(score, 6),
                "matching_signals": signals,
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item.get("heuristic_score", 0.0)),
            str(item.get("submodel", "")),
            str(item.get("element_type", "")),
            str(item.get("id", "")),
        )
    )
    return ranked[:CANDIDATE_SHORTLIST_SIZE]


def prioritise_elements_for_context(
    target_elements: list[dict[str, Any]],
    candidate_shortlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a bounded LLM context that always retains shortlisted candidates."""
    by_id = {
        str(element.get("id")): element
        for element in target_elements
        if element.get("id")
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidate_shortlist:
        candidate_id = str(candidate.get("id", ""))
        if candidate_id in by_id and candidate_id not in seen:
            selected.append(by_id[candidate_id])
            seen.add(candidate_id)
    for element in target_elements:
        element_id = str(element.get("id", ""))
        if element_id and element_id not in seen:
            selected.append(element)
            seen.add(element_id)
        if len(selected) >= MAX_LLM_ELEMENTS:
            break
    return selected[:MAX_LLM_ELEMENTS]


def build_propagation_output_schema() -> dict[str, Any]:
    action_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["CREATE_ELEMENT", "UPDATE_ELEMENT", "CREATE_RELATION", "NO_CHANGE"],
            },
            "existing_element_id": {"type": "string"},
            "submodel": {"type": "string", "enum": list(SUBMODELS)},
            "element_type": {"type": "string", "enum": ALL_ELEMENT_TYPES},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "model_status": {"type": "string", "enum": MODEL_STATUSES},
            "priority": {"type": "string", "enum": PRIORITIES},
            "criticality": {"type": "string", "enum": CRITICALITIES},
            "tags": {"type": "array", "items": {"type": "string"}},
            "relationship_kind": {"type": "string", "enum": ALL_RELATIONSHIP_KINDS},
            "relation_source_id": {"type": "string"},
            "relation_target_id": {"type": "string"},
            "driver": {"type": "string"},
        },
        "required": [
            "action", "existing_element_id", "submodel", "element_type", "title",
            "description", "model_status", "priority", "criticality", "tags",
            "relationship_kind", "relation_source_id", "relation_target_id", "driver",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "candidate_target_element_id": {"type": "string"},
            "candidate_fit": {
                "type": "string",
                "enum": ["NONE", "TOPIC_ONLY", "FUNCTIONAL_ANALOGUE", "DIRECT_COUNTERPART"],
            },
            "candidate_fit_explanation": {"type": "string"},
            "propagation_applicability": {
                "type": "string",
                "enum": ["NONE", "APPLIES"],
            },
            "target_obligation_summary": {"type": "string"},
            "target_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "rationale": {"type": "string"},
            "proposed_actions": {
                "type": "array",
                "items": action_schema,
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": [
            "candidate_target_element_id", "candidate_fit", "candidate_fit_explanation",
            "propagation_applicability", "target_obligation_summary",
            "target_evidence_ids", "rationale", "proposed_actions",
        ],
        "additionalProperties": False,
    }


PROPAGATION_OUTPUT_SCHEMA = build_propagation_output_schema()


def validate_propagation_analysis(
    analysis: dict[str, Any],
    target_unit_id: str,
    target_elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate proposed actions and derive applicability from their model effect."""
    actions = analysis.get("proposed_actions")
    if not isinstance(actions, list) or not actions or not all(
        isinstance(action, dict) for action in actions
    ):
        raise ValueError("proposed_actions must be a non-empty list of objects")

    target_by_id = {
        str(element.get("id")): element
        for element in target_elements
        if element.get("id")
    }
    target_ids = set(target_by_id)
    notes: list[str] = []

    candidate_id = str(analysis.get("candidate_target_element_id", "")).strip()
    if candidate_id and candidate_id not in target_ids:
        notes.append(f"Ignored candidate ID absent from target model: {candidate_id}")
        candidate_id = ""
    analysis["candidate_target_element_id"] = candidate_id

    evidence_ids = {
        str(value).strip()
        for value in analysis.get("target_evidence_ids", [])
        if str(value).strip()
    }
    invalid_evidence = sorted(evidence_ids - target_ids)
    if invalid_evidence:
        notes.append(
            "Ignored evidence IDs absent from target model: " + ", ".join(invalid_evidence)
        )
    analysis["target_evidence_ids"] = sorted(evidence_ids & target_ids)

    no_change = [
        action for action in actions
        if str(action.get("action", "")).upper() == "NO_CHANGE"
    ]
    concrete = [
        action for action in actions
        if str(action.get("action", "")).upper()
        in {"CREATE_ELEMENT", "UPDATE_ELEMENT", "CREATE_RELATION"}
    ]
    if concrete:
        if no_change:
            notes.append("Removed mixed NO_CHANGE because concrete actions were present")
        selected_actions = concrete
        relevant = True
        if str(analysis.get("propagation_applicability", "")).upper() != "APPLIES":
            notes.append("Normalised propagation_applicability to APPLIES from actions")
        analysis["propagation_applicability"] = "APPLIES"
    elif len(no_change) == 1:
        selected_actions = no_change
        relevant = False
        if str(analysis.get("propagation_applicability", "")).upper() != "NONE":
            notes.append("Normalised propagation_applicability to NONE from NO_CHANGE")
        analysis["propagation_applicability"] = "NONE"
    else:
        raise ValueError("Response must contain concrete actions or exactly one NO_CHANGE")

    validated: list[dict[str, Any]] = []
    impacted: set[str] = set()
    for raw in selected_actions:
        action = dict(raw)
        kind = str(action.get("action", "")).upper()
        driver = str(action.get("driver", "")).strip()

        if kind == "NO_CHANGE":
            validated.append(action)
            continue
        if not driver:
            raise ValueError(f"{kind} requires a driver")

        if kind == "CREATE_ELEMENT":
            submodel = str(action.get("submodel", ""))
            element_type = str(action.get("element_type", ""))
            if submodel not in ELEMENT_TYPES:
                raise ValueError(f"Invalid submodel: {submodel!r}")
            if element_type not in ELEMENT_TYPES[submodel]:
                raise ValueError(
                    f"Invalid element type {element_type!r} for {submodel!r}"
                )
            if not str(action.get("title", "")).strip():
                raise ValueError("CREATE_ELEMENT requires a title")

        elif kind == "UPDATE_ELEMENT":
            element_id = str(action.get("existing_element_id", "")).strip()
            if element_id not in target_ids:
                raise ValueError(f"UPDATE_ELEMENT uses unknown target ID: {element_id!r}")
            impacted.add(element_id)
            actual_element = target_by_id[element_id]
            actual_submodel = str(actual_element.get("submodel", ""))
            actual_type = str(actual_element.get("element_type", ""))
            if action.get("submodel") != actual_submodel:
                notes.append(
                    f"Normalised UPDATE_ELEMENT {element_id} submodel to {actual_submodel}"
                )
                action["submodel"] = actual_submodel
            if action.get("element_type") != actual_type:
                notes.append(
                    f"Normalised UPDATE_ELEMENT {element_id} type to {actual_type}"
                )
                action["element_type"] = actual_type

        elif kind == "CREATE_RELATION":
            source_id = str(action.get("relation_source_id", "")).strip()
            target_id = str(action.get("relation_target_id", "")).strip()
            if source_id not in target_ids or target_id not in target_ids:
                raise ValueError("CREATE_RELATION endpoints must be target model element IDs")
            source_element = target_by_id[source_id]
            target_element = target_by_id[target_id]
            if str(source_element.get("unit_id", "")) != target_unit_id:
                raise ValueError("CREATE_RELATION source endpoint must belong to target unit")
            relationship_kind = str(action.get("relationship_kind", ""))
            if relationship_kind not in allowed_relationship_kinds(
                source_element, target_element
            ):
                raise ValueError(
                    f"Invalid relationship {relationship_kind!r} between "
                    f"{source_element.get('submodel')} and {target_element.get('submodel')}"
                )
            impacted.update({source_id, target_id})

        validated.append(action)

    action_labels = [str(action.get("action", "")).replace("_", " ").title() for action in validated]
    if relevant:
        candidate = target_by_id.get(candidate_id, {})
        candidate_name = candidate.get("title") or candidate.get("code") or "the target model"
        title = f"Adapt {candidate_name}" if candidate_id else "Create local 4EM adaptation"
        target_impact = "; ".join(action_labels)
    else:
        title = "No local model change recommended"
        target_impact = "No model operation is necessary for this target unit."

    return {
        **analysis,
        "title": title,
        "summary": str(analysis.get("target_obligation_summary", "")),
        "target_impact": target_impact,
        "relevant": relevant,
        "confidence": 0.0,
        "projection_notes": notes,
        "projected_impacted_element_ids": sorted(impacted),
        "proposed_actions": validated,
    }


# Neo4j persistence


class GraphRepository:
    def __init__(self, driver: Any, database: str):
        self.driver = driver
        self.database = database

    def run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            query,
            parameters_=parameters,
            database_=self.database,
        )
        return [plain(record.data()) for record in records]

    def bootstrap_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT org_unit_id_unique IF NOT EXISTS FOR (u:OrgUnit) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT element_id_unique IF NOT EXISTS FOR (e:Element) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT element_unit_code_unique IF NOT EXISTS FOR (e:Element) REQUIRE (e.unit_id, e.code) IS UNIQUE",
            "CREATE CONSTRAINT change_id_unique IF NOT EXISTS FOR (c:Change) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT suggestion_id_unique IF NOT EXISTS FOR (s:PropagationSuggestion) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT notification_id_unique IF NOT EXISTS FOR (n:ChangeNotification) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT notification_change_target_unique IF NOT EXISTS FOR (n:ChangeNotification) REQUIRE (n.change_id, n.target_unit_id) IS UNIQUE",
            "CREATE INDEX element_unit_index IF NOT EXISTS FOR (e:Element) ON (e.unit_id)",
            "CREATE INDEX element_submodel_index IF NOT EXISTS FOR (e:Element) ON (e.submodel)",
            "CREATE INDEX change_unit_index IF NOT EXISTS FOR (c:Change) ON (c.unit_id)",
            "CREATE INDEX suggestion_target_index IF NOT EXISTS FOR (s:PropagationSuggestion) ON (s.target_unit_id)",
            "CREATE INDEX notification_target_index IF NOT EXISTS FOR (n:ChangeNotification) ON (n.target_unit_id)",
            "CREATE INDEX notification_status_index IF NOT EXISTS FOR (n:ChangeNotification) ON (n.status)",
        ]
        for statement in statements:
            self.run(statement)

        # Prime schema tokens referenced by notification queries before domain data
        # exists. Neo4j retains them after the temporary records are deleted, which
        # prevents misleading warnings on a newly provisioned database.
        self.run(
            """
            CREATE (a:__SchemaTokenSeed {
                id: 'notification-schema-source',
                change_id: '',
                source_unit_id: '',
                target_unit_id: '',
                notification_id: '',
                status: ''
            })
            CREATE (b:__SchemaTokenSeed {id: 'notification-schema-target'})
            CREATE (a)-[:ABOUT_CHANGE]->(b)
            CREATE (a)-[:FROM_UNIT]->(b)
            CREATE (a)-[:FOR_UNIT]->(b)
            CREATE (a)-[:GENERATED_FROM]->(b)
            CREATE (a)-[:BASED_ON]->(b)
            DETACH DELETE a, b
            """
        )

    # Organisational units

    def list_units(self) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (u:OrgUnit)
            OPTIONAL MATCH (u)-[:PART_OF]->(parent:OrgUnit)
            RETURN properties(u) AS unit, parent.id AS parent_id, parent.name AS parent_name
            ORDER BY coalesce(u.level, ''), u.name
            """
        )
        return [
            {**row["unit"], "parent_id": row.get("parent_id"), "parent_name": row.get("parent_name")}
            for row in rows
        ]

    def get_unit(self, unit_id: str) -> dict[str, Any] | None:
        rows = self.run(
            """
            MATCH (u:OrgUnit {id: $unit_id})
            OPTIONAL MATCH (u)-[:PART_OF]->(parent:OrgUnit)
            RETURN properties(u) AS unit, parent.id AS parent_id, parent.name AS parent_name
            """,
            unit_id=unit_id,
        )
        if not rows:
            return None
        return {
            **rows[0]["unit"],
            "parent_id": rows[0].get("parent_id"),
            "parent_name": rows[0].get("parent_name"),
        }

    def remember_display_name(self, unit_id: str, display_name: str) -> None:
        """Persist the unit-level display name used by the accountless prototype."""
        cleaned = display_name.strip() or "Unit modeller"
        rows = self.run(
            """
            MATCH (u:OrgUnit {id: $unit_id})
            SET u.last_display_name = $display_name,
                u.updated_at = $now
            RETURN u.id AS id
            """,
            unit_id=unit_id,
            display_name=cleaned,
            now=utc_now(),
        )
        if not rows:
            raise ValueError("The organisational unit could not be found.")

    def create_unit(
        self,
        name: str,
        level: str,
        description: str,
        parent_id: str | None,
    ) -> dict[str, Any]:
        unit_id = new_id("unit")
        props = {
            "id": unit_id,
            "name": name.strip(),
            "level": level.strip() or "Organisational unit",
            "description": description.strip(),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        if parent_id:
            rows = self.run(
                """
                MATCH (parent:OrgUnit {id: $parent_id})
                CREATE (u:OrgUnit)
                SET u = $props
                CREATE (u)-[:PART_OF]->(parent)
                RETURN properties(u) AS unit
                """,
                parent_id=parent_id,
                props=props,
            )
        else:
            rows = self.run(
                """
                CREATE (u:OrgUnit)
                SET u = $props
                RETURN properties(u) AS unit
                """,
                props=props,
            )
        if not rows:
            raise RuntimeError("The organisational unit could not be created.")
        return rows[0]["unit"]

    # Change notifications

    def create_change_notifications(self, change_id: str, source_unit_id: str) -> int:
        """Notify sibling units and the immediate parent of a source-unit change.

        Recipients are resolved from the ``PART_OF`` hierarchy rather than the
        display-oriented ``level`` property.
        """
        rows = self.run(
            """
            MATCH (c:Change {id: $change_id})
            MATCH (source:OrgUnit {id: $source_unit_id})
            OPTIONAL MATCH (source)-[:PART_OF]->(parent:OrgUnit)
            OPTIONAL MATCH (sibling:OrgUnit)-[:PART_OF]->(parent)
            WITH c, source, parent,
                 [u IN collect(DISTINCT sibling) WHERE u.id <> source.id] +
                 CASE WHEN parent IS NULL THEN [] ELSE [parent] END AS recipients
            UNWIND recipients AS target
            WITH DISTINCT c, source, target
            MERGE (n:ChangeNotification {
                change_id: $change_id,
                target_unit_id: target.id
            })
            ON CREATE SET n.id = randomUUID(),
                          n.source_unit_id = source.id,
                          n.status = 'UNREAD',
                          n.created_at = $now,
                          n.updated_at = $now,
                          n.read_at = '',
                          n.analysis_requested_at = '',
                          n.dismissed_at = ''
            MERGE (n)-[:ABOUT_CHANGE]->(c)
            MERGE (n)-[:FROM_UNIT]->(source)
            MERGE (n)-[:FOR_UNIT]->(target)
            RETURN count(n) AS created_or_matched
            """,
            change_id=change_id,
            source_unit_id=source_unit_id,
            now=utc_now(),
        )
        return int(rows[0]["created_or_matched"]) if rows else 0

    def list_notifications(
        self,
        target_unit_id: str,
        statuses: list[str] | None = None,
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        statuses = statuses or []
        rows = self.run(
            """
            MATCH (n:ChangeNotification)-[:ABOUT_CHANGE]->(c:Change)
            MATCH (n)-[:FROM_UNIT]->(source:OrgUnit)
            MATCH (n)-[:FOR_UNIT]->(target:OrgUnit)
            OPTIONAL MATCH (s:PropagationSuggestion)-[:GENERATED_FROM]->(n)
            WHERE n.target_unit_id = $target_unit_id
              AND (size($statuses) = 0 OR n.status IN $statuses)
            RETURN properties(n) AS notification,
                   properties(c) AS change,
                   source.name AS source_unit_name,
                   target.name AS target_unit_name,
                   s.id AS suggestion_id,
                   s.status AS suggestion_status,
                   s.title AS suggestion_title
            ORDER BY n.created_at DESC
            LIMIT $limit
            """,
            target_unit_id=target_unit_id,
            statuses=statuses,
            limit=limit,
        )
        return [
            {
                **row["notification"],
                "change": row.get("change") or {},
                "source_unit_name": row.get("source_unit_name"),
                "target_unit_name": row.get("target_unit_name"),
                "suggestion_id": row.get("suggestion_id"),
                "suggestion_status": row.get("suggestion_status"),
                "suggestion_title": row.get("suggestion_title"),
            }
            for row in rows
        ]

    def get_notification(
        self, notification_id: str, target_unit_id: str
    ) -> dict[str, Any] | None:
        rows = self.run(
            """
            MATCH (n:ChangeNotification {id: $notification_id, target_unit_id: $target_unit_id})
                  -[:ABOUT_CHANGE]->(c:Change)
            MATCH (n)-[:FROM_UNIT]->(source:OrgUnit)
            MATCH (n)-[:FOR_UNIT]->(target:OrgUnit)
            OPTIONAL MATCH (s:PropagationSuggestion)-[:GENERATED_FROM]->(n)
            RETURN properties(n) AS notification,
                   properties(c) AS change,
                   properties(source) AS source_unit,
                   properties(target) AS target_unit,
                   properties(s) AS suggestion
            """,
            notification_id=notification_id,
            target_unit_id=target_unit_id,
        )
        if not rows:
            return None
        return {
            **rows[0]["notification"],
            "change": rows[0].get("change") or {},
            "source_unit": rows[0].get("source_unit") or {},
            "target_unit": rows[0].get("target_unit") or {},
            "suggestion": rows[0].get("suggestion") or {},
        }

    def update_notification_status(
        self, notification_id: str, target_unit_id: str, status: str
    ) -> None:
        allowed = {"UNREAD", "READ", "ANALYSED", "ADAPTING", "ADAPTED", "DISMISSED"}
        if status not in allowed:
            raise ValueError("Invalid notification status.")
        timestamp_field = {
            "READ": "read_at",
            "ANALYSED": "analysis_requested_at",
            "ADAPTING": "manual_adaptation_started_at",
            "ADAPTED": "manual_adaptation_completed_at",
            "DISMISSED": "dismissed_at",
        }.get(status)
        query = """
            MATCH (n:ChangeNotification {id: $notification_id, target_unit_id: $target_unit_id})
            SET n.status = $status,
                n.updated_at = $now
        """
        if timestamp_field:
            query += f", n.{timestamp_field} = $now"
        self.run(
            query,
            notification_id=notification_id,
            target_unit_id=target_unit_id,
            status=status,
            now=utc_now(),
        )

    # Model elements

    def list_elements(
        self,
        unit_id: str | None = None,
        submodel: str | None = None,
        include_inactive: bool = False,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (e:Element)
            WHERE ($unit_id IS NULL OR e.unit_id = $unit_id)
              AND ($submodel IS NULL OR e.submodel = $submodel)
              AND ($include_inactive OR coalesce(e.active, true) = true)
            RETURN properties(e) AS element
            ORDER BY e.submodel, e.element_type, e.code, e.id
            LIMIT $limit
            """,
            unit_id=unit_id,
            submodel=submodel,
            include_inactive=include_inactive,
            limit=limit,
        )
        return [row["element"] for row in rows]

    def get_element(self, element_id: str) -> dict[str, Any] | None:
        rows = self.run(
            "MATCH (e:Element {id: $element_id}) RETURN properties(e) AS element",
            element_id=element_id,
        )
        return rows[0]["element"] if rows else None

    def create_element(
        self,
        unit_id: str,
        actor_name: str,
        submodel: str,
        element_type: str,
        title: str,
        description: str,
        model_status: str,
        priority: str,
        criticality: str,
        tags: list[str],
        driver: str,
        source_change_id: str = "",
    ) -> dict[str, Any]:
        if submodel not in ELEMENT_TYPES or element_type not in ELEMENT_TYPES[submodel]:
            raise ValueError("The selected element type is not valid for that 4EM sub-model.")
        if not title.strip():
            raise ValueError("A title is required.")
        if not driver.strip():
            raise ValueError("A rationale/driver is required for every change.")

        now = utc_now()
        element_id = new_id("element")
        props = {
            "id": element_id,
            "code": make_4em_code(element_type),
            "unit_id": unit_id,
            "submodel": submodel,
            "element_type": element_type,
            "title": title.strip(),
            "description": description.strip(),
            "model_status": model_status,
            "priority": priority,
            "criticality": criticality,
            "tags": tags,
            "active": True,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "last_change": now,
            "change_type": "CREATE",
            "driver": driver.strip(),
        }
        change = {
            "id": new_id("change"),
            "unit_id": unit_id,
            "actor_name": actor_name,
            "operation": "CREATE",
            "entity_kind": "ELEMENT",
            "entity_id": element_id,
            "entity_code": props["code"],
            "entity_name": props["title"],
            "submodel": submodel,
            "element_type": element_type,
            "driver": driver.strip(),
            "before_json": "{}",
            "after_json": json_text(props),
            "source_change_id": source_change_id,
            "created_at": now,
        }
        rows = self.run(
            """
            MATCH (u:OrgUnit {id: $unit_id})
            CREATE (e:Element)
            SET e = $props
            CREATE (e)-[:OWNED_BY]->(u)
            CREATE (c:Change)
            SET c = $change
            CREATE (c)-[:AFFECTS]->(e)
            CREATE (c)-[:MADE_BY]->(u)
            RETURN properties(e) AS element
            """,
            unit_id=unit_id,
            props=props,
            change=change,
        )
        if not rows:
            raise RuntimeError("Element creation failed. Check that the organisational unit exists.")
        self.create_change_notifications(change["id"], unit_id)
        return rows[0]["element"]

    def update_element(
        self,
        element_id: str,
        unit_id: str,
        actor_name: str,
        fields: dict[str, Any],
        driver: str,
        expected_version: int,
        source_change_id: str = "",
    ) -> dict[str, Any]:
        current = self.get_element(element_id)
        if not current:
            raise ValueError("Element not found.")
        if current.get("unit_id") != unit_id:
            raise PermissionError("Only the owning organisational unit can edit this element.")
        if not driver.strip():
            raise ValueError("A rationale/driver is required for every change.")

        allowed_fields = {
            "title",
            "description",
            "model_status",
            "priority",
            "criticality",
            "tags",
        }
        updates = {k: v for k, v in fields.items() if k in allowed_fields}
        if not str(updates.get("title", current.get("title", ""))).strip():
            raise ValueError("A title is required.")

        now = utc_now()
        updates.update(
            {
                "updated_at": now,
                "last_change": now,
                "change_type": "UPDATE",
                "driver": driver.strip(),
                "version": expected_version + 1,
            }
        )
        after = {**current, **updates}
        change = {
            "id": new_id("change"),
            "unit_id": unit_id,
            "actor_name": actor_name,
            "operation": "UPDATE",
            "entity_kind": "ELEMENT",
            "entity_id": element_id,
            "entity_code": current.get("code", ""),
            "entity_name": updates.get("title", current.get("title", "")),
            "submodel": current.get("submodel", ""),
            "element_type": current.get("element_type", ""),
            "driver": driver.strip(),
            "before_json": json_text(current),
            "after_json": json_text(after),
            "source_change_id": source_change_id,
            "created_at": now,
        }
        rows = self.run(
            """
            MATCH (e:Element {id: $element_id, unit_id: $unit_id})
            WHERE e.version = $expected_version
            SET e += $updates
            WITH e
            MATCH (u:OrgUnit {id: $unit_id})
            CREATE (c:Change)
            SET c = $change
            CREATE (c)-[:AFFECTS]->(e)
            CREATE (c)-[:MADE_BY]->(u)
            RETURN properties(e) AS element
            """,
            element_id=element_id,
            unit_id=unit_id,
            expected_version=expected_version,
            updates=updates,
            change=change,
        )
        if not rows:
            raise RuntimeError(
                "The element changed after it was loaded. Refresh and try the edit again."
            )
        self.create_change_notifications(change["id"], unit_id)
        return rows[0]["element"]

    def retire_element(
        self,
        element_id: str,
        unit_id: str,
        actor_name: str,
        driver: str,
    ) -> None:
        current = self.get_element(element_id)
        if not current:
            raise ValueError("Element not found.")
        if current.get("unit_id") != unit_id:
            raise PermissionError("Only the owning organisational unit can retire this element.")
        if not driver.strip():
            raise ValueError("A rationale/driver is required for every change.")
        now = utc_now()
        after = {
            **current,
            "active": False,
            "model_status": "ARCHIVED",
            "updated_at": now,
            "last_change": now,
            "change_type": "DELETE",
            "driver": driver.strip(),
            "version": int(current.get("version", 1)) + 1,
        }
        change = {
            "id": new_id("change"),
            "unit_id": unit_id,
            "actor_name": actor_name,
            "operation": "DELETE",
            "entity_kind": "ELEMENT",
            "entity_id": element_id,
            "entity_code": current.get("code", ""),
            "entity_name": current.get("title", ""),
            "submodel": current.get("submodel", ""),
            "element_type": current.get("element_type", ""),
            "driver": driver.strip(),
            "before_json": json_text(current),
            "after_json": json_text(after),
            "source_change_id": "",
            "created_at": now,
        }
        self.run(
            """
            MATCH (e:Element {id: $element_id, unit_id: $unit_id})
            SET e += $after
            WITH e
            MATCH (u:OrgUnit {id: $unit_id})
            CREATE (c:Change)
            SET c = $change
            CREATE (c)-[:AFFECTS]->(e)
            CREATE (c)-[:MADE_BY]->(u)
            """,
            element_id=element_id,
            unit_id=unit_id,
            after=after,
            change=change,
        )
        self.run(
            """
            MATCH (e:Element {id: $element_id})-[r:MODEL_RELATION]-()
            SET r.active = false, r.updated_at = $now
            """,
            element_id=element_id,
            now=now,
        )
        self.create_change_notifications(change["id"], unit_id)

    # Model relationships

    def list_relationships(
        self,
        relation_owner_unit_id: str | None = None,
        element_unit_id: str | None = None,
        include_inactive: bool = False,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (source:Element)-[r:MODEL_RELATION]->(target:Element)
            WHERE ($relation_owner_unit_id IS NULL OR r.unit_id = $relation_owner_unit_id)
              AND ($element_unit_id IS NULL OR source.unit_id = $element_unit_id OR target.unit_id = $element_unit_id)
              AND ($include_inactive OR coalesce(r.active, true) = true)
            RETURN properties(r) AS relation,
                   source.id AS source_id,
                   target.id AS target_id,
                   source.unit_id AS source_unit_id,
                   target.unit_id AS target_unit_id
            ORDER BY r.created_at DESC, r.id
            LIMIT $limit
            """,
            relation_owner_unit_id=relation_owner_unit_id,
            element_unit_id=element_unit_id,
            include_inactive=include_inactive,
            limit=limit,
        )
        return [
            {
                **row["relation"],
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "source_unit_id": row["source_unit_id"],
                "target_unit_id": row["target_unit_id"],
            }
            for row in rows
        ]

    def get_relationship(self, relationship_id: str) -> dict[str, Any] | None:
        rows = self.run(
            """
            MATCH (source:Element)-[r:MODEL_RELATION {id: $relationship_id}]->(target:Element)
            RETURN properties(r) AS relation, source.id AS source_id, target.id AS target_id
            """,
            relationship_id=relationship_id,
        )
        if not rows:
            return None
        return {
            **rows[0]["relation"],
            "source_id": rows[0]["source_id"],
            "target_id": rows[0]["target_id"],
        }

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        kind: str,
        rationale: str,
        unit_id: str,
        actor_name: str,
        source_change_id: str = "",
        notation: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        source = self.get_element(source_id)
        target = self.get_element(target_id)
        if not source or not target:
            raise ValueError("Source or target element was not found.")
        if source.get("unit_id") != unit_id:
            raise PermissionError(
                "The source element must belong to the logged-in organisational unit."
            )
        allowed = allowed_relationship_kinds(source, target)
        if kind not in allowed:
            raise ValueError(
                f"{kind} is not permitted for this source/target 4EM combination."
            )
        if not rationale.strip():
            raise ValueError("A rationale is required for every relationship change.")

        now = utc_now()
        rel_id = new_id("relation")
        notation = notation or {}
        props = {
            "id": rel_id,
            "kind": kind,
            "rationale": rationale.strip(),
            "unit_id": unit_id,
            "active": True,
            "created_at": now,
            "updated_at": now,
        }
        for property_name in (
            "strength",
            "source_cardinality",
            "target_cardinality",
            "totality",
        ):
            value = str(notation.get(property_name, "")).strip()
            if value:
                props[property_name] = value
        change = {
            "id": new_id("change"),
            "unit_id": unit_id,
            "actor_name": actor_name,
            "operation": "CREATE",
            "entity_kind": "RELATIONSHIP",
            "entity_id": rel_id,
            "entity_code": kind,
            "entity_name": f"{source.get('code', '')} {kind} {target.get('code', '')}",
            "submodel": f"{source.get('submodel', '')} → {target.get('submodel', '')}",
            "element_type": "MODEL_RELATION",
            "driver": rationale.strip(),
            "before_json": "{}",
            "after_json": json_text(
                {**props, "source_id": source_id, "target_id": target_id}
            ),
            "source_change_id": source_change_id,
            "created_at": now,
        }
        rows = self.run(
            """
            MATCH (source:Element {id: $source_id})
            MATCH (target:Element {id: $target_id})
            MATCH (u:OrgUnit {id: $unit_id})
            CREATE (source)-[r:MODEL_RELATION]->(target)
            SET r = $props
            CREATE (c:Change)
            SET c = $change
            CREATE (c)-[:AFFECTS]->(source)
            CREATE (c)-[:MADE_BY]->(u)
            RETURN properties(r) AS relation
            """,
            source_id=source_id,
            target_id=target_id,
            unit_id=unit_id,
            props=props,
            change=change,
        )
        if not rows:
            raise RuntimeError("Relationship creation failed.")
        self.create_change_notifications(change["id"], unit_id)
        return {
            **rows[0]["relation"],
            "source_id": source_id,
            "target_id": target_id,
        }

    def retire_relationship(
        self,
        relationship_id: str,
        unit_id: str,
        actor_name: str,
        driver: str,
    ) -> None:
        current = self.get_relationship(relationship_id)
        if not current:
            raise ValueError("Relationship not found.")
        if current.get("unit_id") != unit_id:
            raise PermissionError("Only the owning unit can retire this relationship.")
        if not driver.strip():
            raise ValueError("A rationale is required for every relationship change.")
        now = utc_now()
        after = {**current, "active": False, "updated_at": now}
        change = {
            "id": new_id("change"),
            "unit_id": unit_id,
            "actor_name": actor_name,
            "operation": "DELETE",
            "entity_kind": "RELATIONSHIP",
            "entity_id": relationship_id,
            "entity_code": current.get("kind", ""),
            "entity_name": current.get("kind", ""),
            "submodel": "Inter-model relationship",
            "element_type": "MODEL_RELATION",
            "driver": driver.strip(),
            "before_json": json_text(current),
            "after_json": json_text(after),
            "source_change_id": "",
            "created_at": now,
        }
        self.run(
            """
            MATCH (source:Element)-[r:MODEL_RELATION {id: $relationship_id}]->(target:Element)
            WHERE r.unit_id = $unit_id
            SET r.active = false, r.updated_at = $now
            WITH source
            MATCH (u:OrgUnit {id: $unit_id})
            CREATE (c:Change)
            SET c = $change
            CREATE (c)-[:AFFECTS]->(source)
            CREATE (c)-[:MADE_BY]->(u)
            """,
            relationship_id=relationship_id,
            unit_id=unit_id,
            now=now,
            change=change,
        )
        self.create_change_notifications(change["id"], unit_id)

    # Change history

    def list_changes(
        self, unit_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (c:Change)-[:MADE_BY]->(u:OrgUnit)
            WHERE ($unit_id IS NULL OR c.unit_id = $unit_id)
            OPTIONAL MATCH (c)-[:AFFECTS]->(e:Element)
            RETURN properties(c) AS change,
                   u.name AS unit_name,
                   e.id AS affected_element_id,
                   e.code AS affected_element_code,
                   e.title AS affected_element_title
            ORDER BY c.created_at DESC
            LIMIT $limit
            """,
            unit_id=unit_id,
            limit=limit,
        )
        return [
            {
                **row["change"],
                "unit_name": row.get("unit_name"),
                "affected_element_id": row.get("affected_element_id"),
                "affected_element_code": row.get("affected_element_code"),
                "affected_element_title": row.get("affected_element_title"),
            }
            for row in rows
        ]

    def get_change(self, change_id: str) -> dict[str, Any] | None:
        rows = self.run(
            """
            MATCH (c:Change {id: $change_id})-[:MADE_BY]->(u:OrgUnit)
            OPTIONAL MATCH (c)-[:AFFECTS]->(e:Element)
            RETURN properties(c) AS change,
                   properties(u) AS unit,
                   properties(e) AS affected_element
            """,
            change_id=change_id,
        )
        if not rows:
            return None
        return {
            **rows[0]["change"],
            "unit": rows[0].get("unit") or {},
            "affected_element": rows[0].get("affected_element") or {},
        }

    # Propagation suggestions

    def save_suggestion(
        self,
        change_id: str,
        source_unit_id: str,
        target_unit_id: str,
        analysis: dict[str, Any],
        notification_id: str = "",
    ) -> dict[str, Any]:
        now = utc_now()
        props = {
            "id": new_id("suggestion"),
            "change_id": change_id,
            "source_unit_id": source_unit_id,
            "target_unit_id": target_unit_id,
            "notification_id": notification_id,
            "title": analysis.get("title", "Change adaptation review"),
            "summary": analysis.get("summary", analysis.get("target_obligation_summary", "")),
            "rationale": analysis.get("rationale", ""),
            "target_impact": analysis.get("target_impact", ""),
            "relevant": bool(analysis.get("relevant", False)),
            # Preserve the legacy property for existing data. The candidate-first
            # contract intentionally omits model-generated confidence scores.
            "confidence": 0.0,
            "candidate_target_element_id": analysis.get("candidate_target_element_id", ""),
            "candidate_fit": analysis.get("candidate_fit", "NONE"),
            "candidate_fit_explanation": analysis.get("candidate_fit_explanation", ""),
            "propagation_applicability": analysis.get("propagation_applicability", "NONE"),
            "target_obligation_summary": analysis.get("target_obligation_summary", ""),
            "target_evidence_ids_json": json_text(analysis.get("target_evidence_ids", [])),
            "projection_notes_json": json_text(analysis.get("projection_notes", [])),
            "proposed_actions_json": json_text(analysis.get("proposed_actions", [])),
            "analysis_json": json_text(analysis),
            "status": "PENDING" if analysis.get("relevant", False) else "NOT_RECOMMENDED",
            "resolution_note": "",
            "created_at": now,
            "updated_at": now,
        }
        rows = self.run(
            """
            MATCH (c:Change {id: $change_id})
            MATCH (source:OrgUnit {id: $source_unit_id})
            MATCH (target:OrgUnit {id: $target_unit_id})
            OPTIONAL MATCH (n:ChangeNotification {id: $notification_id, target_unit_id: $target_unit_id})
            CREATE (s:PropagationSuggestion)
            SET s = $props
            CREATE (s)-[:BASED_ON]->(c)
            CREATE (s)-[:FROM_UNIT]->(source)
            CREATE (s)-[:FOR_UNIT]->(target)
            FOREACH (_ IN CASE WHEN n IS NULL THEN [] ELSE [1] END |
                CREATE (s)-[:GENERATED_FROM]->(n)
                SET n.status = 'ANALYSED',
                    n.analysis_requested_at = $now,
                    n.updated_at = $now
            )
            RETURN properties(s) AS suggestion
            """,
            change_id=change_id,
            source_unit_id=source_unit_id,
            target_unit_id=target_unit_id,
            notification_id=notification_id,
            now=now,
            props=props,
        )
        if not rows:
            raise RuntimeError("The propagation suggestion could not be saved.")
        return rows[0]["suggestion"]

    def list_suggestions(
        self,
        target_unit_id: str | None = None,
        source_unit_id: str | None = None,
        statuses: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        statuses = statuses or []
        rows = self.run(
            """
            MATCH (s:PropagationSuggestion)-[:BASED_ON]->(c:Change)
            MATCH (s)-[:FROM_UNIT]->(source:OrgUnit)
            MATCH (s)-[:FOR_UNIT]->(target:OrgUnit)
            OPTIONAL MATCH (s)-[:GENERATED_FROM]->(n:ChangeNotification)
            WHERE ($target_unit_id IS NULL OR s.target_unit_id = $target_unit_id)
              AND ($source_unit_id IS NULL OR s.source_unit_id = $source_unit_id)
              AND (size($statuses) = 0 OR s.status IN $statuses)
            RETURN properties(s) AS suggestion,
                   source.name AS source_unit_name,
                   target.name AS target_unit_name,
                   c.entity_code AS source_entity_code,
                   c.entity_name AS source_entity_name,
                   c.operation AS source_operation,
                   c.driver AS source_driver,
                   c.created_at AS source_change_at,
                   n.id AS notification_id
            ORDER BY s.created_at DESC
            LIMIT $limit
            """,
            target_unit_id=target_unit_id,
            source_unit_id=source_unit_id,
            statuses=statuses,
            limit=limit,
        )
        return [
            {
                **row["suggestion"],
                "source_unit_name": row.get("source_unit_name"),
                "target_unit_name": row.get("target_unit_name"),
                "source_entity_code": row.get("source_entity_code"),
                "source_entity_name": row.get("source_entity_name"),
                "source_operation": row.get("source_operation"),
                "source_driver": row.get("source_driver"),
                "source_change_at": row.get("source_change_at"),
                "notification_id": row.get("notification_id"),
            }
            for row in rows
        ]

    def get_suggestion(self, suggestion_id: str) -> dict[str, Any] | None:
        rows = self.run(
            """
            MATCH (s:PropagationSuggestion {id: $suggestion_id})-[:BASED_ON]->(c:Change)
            MATCH (s)-[:FROM_UNIT]->(source:OrgUnit)
            MATCH (s)-[:FOR_UNIT]->(target:OrgUnit)
            RETURN properties(s) AS suggestion,
                   properties(c) AS change,
                   properties(source) AS source_unit,
                   properties(target) AS target_unit
            """,
            suggestion_id=suggestion_id,
        )
        if not rows:
            return None
        return {
            **rows[0]["suggestion"],
            "change": rows[0]["change"],
            "source_unit": rows[0]["source_unit"],
            "target_unit": rows[0]["target_unit"],
        }

    def resolve_suggestion(
        self, suggestion_id: str, target_unit_id: str, status: str, note: str
    ) -> None:
        if status not in {"ADOPTED", "MODIFIED", "REJECTED", "REVIEWED"}:
            raise ValueError("Invalid suggestion resolution status.")
        self.run(
            """
            MATCH (s:PropagationSuggestion {id: $suggestion_id, target_unit_id: $target_unit_id})
            SET s.status = $status,
                s.resolution_note = $note,
                s.updated_at = $now
            """,
            suggestion_id=suggestion_id,
            target_unit_id=target_unit_id,
            status=status,
            note=note.strip(),
            now=utc_now(),
        )

    # Metrics and export

    def metrics(self, unit_id: str) -> dict[str, int]:
        elements = self.run(
            "MATCH (e:Element {unit_id: $unit_id}) WHERE coalesce(e.active, true) RETURN count(e) AS n",
            unit_id=unit_id,
        )[0]["n"]
        relations = self.run(
            "MATCH ()-[r:MODEL_RELATION {unit_id: $unit_id}]->() WHERE coalesce(r.active, true) RETURN count(r) AS n",
            unit_id=unit_id,
        )[0]["n"]
        changes = self.run(
            "MATCH (c:Change {unit_id: $unit_id}) RETURN count(c) AS n",
            unit_id=unit_id,
        )[0]["n"]
        pending = self.run(
            "MATCH (n:ChangeNotification {target_unit_id: $unit_id}) WHERE n.status IN ['UNREAD', 'READ', 'ADAPTING'] RETURN count(n) AS n",
            unit_id=unit_id,
        )[0]["n"]
        return {
            "elements": int(elements),
            "relations": int(relations),
            "changes": int(changes),
            "pending": int(pending),
        }

    def export_unit(self, unit_id: str) -> dict[str, Any]:
        return {
            "exported_at": utc_now(),
            "unit": self.get_unit(unit_id),
            "elements": self.list_elements(unit_id=unit_id, include_inactive=True),
            "relationships": self.list_relationships(
                relation_owner_unit_id=unit_id, include_inactive=True
            ),
            "changes": self.list_changes(unit_id=unit_id, limit=5000),
            "change_notifications": self.list_notifications(
                target_unit_id=unit_id, limit=5000
            ),
            "incoming_suggestions": self.list_suggestions(
                target_unit_id=unit_id, limit=5000
            ),
            "outgoing_suggestions": self.list_suggestions(
                source_unit_id=unit_id, limit=5000
            ),
        }


# Adaptation analysis


class LLMService:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = (
            OpenAI(api_key=api_key, max_retries=0)
            if not is_placeholder(api_key)
            else None
        )

    @property
    def configured(self) -> bool:
        return self.client is not None

    def analyse_change_for_unit(
        self,
        source_change: dict[str, Any],
        source_unit: dict[str, Any],
        target_unit: dict[str, Any],
        target_elements: list[dict[str, Any]],
        target_relationships: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        source_payload = {
            **source_change,
            "before_state": parse_json_object(source_change.get("before_json")),
            "after_state": parse_json_object(source_change.get("after_json")),
            "affected_element": source_change.get("affected_element") or {},
        }
        candidate_shortlist = rank_target_candidates(source_payload, target_elements)
        selected_elements = prioritise_elements_for_context(
            target_elements, candidate_shortlist
        )
        selected_ids = {
            str(element.get("id"))
            for element in selected_elements
            if element.get("id")
        }
        compact_elements = [
            {
                "id": element.get("id"),
                "unit_id": element.get("unit_id"),
                "code": element.get("code"),
                "submodel": element.get("submodel"),
                "element_type": element.get("element_type"),
                "title": element.get("title"),
                "description": element.get("description"),
                "model_status": element.get("model_status"),
                "priority": element.get("priority"),
                "criticality": element.get("criticality"),
                "tags": element.get("tags"),
                "driver": element.get("driver"),
                "metadata": parse_json_object(element.get("metadata_json")),
            }
            for element in selected_elements
        ]

        # Expose only relationships whose endpoints occur in the bounded target
        # context, prioritising topology adjacent to shortlisted elements.
        visible_relationships = [
            relation
            for relation in target_relationships
            if str(relation.get("source_id", "")) in selected_ids
            and str(relation.get("target_id", "")) in selected_ids
        ]
        candidate_ids = {
            str(candidate.get("id", ""))
            for candidate in candidate_shortlist
            if candidate.get("id")
        }
        prioritised_relationships = sorted(
            visible_relationships,
            key=lambda relation: (
                0
                if (
                    str(relation.get("source_id", "")) in candidate_ids
                    or str(relation.get("target_id", "")) in candidate_ids
                )
                else 1,
                str(relation.get("source_id", "")),
                str(relation.get("target_id", "")),
            ),
        )[:MAX_LLM_RELATIONSHIPS]
        elements_by_id = {
            str(element.get("id")): element for element in selected_elements
        }
        compact_relationships = []
        for relation in prioritised_relationships:
            source = elements_by_id.get(str(relation.get("source_id", "")), {})
            target = elements_by_id.get(str(relation.get("target_id", "")), {})
            compact_relationships.append(
                {
                    "source_id": relation.get("source_id"),
                    "source_title": source.get("title"),
                    "source_submodel": source.get("submodel"),
                    "source_element_type": source.get("element_type"),
                    "source_unit_id": source.get("unit_id"),
                    "target_id": relation.get("target_id"),
                    "target_title": target.get("title"),
                    "target_submodel": target.get("submodel"),
                    "target_element_type": target.get("element_type"),
                    "target_unit_id": target.get("unit_id"),
                    "kind": relation.get("kind") or relation.get("relationship_kind"),
                    "rationale": relation.get("rationale"),
                }
            )

        context = {
            "method": {
                "purpose": "Candidate-first, action-consistent model-change propagation",
                "unit_of_analysis": "one source change and one target unit",
                "organisational_relation": organisational_relation(
                    source_unit, target_unit
                ),
                "action_contract": {
                    "NO_CHANGE": "exactly one NO_CHANGE action",
                    "CHANGE": "smallest concrete action set; prefer one UPDATE_ELEMENT",
                },
                "anti_bias_rule": (
                    "Compare the best existing candidate with the null case; "
                    "neither change nor no-change is the default."
                ),
            },
            "source_unit": compact_unit_for_llm(source_unit),
            "source_change": source_payload,
            "target_unit": compact_unit_for_llm(target_unit),
            "candidate_shortlist": candidate_shortlist,
            "target_4em_elements": compact_elements,
            "target_4em_relationships": compact_relationships,
            "allowed_element_types": ELEMENT_TYPES,
            "allowed_relationships": ALL_RELATIONSHIP_KINDS,
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=PROPAGATION_INSTRUCTIONS,
            input=json_text(context),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "candidate_first_model_change_propagation",
                    "schema": PROPAGATION_OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
            reasoning={"effort": REASONING_EFFORT},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
        status = str(getattr(response, "status", "") or "")
        output_text = str(getattr(response, "output_text", "") or "")
        if status == "incomplete":
            raise RuntimeError(
                "OpenAI returned an incomplete response: "
                + json_text(plain(getattr(response, "incomplete_details", {})))
            )
        if not output_text.strip():
            raise RuntimeError(
                f"OpenAI returned no structured output; status={status!r}"
            )

        raw_analysis = json.loads(output_text)
        analysis = validate_propagation_analysis(
            raw_analysis,
            str(target_unit.get("id", "")),
            target_elements,
        )
        analysis["response_metadata"] = {
            "response_id": getattr(response, "id", ""),
            "response_model": getattr(response, "model", self.model),
            "usage": plain(getattr(response, "usage", {})),
            "status": status,
            "candidate_shortlist": candidate_shortlist,
            "context_element_count": len(compact_elements),
            "context_relationship_count": len(compact_relationships),
        }
        return analysis


# Graph visualisation


def _relationship_display_label(relation: dict[str, Any]) -> str:
    kind = str(relation.get("kind", "RELATES_TO"))
    labels = {
        "CONFLICTS": "contradicts",
        "CAUSES": "causes",
        "PART_OF": "PartOF",
        "ISA": "ISA",
        "HAS_ATTRIBUTE": "has attribute",
        "CONTROL_FLOW": "",
        "DECOMPOSES_TO": "decomposes to",
    }
    label = labels.get(kind, kind.replace("_", " ").lower())
    strength = str(relation.get("strength", "")).strip().lower()
    if strength and kind in {"SUPPORTS", "HINDERS", "CONFLICTS"}:
        label = f"{label} ({strength})"
    source_cardinality = str(relation.get("source_cardinality", "")).strip()
    target_cardinality = str(relation.get("target_cardinality", "")).strip()
    if source_cardinality or target_cardinality:
        label = f"{label}\n{source_cardinality or '?'} — {target_cardinality or '?'}"
    return label


def build_model_dot(
    elements: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> str:
    """Render a Graphviz fallback with the closest available 4EM-style symbols."""
    elements = elements[:MAX_GRAPH_ELEMENTS]
    element_ids = {e.get("id") for e in elements}
    relationships = [
        r
        for r in relationships
        if r.get("source_id") in element_ids and r.get("target_id") in element_ids
    ]
    units_by_id = {u.get("id"): u for u in units}
    elements_by_id = {e.get("id"): e for e in elements}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for element in elements:
        grouped.setdefault(element.get("unit_id", "unknown"), []).append(element)

    type_style: dict[str, tuple[str, str, str]] = {
        "Goal": ("box", BOOK_4EM_COLOURS["goal"], BOOK_4EM_COLOURS["line"]),
        "Problem": ("box", BOOK_4EM_COLOURS["problem"], BOOK_4EM_COLOURS["line"]),
        "Weakness": ("box", BOOK_4EM_COLOURS["problem"], BOOK_4EM_COLOURS["line"]),
        "Threat": ("box", BOOK_4EM_COLOURS["problem"], BOOK_4EM_COLOURS["line"]),
        "Cause": ("box", BOOK_4EM_COLOURS["cause"], BOOK_4EM_COLOURS["line"]),
        "Constraint": ("box", "#FFFFFF", BOOK_4EM_COLOURS["constraint_border"]),
        "Opportunity": ("box", BOOK_4EM_COLOURS["opportunity"], BOOK_4EM_COLOURS["line"]),
        "Concept": ("oval", BOOK_4EM_COLOURS["concept"], BOOK_4EM_COLOURS["concept_border"]),
        "Attribute": ("oval", "#FFFFFF", BOOK_4EM_COLOURS["concept_border"]),
        "Process": ("Mrecord", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "ExternalProcess": ("box3d", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "InformationSet": ("parallelogram", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "MaterialSet": ("parallelogram", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "Individual": ("box", BOOK_4EM_COLOURS["individual"], BOOK_4EM_COLOURS["line"]),
        "Role": ("box", BOOK_4EM_COLOURS["role"], "#5F4A78"),
        "NonHumanResource": ("box", BOOK_4EM_COLOURS["resource"], BOOK_4EM_COLOURS["line"]),
        "OrganisationalUnit": ("box", "#FFFFFF", BOOK_4EM_COLOURS["org_unit_border"]),
        "TechnicalComponent": ("ellipse", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "ISRequirement": ("ellipse", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "FunctionalRequirement": ("ellipse", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
        "NonFunctionalRequirement": ("ellipse", "#FFFFFF", BOOK_4EM_COLOURS["line"]),
    }
    lines = [
        "digraph G {",
        'graph [rankdir="LR", bgcolor="transparent", pad="0.25", nodesep="0.45", ranksep="0.8", fontname="Arial"];',
        'node [fontname="Arial", fontsize="10", penwidth="1.2", margin="0.12,0.08"];',
        'edge [fontname="Arial", fontsize="8", color="#5F6368", arrowsize="0.7"];',
    ]
    for unit_id, unit_elements in grouped.items():
        unit = units_by_id.get(unit_id, {"name": unit_id})
        cluster_id = safe_dot_id(str(unit_id))
        lines.append(f"subgraph cluster_{cluster_id} {{")
        lines.append(
            f'label="{dot_escape(unit.get("name", unit_id))}"; color="#D0D5DD"; style="rounded"; penwidth="1";'
        )
        for element in unit_elements:
            shape, fill, border = type_style.get(
                str(element.get("element_type", "")), ("box", "#FFFFFF", "#5F6368")
            )
            label = f"{element.get('code', '')}\n{element.get('title', '')}"
            font_color = "#FFFFFF" if element.get("element_type") in {"Opportunity", "Role"} else "#202124"
            style = "filled,dashed" if element.get("element_type") in {
                "ISRequirement", "FunctionalRequirement", "NonFunctionalRequirement"
            } else "filled"
            lines.append(
                f'"{dot_escape(element.get("id"))}" '
                f'[label="{dot_escape(label)}", shape="{shape}", style="{style}", '
                f'fillcolor="{fill}", color="{border}", fontcolor="{font_color}"];'
            )
        lines.append("}")
    for relation in relationships:
        source = elements_by_id.get(relation.get("source_id"), {})
        target = elements_by_id.get(relation.get("target_id"), {})
        cross_model = source.get("submodel") != target.get("submodel")
        edge_style = "dashed" if cross_model else "solid"
        label = _relationship_display_label(relation)
        lines.append(
            f'"{dot_escape(relation.get("source_id"))}" -> "{dot_escape(relation.get("target_id"))}" '
            f'[label="{dot_escape(label)}", style="{edge_style}", tooltip="{dot_escape(relation.get("rationale", ""))}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def render_interactive_model_graph(
    elements: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    units: list[dict[str, Any]],
    graph_key: str,
    height: int = 680,
) -> None:
    """Render an interactive diagram using the Chapter 8 4EM notation.

    SVG nodes and connector markers preserve notation that vis-network primitives
    cannot express. Layout state remains browser-local and never mutates the
    semantic graph.
    """
    visible_elements = elements[:MAX_GRAPH_ELEMENTS]
    visible_ids = {str(e.get("id", "")) for e in visible_elements}
    visible_relationships = [
        r
        for r in relationships
        if str(r.get("source_id", "")) in visible_ids
        and str(r.get("target_id", "")) in visible_ids
    ]
    units_by_id = {str(u.get("id", "")): u for u in units}
    elements_by_id = {str(e.get("id", "")): e for e in visible_elements}

    decomposed_ids: set[str] = set()
    for relation in visible_relationships:
        kind = str(relation.get("kind", ""))
        if kind in {"AND_REFINES", "OR_REFINES", "AND_OR_REFINES", "ISA", "PART_OF"}:
            decomposed_ids.add(str(relation.get("target_id", "")))
        elif kind == "DECOMPOSES_TO":
            decomposed_ids.add(str(relation.get("source_id", "")))

    nodes: list[dict[str, Any]] = []
    for element in visible_elements:
        element_id = str(element.get("id", ""))
        unit = units_by_id.get(str(element.get("unit_id", "")), {})
        submodel = str(element.get("submodel", ""))
        title_html = (
            f"<b>{html_lib.escape(str(element.get('code', '')))}</b><br>"
            f"{html_lib.escape(str(element.get('element_type', '')))}<br>"
            f"{html_lib.escape(str(element.get('title', '')))}<br><br>"
            f"<b>Unit:</b> {html_lib.escape(str(unit.get('name', element.get('unit_id', ''))))}<br>"
            f"<b>4EM sub-model:</b> {html_lib.escape(submodel)}<br>"
            f"<b>Status:</b> {html_lib.escape(str(element.get('model_status', '')))}<br>"
            f"<b>Description:</b> {html_lib.escape(str(element.get('description', '')))}"
        )
        nodes.append(
            {
                "id": element_id,
                "label": "",
                "title": title_html,
                "shape": "image",
                "image": _notation_node_svg(element, element_id in decomposed_ids),
                "size": 60,
                "borderWidth": 0,
                "shapeProperties": {
                    "useBorderWithImage": False,
                    "interpolation": True,
                },
                "mass": 1.4,
            }
        )

    edges: list[dict[str, Any]] = []
    processed_relation_ids: set[str] = set()

    def relation_id(relation: dict[str, Any], index: int = 0) -> str:
        return str(relation.get("id") or f"edge_{index}_{relation.get('source_id')}_{relation.get('target_id')}")

    def add_connector(
        connector_id: str,
        shape: str,
        title: str,
        filled: bool = False,
        size: int = 16,
    ) -> None:
        nodes.append(
            {
                "id": connector_id,
                "label": "",
                "title": html_lib.escape(title),
                "shape": shape,
                "size": size,
                "borderWidth": 1.5,
                "color": {
                    "background": "#111111" if filled else "#FFFFFF",
                    "border": BOOK_4EM_COLOURS["line"],
                    "highlight": {
                        "background": "#111111" if filled else "#FFFFFF",
                        "border": "#175CD3",
                    },
                },
                "font": {"size": 1},
                "mass": 0.45,
            }
        )

    def add_edge(
        edge_id: str,
        source_id: str,
        target_id: str,
        label: str = "",
        title: str = "",
        dashed: bool = False,
        arrow: bool = True,
        width: float = 1.25,
    ) -> None:
        edge: dict[str, Any] = {
            "id": edge_id,
            "from": source_id,
            "to": target_id,
            "label": label,
            "title": html_lib.escape(title),
            "color": {"color": BOOK_4EM_COLOURS["line"], "highlight": "#175CD3"},
            "font": {
                "face": "Arial",
                "size": 10,
                "align": "middle",
                "background": "rgba(255,255,255,0.86)",
                "strokeWidth": 0,
            },
            "smooth": {"enabled": True, "type": "dynamic"},
            "width": width,
            "dashes": [8, 6] if dashed else False,
        }
        if arrow:
            edge["arrows"] = {"to": {"enabled": True, "scaleFactor": 0.65}}
        edges.append(edge)

    # Refinement markers shared by Goals, Business Rules, and TCRM.
    refinement_shapes = {
        "AND_REFINES": ("triangle", "AND refinement"),
        "OR_REFINES": ("triangleDown", "OR refinement"),
        "AND_OR_REFINES": ("diamond", "AND/OR refinement"),
    }
    for kind, (shape, title) in refinement_shapes.items():
        buckets: dict[str, list[dict[str, Any]]] = {}
        for index, relation in enumerate(visible_relationships):
            if relation.get("kind") == kind:
                buckets.setdefault(str(relation.get("target_id", "")), []).append(relation)
        for target_id, bucket in buckets.items():
            connector_id = f"connector:{kind}:{target_id}"
            add_connector(connector_id, shape, title)
            for index, relation in enumerate(bucket):
                rid = relation_id(relation, index)
                processed_relation_ids.add(rid)
                add_edge(
                    f"{rid}:child",
                    str(relation.get("source_id", "")),
                    connector_id,
                    title=str(relation.get("rationale", "")),
                )
            add_edge(f"{connector_id}:parent", connector_id, target_id)

    # Circles represent ISA and squares represent PART_OF; fill distinguishes
    # total from partial decomposition.
    for kind, shape in (("ISA", "dot"), ("PART_OF", "square")):
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for relation in visible_relationships:
            if relation.get("kind") == kind:
                totality = str(relation.get("totality", "PARTIAL")).upper()
                buckets.setdefault((str(relation.get("target_id", "")), totality), []).append(relation)
        for (target_id, totality), bucket in buckets.items():
            connector_id = f"connector:{kind}:{totality}:{target_id}"
            add_connector(
                connector_id,
                shape,
                f"{totality.title()} {kind if kind == 'ISA' else 'PartOF'} decomposition",
                filled=totality == "TOTAL",
                size=15,
            )
            for index, relation in enumerate(bucket):
                rid = relation_id(relation, index)
                processed_relation_ids.add(rid)
                add_edge(
                    f"{rid}:child",
                    str(relation.get("source_id", "")),
                    connector_id,
                    title=str(relation.get("rationale", "")),
                )
            add_edge(f"{connector_id}:parent", connector_id, target_id)

    # Business-process control-flow split and join markers.
    for kind, group_on, shape, title in (
        ("AND_SPLIT", "source", "dot", "AND split"),
        ("OR_SPLIT", "source", "diamond", "OR split"),
        ("AND_JOIN", "target", "dot", "AND join"),
        ("OR_JOIN", "target", "diamond", "OR join"),
    ):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for relation in visible_relationships:
            if relation.get("kind") == kind:
                key = str(relation.get("source_id" if group_on == "source" else "target_id", ""))
                buckets.setdefault(key, []).append(relation)
        for anchor_id, bucket in buckets.items():
            connector_id = f"connector:{kind}:{anchor_id}"
            add_connector(connector_id, shape, title, size=15)
            if group_on == "source":
                add_edge(f"{connector_id}:in", anchor_id, connector_id)
                for index, relation in enumerate(bucket):
                    rid = relation_id(relation, index)
                    processed_relation_ids.add(rid)
                    add_edge(
                        f"{rid}:out",
                        connector_id,
                        str(relation.get("target_id", "")),
                        title=str(relation.get("rationale", "")),
                    )
            else:
                for index, relation in enumerate(bucket):
                    rid = relation_id(relation, index)
                    processed_relation_ids.add(rid)
                    add_edge(
                        f"{rid}:in",
                        str(relation.get("source_id", "")),
                        connector_id,
                        title=str(relation.get("rationale", "")),
                    )
                add_edge(f"{connector_id}:out", connector_id, anchor_id)

    for index, relation in enumerate(visible_relationships):
        rid = relation_id(relation, index)
        if rid in processed_relation_ids:
            continue
        source_id = str(relation.get("source_id", ""))
        target_id = str(relation.get("target_id", ""))
        source = elements_by_id.get(source_id, {})
        target = elements_by_id.get(target_id, {})
        kind = str(relation.get("kind", "RELATES_TO"))
        cross_model = source.get("submodel") != target.get("submodel")
        dashed = cross_model or kind == "DECOMPOSES_TO"
        arrow = kind not in {"HAS_ATTRIBUTE"}
        add_edge(
            rid,
            source_id,
            target_id,
            label=_relationship_display_label(relation),
            title=str(relation.get("rationale", "")),
            dashed=dashed,
            arrow=arrow,
        )

    legend_specs = [
        ("Goal", "Goal"),
        ("Problem", "Problem / weakness / threat"),
        ("Cause", "Cause"),
        ("Constraint", "Constraint"),
        ("Opportunity", "Opportunity"),
        ("BusinessRule", "Business rule"),
        ("Concept", "Concept"),
        ("Attribute", "Attribute"),
        ("Process", "Process"),
        ("ExternalProcess", "External process"),
        ("InformationSet", "Information / material set"),
        ("Individual", "Individual"),
        ("Role", "Role"),
        ("NonHumanResource", "Resource"),
        ("OrganisationalUnit", "Organisational unit"),
        ("TechnicalComponent", "Technical component"),
        ("ISRequirement", "Requirement"),
    ]
    legend_items: list[str] = []
    for element_type, label in legend_specs:
        sample = {
            "element_type": element_type,
            "code": TYPE_PREFIX.get(element_type, element_type),
            "title": "Component text",
        }
        legend_items.append(
            '<div class="legend-item">'
            f'<img src="{_notation_node_svg(sample)}" alt="{html_lib.escape(label)} symbol">'
            f'<span>{html_lib.escape(label)}</span>'
            '</div>'
        )
    legend_html = "".join(legend_items)

    safe_nodes = json.dumps(nodes, ensure_ascii=False).replace("</", "<\\/")
    safe_edges = json.dumps(edges, ensure_ascii=False).replace("</", "<\\/")
    storage_key = json.dumps(f"fractal-4em-layout:{graph_key}")
    component_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.9/styles/vis-network.min.css" />
  <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; background: transparent; }}
    #toolbar {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:0 0 8px 0; }}
    #toolbar button {{ border:1px solid #D0D5DD; border-radius:8px; background:#fff; color:#344054;
      padding:7px 11px; cursor:pointer; font-size:12px; }}
    #toolbar button:hover {{ background:#F9FAFB; }}
    #hint {{ color:#667085; font-size:12px; margin-left:auto; }}
    #notation-legend {{ display:none; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px;
      border:1px solid #EAECF0; border-radius:10px; padding:10px; margin:0 0 8px 0; background:#FFFFFF; }}
    .legend-item {{ display:flex; align-items:center; gap:8px; min-height:48px; color:#344054; font-size:11px; }}
    .legend-item img {{ width:88px; height:38px; object-fit:contain; flex:0 0 auto; }}
    #network {{ width:100%; height:{height - 50}px; border:1px solid #EAECF0; border-radius:12px; background:#FCFCFD; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="fit-button" type="button">Fit model</button>
    <button id="physics-button" type="button">Toggle auto-layout</button>
    <button id="save-button" type="button">Save positions</button>
    <button id="reset-button" type="button">Reset saved layout</button>
    <button id="legend-button" type="button">4EM notation legend</button>
    <span id="hint">Drag components to arrange the model. Dashed arrows are inter-model links.</span>
  </div>
  <div id="notation-legend">{legend_html}</div>
  <div id="network"></div>
  <script>
    const storageKey = {storage_key};
    const nodes = new vis.DataSet({safe_nodes});
    const edges = new vis.DataSet({safe_edges});
    const container = document.getElementById('network');
    const options = {{
      interaction: {{ hover:true, navigationButtons:true, keyboard:true, multiselect:true }},
      manipulation: {{ enabled:false }},
      physics: {{
        enabled:true,
        stabilization: {{ iterations:220, updateInterval:25 }},
        solver:'forceAtlas2Based',
        forceAtlas2Based: {{ gravitationalConstant:-70, centralGravity:0.012, springLength:235, springConstant:0.045 }}
      }},
      layout: {{ improvedLayout:true, randomSeed:7 }},
      edges: {{ selectionWidth:2.2 }},
      nodes: {{ chosen:true }}
    }};
    const network = new vis.Network(container, {{nodes, edges}}, options);
    let physicsEnabled = true;

    function readSavedPositions() {{
      try {{ return JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }}
      catch (_) {{ return {{}}; }}
    }}
    function restorePositions() {{
      const saved = readSavedPositions();
      let restored = 0;
      Object.entries(saved).forEach(([id, pos]) => {{
        if (nodes.get(id) && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {{
          network.moveNode(id, pos.x, pos.y);
          restored += 1;
        }}
      }});
      if (restored > 0) {{ network.setOptions({{physics:false}}); physicsEnabled = false; }}
      return restored;
    }}
    function savePositions() {{
      localStorage.setItem(storageKey, JSON.stringify(network.getPositions()));
      document.getElementById('hint').textContent = 'Layout saved in this browser.';
    }}
    function clearPositions() {{
      localStorage.removeItem(storageKey);
      network.setOptions({{physics:true}});
      physicsEnabled = true;
      network.stabilize(220);
      document.getElementById('hint').textContent = 'Saved layout cleared; auto-layout restarted.';
    }}
    function fitGraph() {{ network.fit({{animation:{{duration:350, easingFunction:'easeInOutQuad'}}}}); }}
    function togglePhysics() {{
      physicsEnabled = !physicsEnabled;
      network.setOptions({{physics:physicsEnabled}});
      if (physicsEnabled) network.stabilize(140);
    }}
    function toggleLegend() {{
      const legend = document.getElementById('notation-legend');
      legend.style.display = legend.style.display === 'grid' ? 'none' : 'grid';
    }}
    document.getElementById('fit-button').addEventListener('click', fitGraph);
    document.getElementById('physics-button').addEventListener('click', togglePhysics);
    document.getElementById('save-button').addEventListener('click', savePositions);
    document.getElementById('reset-button').addEventListener('click', clearPositions);
    document.getElementById('legend-button').addEventListener('click', toggleLegend);
    network.once('stabilizationIterationsDone', () => {{
      network.setOptions({{physics:false}});
      physicsEnabled = false;
      restorePositions();
      fitGraph();
    }});
    network.on('dragEnd', savePositions);
    network.on('zoom', () => {{
      document.getElementById('hint').textContent = 'Drag components; use Fit model to centre the view.';
    }});
    if (restorePositions() > 0) {{ setTimeout(fitGraph, 100); }}
  </script>
</body>
</html>
"""
    components.html(component_html, height=height, scrolling=False)


def build_hierarchy_dot(units: list[dict[str, Any]]) -> str:
    lines = [
        "digraph Units {",
        'graph [rankdir="TB", bgcolor="transparent", pad="0.25", nodesep="0.4", ranksep="0.6"];',
        'node [shape="box", style="rounded,filled", fillcolor="#EEF4FF", color="#84ADFF", fontname="Arial", fontsize="10"];',
        'edge [color="#98A2B3", arrowsize="0.7"];',
    ]
    for unit in units:
        label = f"{unit.get('name', '')}\n{unit.get('level', '')}"
        lines.append(
            f'"{dot_escape(unit.get("id"))}" [label="{dot_escape(label)}"];'
        )
    for unit in units:
        parent_id = unit.get("parent_id")
        if parent_id:
            lines.append(
                f'"{dot_escape(parent_id)}" -> "{dot_escape(unit.get("id"))}" [label="contains"];'
            )
    lines.append("}")
    return "\n".join(lines)


# Suggestion execution


def apply_suggestion_actions(
    repo: GraphRepository,
    suggestion: dict[str, Any],
    target_unit_id: str,
    actor_name: str,
    actions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Apply reviewed adaptation actions sequentially.

    Revalidate at the persistence boundary so neither generated nor manually
    edited actions can bypass ownership and relationship invariants.
    """
    applied: list[str] = []
    errors: list[str] = []
    source_change_id = suggestion.get("change_id", "")

    for index, action in enumerate(actions, start=1):
        kind = action.get("action", "NO_CHANGE")
        try:
            if kind == "NO_CHANGE":
                applied.append(f"Action {index}: no model change required.")
                continue

            if kind == "CREATE_ELEMENT":
                submodel = action.get("submodel", "")
                element_type = action.get("element_type", "")
                if element_type not in ELEMENT_TYPES.get(submodel, []):
                    raise ValueError(
                        f"{element_type} is not valid in the {submodel} sub-model."
                    )
                tags = sorted(
                    set(clean_tags(action.get("tags", []))) | {"propagated-change"}
                )
                created = repo.create_element(
                    unit_id=target_unit_id,
                    actor_name=actor_name,
                    submodel=submodel,
                    element_type=element_type,
                    title=action.get("title", "").strip(),
                    description=action.get("description", "").strip(),
                    model_status=action.get("model_status", "PROPOSED")
                    if action.get("model_status", "PROPOSED") in MODEL_STATUSES
                    else "PROPOSED",
                    priority=action.get("priority", "")
                    if action.get("priority", "") in PRIORITIES
                    else "",
                    criticality=action.get("criticality", "")
                    if action.get("criticality", "") in CRITICALITIES
                    else "",
                    tags=tags,
                    driver=action.get("driver", "").strip()
                    or f"Adopted from propagation suggestion {suggestion.get('id', '')}",
                    source_change_id=source_change_id,
                )
                applied.append(f"Action {index}: created {created.get('code')}.")
                continue

            if kind == "UPDATE_ELEMENT":
                element_id = action.get("existing_element_id", "")
                current = repo.get_element(element_id)
                if not current or current.get("unit_id") != target_unit_id:
                    raise ValueError(
                        "The proposed update does not reference an element owned by this unit."
                    )
                proposed_tags = clean_tags(action.get("tags", current.get("tags", [])))
                fields = {
                    "title": action.get("title", "").strip() or current.get("title", ""),
                    "description": action.get("description", "").strip()
                    or current.get("description", ""),
                    "model_status": action.get("model_status", current.get("model_status", "PROPOSED"))
                    if action.get("model_status", current.get("model_status", "PROPOSED")) in MODEL_STATUSES
                    else current.get("model_status", "PROPOSED"),
                    "priority": action.get("priority", current.get("priority", ""))
                    if action.get("priority", current.get("priority", "")) in PRIORITIES
                    else current.get("priority", ""),
                    "criticality": action.get("criticality", current.get("criticality", ""))
                    if action.get("criticality", current.get("criticality", "")) in CRITICALITIES
                    else current.get("criticality", ""),
                    "tags": sorted(set(proposed_tags) | {"propagated-change"}),
                }
                updated = repo.update_element(
                    element_id=element_id,
                    unit_id=target_unit_id,
                    actor_name=actor_name,
                    fields=fields,
                    driver=action.get("driver", "").strip()
                    or f"Adapted from propagation suggestion {suggestion.get('id', '')}",
                    expected_version=int(current.get("version", 1)),
                    source_change_id=source_change_id,
                )
                applied.append(f"Action {index}: updated {updated.get('code')}.")
                continue

            if kind == "CREATE_RELATION":
                source_id = action.get("relation_source_id", "")
                target_id = action.get("relation_target_id", "")
                source = repo.get_element(source_id)
                target = repo.get_element(target_id)
                if not source or source.get("unit_id") != target_unit_id:
                    raise ValueError(
                        "The proposed relation source must be an element owned by this unit."
                    )
                if not target or target.get("unit_id") != target_unit_id:
                    raise ValueError(
                        "The proposed relation target must belong to this target unit."
                    )
                relationship_kind = action.get("relationship_kind", "RELATES_TO")
                if relationship_kind not in allowed_relationship_kinds(source, target):
                    raise ValueError(
                        f"{relationship_kind} is not valid between the selected 4EM elements."
                    )
                relation = repo.create_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    kind=relationship_kind,
                    rationale=action.get("driver", "").strip()
                    or f"Adopted from propagation suggestion {suggestion.get('id', '')}",
                    unit_id=target_unit_id,
                    actor_name=actor_name,
                    source_change_id=source_change_id,
                )
                applied.append(f"Action {index}: created relation {relation.get('kind')}.")
                continue

            raise ValueError(f"Unsupported action: {kind}")
        except Exception as exc:  # Keep independent actions reviewable.
            errors.append(f"Action {index} ({kind}): {exc}")

    return applied, errors


# Streamlit presentation


@st.cache_resource(show_spinner=False)
def cached_driver(uri: str, username: str, password: str) -> Any:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    return driver


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
        .app-kicker {font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #475467;}
        .method-card {border: 1px solid #E4E7EC; border-radius: 12px; padding: 14px; background: #FCFCFD; min-height: 126px;}
        .method-card strong {display: block; margin-bottom: 6px;}
        .small-muted {font-size: .82rem; color: #667085;}
        .status-pill {display:inline-block; border-radius:999px; padding:2px 9px; background:#EEF4FF; color:#3538CD; font-size:.76rem; font-weight:700;}
        div[data-testid="stMetric"] {border: 1px solid #EAECF0; padding: 12px; border-radius: 12px; background: #FFFFFF;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_error(exc: Exception) -> None:
    st.error(str(exc))


def safe_index(options: list[Any], value: Any, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def normalise_adaptation_action(action: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = action or {}
    submodel = raw.get("submodel") if raw.get("submodel") in SUBMODELS else "Goals"
    valid_types = ELEMENT_TYPES[submodel]
    element_type = (
        raw.get("element_type")
        if raw.get("element_type") in valid_types
        else valid_types[0]
    )
    tags = clean_tags(raw.get("tags", []))
    return {
        "_editor_id": raw.get("_editor_id") or new_id("action-editor"),
        "action": raw.get("action")
        if raw.get("action") in {"CREATE_ELEMENT", "UPDATE_ELEMENT", "CREATE_RELATION", "NO_CHANGE"}
        else "CREATE_ELEMENT",
        "existing_element_id": str(raw.get("existing_element_id", "")),
        "submodel": submodel,
        "element_type": element_type,
        "title": str(raw.get("title", "")),
        "description": str(raw.get("description", "")),
        "model_status": raw.get("model_status")
        if raw.get("model_status") in MODEL_STATUSES
        else "PROPOSED",
        "priority": raw.get("priority") if raw.get("priority") in PRIORITIES else "",
        "criticality": raw.get("criticality")
        if raw.get("criticality") in CRITICALITIES
        else "",
        "tags": tags,
        "relationship_kind": str(raw.get("relationship_kind", "RELATES_TO")),
        "relation_source_id": str(raw.get("relation_source_id", "")),
        "relation_target_id": str(raw.get("relation_target_id", "")),
        "driver": str(raw.get("driver", "")),
    }


def public_action(action: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in action.items() if not k.startswith("_")}


def render_structured_action_editor(
    suggestion_id: str,
    original_actions: list[dict[str, Any]],
    local_elements: list[dict[str, Any]],
    all_elements: list[dict[str, Any]],
    unit_names: dict[str, str],
) -> list[dict[str, Any]]:
    """Render the action-review editor and return the user-approved action set."""
    state_key = f"structured_actions_{suggestion_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [
            normalise_adaptation_action(action) for action in original_actions
        ] or [normalise_adaptation_action({"action": "NO_CHANGE"})]

    toolbar = st.columns([1, 1, 3])
    if toolbar[0].button(
        "Add manual action", key=f"add_manual_action_{suggestion_id}"
    ):
        st.session_state[state_key].append(
            normalise_adaptation_action({"action": "CREATE_ELEMENT"})
        )
        st.rerun()
    if toolbar[1].button(
        "Reset to LLM proposal", key=f"reset_actions_{suggestion_id}"
    ):
        st.session_state[state_key] = [
            normalise_adaptation_action(action) for action in original_actions
        ] or [normalise_adaptation_action({"action": "NO_CHANGE"})]
        st.rerun()
    toolbar[2].caption(
        "Every element and relationship can be changed before anything is written to Neo4j."
    )

    working_actions = list(st.session_state[state_key])
    edited_actions: list[dict[str, Any]] = []
    action_labels = {
        "CREATE_ELEMENT": "Add a new 4EM element",
        "UPDATE_ELEMENT": "Change an existing 4EM element",
        "CREATE_RELATION": "Add a 4EM relationship",
        "NO_CHANGE": "Record no local model change",
    }
    action_options = list(action_labels)

    for position, base_action in enumerate(working_actions, start=1):
        action = normalise_adaptation_action(base_action)
        editor_id = action["_editor_id"]
        with st.container(border=True):
            heading, remove_col = st.columns([6, 1])
            heading.markdown(f"**Reviewed action {position}**")
            if remove_col.button(
                "Remove", key=f"remove_action_{suggestion_id}_{editor_id}"
            ):
                st.session_state[state_key] = [
                    item
                    for item in working_actions
                    if item.get("_editor_id") != editor_id
                ]
                if not st.session_state[state_key]:
                    st.session_state[state_key] = [
                        normalise_adaptation_action({"action": "NO_CHANGE"})
                    ]
                st.rerun()

            kind = st.selectbox(
                "Action",
                options=action_options,
                index=safe_index(action_options, action.get("action")),
                format_func=lambda value: action_labels[value],
                key=f"action_kind_{suggestion_id}_{editor_id}",
            )
            edited = normalise_adaptation_action({**action, "action": kind})

            if kind == "CREATE_ELEMENT":
                submodel = st.selectbox(
                    "4EM sub-model",
                    options=list(SUBMODELS),
                    index=safe_index(list(SUBMODELS), action.get("submodel")),
                    format_func=lambda value: SUBMODELS[value],
                    key=f"action_submodel_{suggestion_id}_{editor_id}",
                )
                type_options = ELEMENT_TYPES[submodel]
                element_type = st.selectbox(
                    "Element type",
                    options=type_options,
                    index=safe_index(type_options, action.get("element_type")),
                    key=f"action_type_{suggestion_id}_{editor_id}_{submodel}",
                )
                title = st.text_input(
                    "Title",
                    value=action.get("title", ""),
                    key=f"action_title_{suggestion_id}_{editor_id}",
                )
                description = st.text_area(
                    "Description",
                    value=action.get("description", ""),
                    key=f"action_description_{suggestion_id}_{editor_id}",
                )
                c1, c2, c3 = st.columns(3)
                model_status = c1.selectbox(
                    "Model state",
                    MODEL_STATUSES,
                    index=safe_index(MODEL_STATUSES, action.get("model_status"), 2),
                    key=f"action_status_{suggestion_id}_{editor_id}",
                )
                priority = c2.selectbox(
                    "Priority",
                    PRIORITIES,
                    index=safe_index(PRIORITIES, action.get("priority")),
                    key=f"action_priority_{suggestion_id}_{editor_id}",
                )
                criticality = c3.selectbox(
                    "Criticality",
                    CRITICALITIES,
                    index=safe_index(CRITICALITIES, action.get("criticality")),
                    key=f"action_criticality_{suggestion_id}_{editor_id}",
                )
                tags = st.text_input(
                    "Tags",
                    value=", ".join(clean_tags(action.get("tags", []))),
                    key=f"action_tags_{suggestion_id}_{editor_id}",
                )
                driver = st.text_area(
                    "Adaptation rationale",
                    value=action.get("driver", ""),
                    key=f"action_driver_{suggestion_id}_{editor_id}",
                )
                edited.update(
                    {
                        "submodel": submodel,
                        "element_type": element_type,
                        "title": title,
                        "description": description,
                        "model_status": model_status,
                        "priority": priority,
                        "criticality": criticality,
                        "tags": clean_tags(tags),
                        "driver": driver,
                        "existing_element_id": "",
                        "relation_source_id": "",
                        "relation_target_id": "",
                        "relationship_kind": "RELATES_TO",
                    }
                )

            elif kind == "UPDATE_ELEMENT":
                if not local_elements:
                    st.warning("This unit has no element that can be updated.")
                    edited["existing_element_id"] = ""
                else:
                    local_ids = [str(e["id"]) for e in local_elements]
                    selected_id = st.selectbox(
                        "Existing local element",
                        options=local_ids,
                        index=safe_index(local_ids, action.get("existing_element_id")),
                        format_func=lambda item: element_label(
                            next(e for e in local_elements if str(e["id"]) == item)
                        ),
                        key=f"action_existing_{suggestion_id}_{editor_id}",
                    )
                    current = next(
                        e for e in local_elements if str(e["id"]) == selected_id
                    )
                    title = st.text_input(
                        "Revised title",
                        value=action.get("title") or current.get("title", ""),
                        key=f"action_update_title_{suggestion_id}_{editor_id}",
                    )
                    description = st.text_area(
                        "Revised description",
                        value=action.get("description")
                        or current.get("description", ""),
                        key=f"action_update_description_{suggestion_id}_{editor_id}",
                    )
                    c1, c2, c3 = st.columns(3)
                    model_status = c1.selectbox(
                        "Model state",
                        MODEL_STATUSES,
                        index=safe_index(
                            MODEL_STATUSES,
                            action.get("model_status")
                            or current.get("model_status", "PROPOSED"),
                            2,
                        ),
                        key=f"action_update_status_{suggestion_id}_{editor_id}",
                    )
                    priority = c2.selectbox(
                        "Priority",
                        PRIORITIES,
                        index=safe_index(
                            PRIORITIES,
                            action.get("priority", current.get("priority", "")),
                        ),
                        key=f"action_update_priority_{suggestion_id}_{editor_id}",
                    )
                    criticality = c3.selectbox(
                        "Criticality",
                        CRITICALITIES,
                        index=safe_index(
                            CRITICALITIES,
                            action.get(
                                "criticality", current.get("criticality", "")
                            ),
                        ),
                        key=f"action_update_criticality_{suggestion_id}_{editor_id}",
                    )
                    tags = st.text_input(
                        "Tags",
                        value=", ".join(
                            clean_tags(action.get("tags") or current.get("tags", []))
                        ),
                        key=f"action_update_tags_{suggestion_id}_{editor_id}",
                    )
                    driver = st.text_area(
                        "Adaptation rationale",
                        value=action.get("driver", ""),
                        key=f"action_update_driver_{suggestion_id}_{editor_id}",
                    )
                    edited.update(
                        {
                            "existing_element_id": selected_id,
                            "submodel": current.get("submodel", "Goals"),
                            "element_type": current.get("element_type", "Goal"),
                            "title": title,
                            "description": description,
                            "model_status": model_status,
                            "priority": priority,
                            "criticality": criticality,
                            "tags": clean_tags(tags),
                            "driver": driver,
                            "relation_source_id": "",
                            "relation_target_id": "",
                            "relationship_kind": "RELATES_TO",
                        }
                    )

            elif kind == "CREATE_RELATION":
                if not local_elements or len(all_elements) < 2:
                    st.warning(
                        "At least one local source element and a second target element are required."
                    )
                    edited["relation_source_id"] = ""
                    edited["relation_target_id"] = ""
                else:
                    local_ids = [str(e["id"]) for e in local_elements]
                    source_id = st.selectbox(
                        "Source element (owned by this unit)",
                        options=local_ids,
                        index=safe_index(local_ids, action.get("relation_source_id")),
                        format_func=lambda item: element_label(
                            next(e for e in local_elements if str(e["id"]) == item)
                        ),
                        key=f"action_relation_source_{suggestion_id}_{editor_id}",
                    )
                    target_elements = [
                        e for e in all_elements if str(e.get("id")) != source_id
                    ]
                    target_ids = [str(e["id"]) for e in target_elements]
                    target_id = st.selectbox(
                        "Target element",
                        options=target_ids,
                        index=safe_index(target_ids, action.get("relation_target_id")),
                        format_func=lambda item: element_label(
                            next(e for e in target_elements if str(e["id"]) == item),
                            unit_names,
                        ),
                        key=f"action_relation_target_{suggestion_id}_{editor_id}",
                    )
                    source = next(
                        e for e in local_elements if str(e["id"]) == source_id
                    )
                    target = next(
                        e for e in target_elements if str(e["id"]) == target_id
                    )
                    relationship_options = allowed_relationship_kinds(source, target)
                    relationship_kind = st.selectbox(
                        "4EM relationship",
                        options=relationship_options,
                        index=safe_index(
                            relationship_options, action.get("relationship_kind")
                        ),
                        key=f"action_relation_kind_{suggestion_id}_{editor_id}_{source_id}_{target_id}",
                    )
                    driver = st.text_area(
                        "Relationship rationale",
                        value=action.get("driver", ""),
                        key=f"action_relation_driver_{suggestion_id}_{editor_id}",
                    )
                    edited.update(
                        {
                            "relation_source_id": source_id,
                            "relation_target_id": target_id,
                            "relationship_kind": relationship_kind,
                            "driver": driver,
                            "existing_element_id": "",
                            "submodel": source.get("submodel", "Goals"),
                            "element_type": source.get("element_type", "Goal"),
                            "title": "",
                            "description": "",
                        }
                    )

            else:
                driver = st.text_area(
                    "Review rationale",
                    value=action.get("driver", ""),
                    key=f"action_no_change_driver_{suggestion_id}_{editor_id}",
                )
                edited.update(
                    {
                        "driver": driver,
                        "existing_element_id": "",
                        "relation_source_id": "",
                        "relation_target_id": "",
                    }
                )

            edited_actions.append(edited)

    st.session_state[state_key] = edited_actions
    with st.expander("Advanced: reviewed actions as JSON"):
        st.code(
            json.dumps([public_action(action) for action in edited_actions], indent=2),
            language="json",
        )
    return [public_action(action) for action in edited_actions]


def render_manual_adaptation_workspace(
    repo: GraphRepository,
    notification: dict[str, Any],
    current_unit: dict[str, Any],
    actor_name: str,
    units: list[dict[str, Any]],
) -> None:
    """Render the non-LLM workflow for adapting a notification locally."""
    notification_id = notification.get("id", "")
    unit_id = current_unit.get("id", "")
    source_change = notification.get("change", {})
    source_change_id = source_change.get("id", "")
    unit_names = {u["id"]: u["name"] for u in units}
    local_elements = repo.list_elements(unit_id=unit_id, limit=1500)
    all_elements = repo.list_elements(limit=3000)

    st.markdown("#### Manual adaptation workspace")
    st.caption(
        "Create or revise local elements and relationships while keeping a trace to "
        "the source unit's change. You can perform several operations before marking "
        "the adaptation complete."
    )
    mode = st.radio(
        "Manual adaptation operation",
        ["Add element", "Update element", "Add relationship"],
        horizontal=True,
        key=f"manual_mode_{notification_id}",
    )

    if mode == "Add element":
        submodel = st.selectbox(
            "4EM sub-model",
            list(SUBMODELS),
            format_func=lambda value: SUBMODELS[value],
            key=f"manual_add_submodel_{notification_id}",
        )
        with st.form(f"manual_add_element_{notification_id}", clear_on_submit=True):
            element_type = st.selectbox(
                "Element type", ELEMENT_TYPES[submodel]
            )
            title = st.text_input("Title")
            description = st.text_area("Description")
            c1, c2, c3 = st.columns(3)
            model_status = c1.selectbox("Model state", MODEL_STATUSES, index=2)
            priority = c2.selectbox("Priority", PRIORITIES)
            criticality = c3.selectbox("Criticality", CRITICALITIES)
            tags = st.text_input("Tags", value="manually-adapted")
            driver = st.text_area(
                "Adaptation rationale",
                value=f"Manual adaptation of change {source_change.get('entity_code', '')} from {notification.get('source_unit', {}).get('name', '')}.",
            )
            submitted = st.form_submit_button(
                "Add adapted element", type="primary"
            )
        if submitted:
            try:
                created = repo.create_element(
                    unit_id=unit_id,
                    actor_name=actor_name,
                    submodel=submodel,
                    element_type=element_type,
                    title=title,
                    description=description,
                    model_status=model_status,
                    priority=priority,
                    criticality=criticality,
                    tags=sorted(set(clean_tags(tags)) | {"manually-adapted"}),
                    driver=driver,
                    source_change_id=source_change_id,
                )
                repo.update_notification_status(notification_id, unit_id, "ADAPTING")
                st.success(f"Created adapted element {created.get('code')}.")
                st.rerun()
            except Exception as exc:
                show_error(exc)

    elif mode == "Update element":
        if not local_elements:
            st.info("Create a local element before updating one.")
        else:
            element_id = st.selectbox(
                "Local element",
                [str(e["id"]) for e in local_elements],
                format_func=lambda item: element_label(
                    next(e for e in local_elements if str(e["id"]) == item)
                ),
                key=f"manual_update_select_{notification_id}",
            )
            current = next(e for e in local_elements if str(e["id"]) == element_id)
            with st.form(f"manual_update_element_{notification_id}_{element_id}"):
                title = st.text_input("Title", value=current.get("title", ""))
                description = st.text_area(
                    "Description", value=current.get("description", "")
                )
                c1, c2, c3 = st.columns(3)
                model_status = c1.selectbox(
                    "Model state",
                    MODEL_STATUSES,
                    index=safe_index(
                        MODEL_STATUSES, current.get("model_status", "AS_IS")
                    ),
                )
                priority = c2.selectbox(
                    "Priority",
                    PRIORITIES,
                    index=safe_index(PRIORITIES, current.get("priority", "")),
                )
                criticality = c3.selectbox(
                    "Criticality",
                    CRITICALITIES,
                    index=safe_index(
                        CRITICALITIES, current.get("criticality", "")
                    ),
                )
                tags = st.text_input(
                    "Tags",
                    value=", ".join(
                        sorted(set(current.get("tags", [])) | {"manually-adapted"})
                    ),
                )
                driver = st.text_area(
                    "Adaptation rationale",
                    value=f"Manual adaptation of change {source_change.get('entity_code', '')} from {notification.get('source_unit', {}).get('name', '')}.",
                )
                submitted = st.form_submit_button(
                    "Save adapted version", type="primary"
                )
            if submitted:
                try:
                    repo.update_element(
                        element_id=element_id,
                        unit_id=unit_id,
                        actor_name=actor_name,
                        fields={
                            "title": title,
                            "description": description,
                            "model_status": model_status,
                            "priority": priority,
                            "criticality": criticality,
                            "tags": sorted(
                                set(clean_tags(tags)) | {"manually-adapted"}
                            ),
                        },
                        driver=driver,
                        expected_version=int(current.get("version", 1)),
                        source_change_id=source_change_id,
                    )
                    repo.update_notification_status(
                        notification_id, unit_id, "ADAPTING"
                    )
                    st.success("Adapted element version saved.")
                    st.rerun()
                except Exception as exc:
                    show_error(exc)

    else:
        if not local_elements or len(all_elements) < 2:
            st.info(
                "At least one local source element and a second model element are required."
            )
        else:
            source_id = st.selectbox(
                "Source element (owned by this unit)",
                [str(e["id"]) for e in local_elements],
                format_func=lambda item: element_label(
                    next(e for e in local_elements if str(e["id"]) == item)
                ),
                key=f"manual_relation_source_{notification_id}",
            )
            target_elements = [
                e for e in all_elements if str(e.get("id")) != source_id
            ]
            target_id = st.selectbox(
                "Target element",
                [str(e["id"]) for e in target_elements],
                format_func=lambda item: element_label(
                    next(e for e in target_elements if str(e["id"]) == item),
                    unit_names,
                ),
                key=f"manual_relation_target_{notification_id}",
            )
            source = next(e for e in local_elements if str(e["id"]) == source_id)
            target = next(e for e in target_elements if str(e["id"]) == target_id)
            permitted = allowed_relationship_kinds(source, target)
            with st.form(
                f"manual_add_relationship_{notification_id}_{source_id}_{target_id}"
            ):
                kind = st.selectbox("4EM relationship", permitted)
                notation = render_relationship_notation_inputs(
                    source,
                    target,
                    kind,
                    f"manual_relationship_{notification_id}",
                )
                rationale = st.text_area(
                    "Relationship rationale",
                    value=f"Manual adaptation of change {source_change.get('entity_code', '')} from {notification.get('source_unit', {}).get('name', '')}.",
                )
                submitted = st.form_submit_button(
                    "Add adapted relationship", type="primary"
                )
            if submitted:
                try:
                    relation = repo.create_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        kind=kind,
                        rationale=rationale,
                        unit_id=unit_id,
                        actor_name=actor_name,
                        source_change_id=source_change_id,
                        notation=notation,
                    )
                    repo.update_notification_status(
                        notification_id, unit_id, "ADAPTING"
                    )
                    st.success(
                        f"Created adapted relationship {relation.get('kind')}."
                    )
                    st.rerun()
                except Exception as exc:
                    show_error(exc)

    complete_col, keep_col = st.columns([1, 2])
    if complete_col.button(
        "Finish manual adaptation",
        type="primary",
        key=f"complete_manual_adaptation_{notification_id}",
    ):
        try:
            repo.update_notification_status(notification_id, unit_id, "ADAPTED")
            if st.session_state.get("manual_notification_id") == notification_id:
                st.session_state.pop("manual_notification_id", None)
            st.success("Notification marked as manually adapted.")
            st.rerun()
        except Exception as exc:
            show_error(exc)
    keep_col.caption(
        "Leave this workspace open when more than one local element or relationship must be adapted."
    )


def render_login(repo: GraphRepository) -> None:
    st.markdown(
        '<div class="app-kicker">Federated enterprise modelling</div>',
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.write(
        "Choose the organisational level and then the organisational-unit name "
        "whose local 4EM workspace you want to open. No password is requested."
    )
    st.warning(
        "This is view separation, not security authentication. Add an identity provider "
        "and server-side authorisation before using the app with untrusted users."
    )

    units = repo.list_units()

    # Treat persisted organisational units as the login directory. Bootstrap the
    # root unit only when the repository is empty.
    if not units:
        st.info(
            "No organisational units were found in Neo4j AuraDB. Create the first "
            "unit once; after that, login uses the level and unit-name dropdowns."
        )
        with st.form("first_unit_form"):
            name = st.text_input("Organisational-unit name", value="Organisation")
            level = st.text_input("Organisational level", value="Enterprise")
            description = st.text_area("Description")
            submitted = st.form_submit_button(
                "Create first organisational unit", type="primary"
            )
        if submitted:
            try:
                repo.create_unit(name, level, description, parent_id=None)
                st.success("First organisational unit created.")
                st.rerun()
            except Exception as exc:
                show_error(exc)
        st.stop()

    def clean_level(unit: dict[str, Any]) -> str:
        return str(unit.get("level") or "Unspecified level").strip()

    def natural_sort_key(value: str) -> list[tuple[int, Any]]:
        # Sort dotted numeric hierarchy labels naturally (1.2 before 1.10).
        parts = re.split(r"(\d+)", value.casefold())
        return [
            (0, int(part)) if part.isdigit() else (1, part)
            for part in parts
            if part != ""
        ]

    levels = sorted({clean_level(unit) for unit in units}, key=natural_sort_key)
    level_key = "login_selected_level"
    if st.session_state.get(level_key) not in levels:
        st.session_state[level_key] = levels[0]

    selected_level = st.selectbox(
        "Organisational level",
        options=levels,
        key=level_key,
        help="Levels are read from the existing OrgUnit records in Neo4j AuraDB.",
    )

    units_at_level = sorted(
        [unit for unit in units if clean_level(unit) == selected_level],
        key=lambda unit: (
            str(unit.get("name") or "").casefold(),
            str(unit.get("parent_name") or "").casefold(),
            str(unit.get("id") or ""),
        ),
    )
    unit_ids = [unit["id"] for unit in units_at_level]
    unit_key = "login_selected_unit"
    if st.session_state.get(unit_key) not in unit_ids:
        st.session_state[unit_key] = unit_ids[0]

    name_counts: dict[str, int] = {}
    for unit in units_at_level:
        unit_name = str(unit.get("name") or "Unnamed unit")
        name_counts[unit_name] = name_counts.get(unit_name, 0) + 1

    def unit_name_label(unit_id: str) -> str:
        unit = next(item for item in units_at_level if item["id"] == unit_id)
        unit_name = str(unit.get("name") or "Unnamed unit")
        if name_counts.get(unit_name, 0) == 1:
            return unit_name
        parent_name = str(unit.get("parent_name") or "No parent")
        return f"{unit_name} — parent: {parent_name}"

    selected_id = st.selectbox(
        "Organisational-unit name",
        options=unit_ids,
        format_func=unit_name_label,
        key=unit_key,
        help="Only units belonging to the selected organisational level are listed.",
    )
    selected_unit = next(unit for unit in units_at_level if unit["id"] == selected_id)

    details = [f"Level: {clean_level(selected_unit)}"]
    if selected_unit.get("parent_name"):
        details.append(f"Parent: {selected_unit['parent_name']}")
    st.caption(" · ".join(details))

    display_key = f"login_display_name_{selected_id}"
    if display_key not in st.session_state:
        st.session_state[display_key] = (
            selected_unit.get("last_display_name") or "Unit modeller"
        )
    actor_name = st.text_input(
        "Your display name",
        key=display_key,
        help=(
            "Recorded in the change history. Because this prototype has no user "
            "accounts, the remembered name is stored against the selected organisational unit."
        ),
    )
    st.caption(
        "The most recently used display name for this organisational unit is remembered in Neo4j AuraDB."
    )

    if st.button("Open workspace", type="primary"):
        try:
            cleaned_name = actor_name.strip() or "Unit modeller"
            repo.remember_display_name(selected_id, cleaned_name)
            st.session_state.current_unit_id = selected_id
            st.session_state.actor_name = cleaned_name
            st.rerun()
        except Exception as exc:
            show_error(exc)
    st.stop()


def render_sidebar(current_unit: dict[str, Any], llm: LLMService) -> None:
    with st.sidebar:
        st.markdown("### Current workspace")
        st.markdown(f"**{current_unit.get('name', '')}**")
        st.caption(current_unit.get("level", "Organisational unit"))
        st.write(f"Modeller: **{st.session_state.get('actor_name', 'Unit modeller')}**")
        st.divider()
        st.markdown("**Connections**")
        st.success("Neo4j AuraDB connected")
        if llm.configured:
            st.success(f"OpenAI ready · {OPENAI_MODEL}")
        else:
            st.warning("OpenAI key not configured")
        st.divider()
        if st.button("Switch organisational unit", use_container_width=True):
            st.session_state.pop("current_unit_id", None)
            st.session_state.pop("actor_name", None)
            st.rerun()


def render_element_table(elements: list[dict[str, Any]], unit_names: dict[str, str]) -> None:
    rows = [
        {
            "Code": e.get("code"),
            "Unit": unit_names.get(e.get("unit_id"), e.get("unit_id")),
            "Sub-model": e.get("submodel"),
            "Type": e.get("element_type"),
            "Title": e.get("title"),
            "State": e.get("model_status"),
            "Version": e.get("version"),
            "Last change": e.get("last_change"),
            "Driver": e.get("driver"),
        }
        for e in elements
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_change_table(changes: list[dict[str, Any]]) -> None:
    rows = [
        {
            "When": c.get("created_at"),
            "Unit": c.get("unit_name"),
            "Operation": c.get("operation"),
            "Kind": c.get("entity_kind"),
            "Element": c.get("entity_code"),
            "Name": c.get("entity_name"),
            "Driver": c.get("driver"),
            "By": c.get("actor_name"),
        }
        for c in changes
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def seed_minimal_4em(repo: GraphRepository, unit_id: str, actor_name: str) -> None:
    goal = repo.create_element(
        unit_id,
        actor_name,
        "Goals",
        "Goal",
        "Improve organisational adaptability",
        "Enable the unit to perceive, evaluate, and respond to relevant changes.",
        "AS_IS",
        "HIGH",
        "HIGH",
        ["demo", "fractal"],
        "Initial 4EM demonstration model.",
    )
    rule = repo.create_element(
        unit_id,
        actor_name,
        "Business Rules",
        "BusinessRule",
        "Every architecture change must include its driver",
        "A change is not committed without a rationale describing why it is needed.",
        "AS_IS",
        "",
        "",
        ["demo", "rationale"],
        "Initial 4EM demonstration model.",
    )
    concept = repo.create_element(
        unit_id,
        actor_name,
        "Concepts",
        "Concept",
        "Enterprise model change",
        "A versioned creation, update, retirement, or relationship modification in a local model.",
        "AS_IS",
        "",
        "",
        ["demo"],
        "Initial 4EM demonstration model.",
    )
    process = repo.create_element(
        unit_id,
        actor_name,
        "Business Processes",
        "Process",
        "Review and propagate a local model change",
        "Capture the change, preserve its driver, notify sibling and parent units, and let notified units decide whether to request adaptation advice.",
        "AS_IS",
        "",
        "",
        ["demo", "propagation"],
        "Initial 4EM demonstration model.",
    )
    role = repo.create_element(
        unit_id,
        actor_name,
        "Actors & Resources",
        "Role",
        "Unit modeller",
        "Maintains the local 4EM model, reviews automatic change notifications, and may request adaptation suggestions.",
        "AS_IS",
        "",
        "",
        ["demo"],
        "Initial 4EM demonstration model.",
    )
    technical = repo.create_element(
        unit_id,
        actor_name,
        "Technical Components & Requirements",
        "TechnicalComponent",
        "Neo4j AuraDB knowledge graph",
        "Stores local 4EM subgraphs, temporal changes, rationale, and propagation decisions.",
        "AS_IS",
        "",
        "",
        ["demo", "neo4j"],
        "Initial 4EM demonstration model.",
    )
    demo_relations = [
        (goal, process, "MOTIVATES"),
        (rule, process, "TRIGGERS"),
        (process, concept, "REFERS_TO"),
        (role, process, "PERFORMS"),
        (technical, process, "SUPPORTS"),
        (rule, goal, "SUPPORTS"),
    ]
    for source, target, kind in demo_relations:
        repo.create_relationship(
            source["id"],
            target["id"],
            kind,
            "Initial inter-model traceability link.",
            unit_id,
            actor_name,
        )


# Primary view renderers


def render_overview(
    repo: GraphRepository,
    current_unit: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    unit_id = current_unit["id"]
    metrics = repo.metrics(unit_id)
    st.subheader("Local model overview")
    cols = st.columns(4)
    cols[0].metric("Active 4EM elements", metrics["elements"])
    cols[1].metric("Model relationships", metrics["relations"])
    cols[2].metric("Recorded changes", metrics["changes"])
    cols[3].metric("Open notifications", metrics["pending"])

    elements = repo.list_elements(unit_id=unit_id, limit=300)
    relationships = repo.list_relationships(element_unit_id=unit_id, limit=500)
    if not elements:
        st.info("This unit does not yet have any 4EM elements.")
        if st.button("Load a minimal six-sub-model example", type="primary"):
            try:
                seed_minimal_4em(
                    repo, unit_id, st.session_state.get("actor_name", "Unit modeller")
                )
                st.success("Example model created.")
                st.rerun()
            except Exception as exc:
                show_error(exc)
        return

    goal_elements = [e for e in elements if e.get("submodel") == "Goals"]
    recent_changes = repo.list_changes(unit_id=unit_id, limit=8)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Current goal-oriented view")
        if goal_elements:
            render_interactive_model_graph(
                goal_elements,
                relationships,
                units,
                graph_key=f"overview-goals:{unit_id}",
                height=520,
            )
        else:
            st.caption("No Goals Model elements have been defined yet.")
    with right:
        st.markdown("#### Recent rationale-aware changes")
        for change in recent_changes:
            st.markdown(
                f"**{change.get('operation')} · {change.get('entity_code', '')}**  \n"
                f"{change.get('entity_name', '')}  \n"
                f"_Driver:_ {change.get('driver', '')}"
            )
            st.caption(f"{change.get('created_at', '')} · {change.get('actor_name', '')}")


def render_my_model(
    repo: GraphRepository,
    current_unit: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    unit_id = current_unit["id"]
    actor_name = st.session_state.get("actor_name", "Unit modeller")
    unit_names = {u["id"]: u["name"] for u in units}

    st.subheader("My unit's 4EM model")
    selected_submodel = st.selectbox(
        "Sub-model view",
        options=["All six sub-models"] + list(SUBMODELS),
        format_func=lambda x: SUBMODELS.get(x, x),
    )
    submodel_filter = None if selected_submodel == "All six sub-models" else selected_submodel
    elements = repo.list_elements(unit_id=unit_id, submodel=submodel_filter, limit=1000)
    all_local_elements = repo.list_elements(unit_id=unit_id, limit=1000)
    all_elements = repo.list_elements(limit=2000)
    relationships = repo.list_relationships(element_unit_id=unit_id, limit=2000)

    view_graph, view_table = st.tabs(["Graph", "Element register"])
    with view_graph:
        if len(elements) > MAX_GRAPH_ELEMENTS:
            st.warning(
                f"The graph is limited to the first {MAX_GRAPH_ELEMENTS} elements. Use the sub-model filter for a focused view."
            )
        if elements:
            render_interactive_model_graph(
                elements,
                relationships,
                units,
                graph_key=f"local-model:{unit_id}:{selected_submodel}",
                height=720,
            )
        else:
            st.info("No active elements in this view.")
    with view_table:
        render_element_table(elements, unit_names)

    st.markdown("#### Model maintenance")
    add_tab, edit_tab, relation_tab, remove_relation_tab, export_tab = st.tabs(
        ["Add element", "Edit or retire", "Add relationship", "Retire relationship", "Export"]
    )

    with add_tab:
        submodel = st.selectbox(
            "4EM sub-model", options=list(SUBMODELS), key="add_element_submodel"
        )
        with st.form("add_element_form", clear_on_submit=True):
            element_type = st.selectbox(
                "Element type", options=ELEMENT_TYPES[submodel]
            )
            title = st.text_input("Title")
            description = st.text_area("Description")
            col1, col2, col3 = st.columns(3)
            model_status = col1.selectbox("Model state", MODEL_STATUSES)
            priority = col2.selectbox("Priority", PRIORITIES)
            criticality = col3.selectbox("Criticality", CRITICALITIES)
            tags = st.text_input("Tags", help="Comma-separated")
            driver = st.text_area(
                "Change driver / rationale",
                help="Required. This is stored in the temporal change history.",
            )
            submitted = st.form_submit_button("Create 4EM element", type="primary")
        if submitted:
            try:
                created = repo.create_element(
                    unit_id,
                    actor_name,
                    submodel,
                    element_type,
                    title,
                    description,
                    model_status,
                    priority,
                    criticality,
                    clean_tags(tags),
                    driver,
                )
                st.success(f"Created {created.get('code')}.")
                st.rerun()
            except Exception as exc:
                show_error(exc)

    with edit_tab:
        if not all_local_elements:
            st.info("Create an element before editing it.")
        else:
            edit_id = st.selectbox(
                "Element",
                options=[e["id"] for e in all_local_elements],
                format_func=lambda item: element_label(
                    next(e for e in all_local_elements if e["id"] == item)
                ),
                key="edit_element_select",
            )
            current = next(e for e in all_local_elements if e["id"] == edit_id)
            with st.form(f"edit_element_form_{edit_id}"):
                st.caption(
                    f"{current.get('submodel')} · {current.get('element_type')} · version {current.get('version')}"
                )
                title = st.text_input("Title", value=current.get("title", ""))
                description = st.text_area(
                    "Description", value=current.get("description", "")
                )
                c1, c2, c3 = st.columns(3)
                model_status = c1.selectbox(
                    "Model state",
                    MODEL_STATUSES,
                    index=MODEL_STATUSES.index(current.get("model_status", "AS_IS"))
                    if current.get("model_status", "AS_IS") in MODEL_STATUSES
                    else 0,
                )
                priority = c2.selectbox(
                    "Priority",
                    PRIORITIES,
                    index=PRIORITIES.index(current.get("priority", ""))
                    if current.get("priority", "") in PRIORITIES
                    else 0,
                )
                criticality = c3.selectbox(
                    "Criticality",
                    CRITICALITIES,
                    index=CRITICALITIES.index(current.get("criticality", ""))
                    if current.get("criticality", "") in CRITICALITIES
                    else 0,
                )
                tags = st.text_input(
                    "Tags", value=", ".join(current.get("tags", []))
                )
                driver = st.text_area("Why is this element being changed?")
                update_submitted = st.form_submit_button("Save version", type="primary")
            if update_submitted:
                try:
                    repo.update_element(
                        edit_id,
                        unit_id,
                        actor_name,
                        {
                            "title": title,
                            "description": description,
                            "model_status": model_status,
                            "priority": priority,
                            "criticality": criticality,
                            "tags": clean_tags(tags),
                        },
                        driver,
                        int(current.get("version", 1)),
                    )
                    st.success("New version saved.")
                    st.rerun()
                except Exception as exc:
                    show_error(exc)

            with st.expander("Retire this element"):
                with st.form(f"retire_element_form_{edit_id}"):
                    retire_reason = st.text_area("Retirement rationale")
                    confirmed = st.checkbox(
                        "I understand this is a soft deletion and the history remains available."
                    )
                    retire_submitted = st.form_submit_button("Retire element")
                if retire_submitted:
                    if not confirmed:
                        st.error("Confirm the retirement first.")
                    else:
                        try:
                            repo.retire_element(
                                edit_id, unit_id, actor_name, retire_reason
                            )
                            st.success("Element retired.")
                            st.rerun()
                        except Exception as exc:
                            show_error(exc)

    with relation_tab:
        if not all_local_elements:
            st.info("Create a local source element first.")
        elif len(all_elements) < 2:
            st.info("At least two model elements are required.")
        else:
            source_id = st.selectbox(
                "Source element (must belong to this unit)",
                options=[e["id"] for e in all_local_elements],
                format_func=lambda item: element_label(
                    next(e for e in all_local_elements if e["id"] == item)
                ),
                key="relation_source",
            )
            target_options = [e for e in all_elements if e["id"] != source_id]
            target_id = st.selectbox(
                "Target element",
                options=[e["id"] for e in target_options],
                format_func=lambda item: element_label(
                    next(e for e in target_options if e["id"] == item), unit_names
                ),
                key="relation_target",
            )
            source = next(e for e in all_local_elements if e["id"] == source_id)
            target = next(e for e in target_options if e["id"] == target_id)
            permitted = allowed_relationship_kinds(source, target)
            with st.form("add_relationship_form", clear_on_submit=False):
                kind = st.selectbox("4EM relationship", options=permitted)
                notation = render_relationship_notation_inputs(
                    source, target, kind, "add_relationship"
                )
                rationale = st.text_area("Relationship rationale")
                relation_submitted = st.form_submit_button(
                    "Create relationship", type="primary"
                )
            if relation_submitted:
                try:
                    repo.create_relationship(
                        source_id,
                        target_id,
                        kind,
                        rationale,
                        unit_id,
                        actor_name,
                        notation=notation,
                    )
                    st.success("Relationship created.")
                    st.rerun()
                except Exception as exc:
                    show_error(exc)

    with remove_relation_tab:
        owned_relationships = repo.list_relationships(
            relation_owner_unit_id=unit_id, limit=2000
        )
        all_by_id = {e["id"]: e for e in all_elements}
        if not owned_relationships:
            st.info("This unit owns no active model relationships.")
        else:
            rel_id = st.selectbox(
                "Relationship",
                options=[r["id"] for r in owned_relationships],
                format_func=lambda item: relationship_label(
                    next(r for r in owned_relationships if r["id"] == item),
                    all_by_id,
                ),
            )
            with st.form("retire_relationship_form"):
                reason = st.text_area("Retirement rationale")
                rel_submitted = st.form_submit_button("Retire relationship")
            if rel_submitted:
                try:
                    repo.retire_relationship(rel_id, unit_id, actor_name, reason)
                    st.success("Relationship retired.")
                    st.rerun()
                except Exception as exc:
                    show_error(exc)

    with export_tab:
        export = repo.export_unit(unit_id)
        st.download_button(
            "Download this unit's complete JSON export",
            data=json.dumps(export, ensure_ascii=False, indent=2, default=str),
            file_name=f"{safe_dot_id(current_unit.get('name', 'unit')).lower()}_4em_export.json",
            mime="application/json",
        )
        st.caption(
            "The export includes active and retired elements, relationships, temporal changes, automatic notifications, and adaptation decisions."
        )


def render_federated_view(
    repo: GraphRepository,
    current_unit: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    st.subheader("Federated organisation view")
    st.caption(
        "All units are visible for learning; only the logged-in unit is editable. Related-unit changes are delivered as notifications."
    )
    unit_names = {u["id"]: u["name"] for u in units}
    scope_options = ["ALL"] + [u["id"] for u in units]
    selected_scope = st.selectbox(
        "Organisational scope",
        scope_options,
        format_func=lambda x: "All organisational units" if x == "ALL" else unit_names[x],
    )
    selected_submodel = st.selectbox(
        "4EM sub-model",
        ["ALL"] + list(SUBMODELS),
        format_func=lambda x: "All six sub-models" if x == "ALL" else SUBMODELS[x],
        key="federated_submodel",
    )
    elements = repo.list_elements(
        unit_id=None if selected_scope == "ALL" else selected_scope,
        submodel=None if selected_submodel == "ALL" else selected_submodel,
        limit=3000,
    )
    relationships = repo.list_relationships(limit=4000)
    if elements:
        if len(elements) > MAX_GRAPH_ELEMENTS:
            st.warning(
                f"The visual graph shows the first {MAX_GRAPH_ELEMENTS} elements. Narrow the scope for a complete focused view."
            )
        render_interactive_model_graph(
            elements,
            relationships,
            units,
            graph_key=f"federated-model:{selected_scope}:{selected_submodel}",
            height=760,
        )
        render_element_table(elements, unit_names)
    else:
        st.info("No elements in the selected federated view.")

    st.markdown("#### Self-similarity explorer")
    local_elements = repo.list_elements(unit_id=current_unit["id"], limit=1000)
    other_elements = [
        e for e in repo.list_elements(limit=3000) if e.get("unit_id") != current_unit["id"]
    ]
    if not local_elements or not other_elements:
        st.caption("Add comparable elements in at least two units to explore self-similarity.")
        return
    selected_id = st.selectbox(
        "Local element",
        options=[e["id"] for e in local_elements],
        format_func=lambda item: element_label(
            next(e for e in local_elements if e["id"] == item)
        ),
        key="similarity_element",
    )
    local = next(e for e in local_elements if e["id"] == selected_id)
    candidates = [
        e
        for e in other_elements
        if e.get("submodel") == local.get("submodel")
        and e.get("element_type") == local.get("element_type")
    ]
    scored = sorted(
        [
            {
                "Similarity": round(lexical_similarity(local, candidate), 3),
                "Unit": unit_names.get(candidate.get("unit_id"), ""),
                "Code": candidate.get("code"),
                "Title": candidate.get("title"),
                "Description": candidate.get("description"),
            }
            for candidate in candidates
        ],
        key=lambda row: row["Similarity"],
        reverse=True,
    )[:20]
    if scored:
        st.dataframe(scored, use_container_width=True, hide_index=True)
        st.caption(
            "Similarity is a transparent token-overlap aid, not an automatic equivalence decision. OpenAI adaptation analysis occurs only when a notified unit explicitly requests it."
        )
    else:
        st.caption("No same-type elements exist in other organisational units.")


def render_change_propagation(
    repo: GraphRepository,
    llm: LLMService,
    current_unit: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    unit_id = current_unit["id"]
    actor_name = st.session_state.get("actor_name", "Unit modeller")

    notifications_tab, suggestions_tab, history_tab = st.tabs(
        ["Change notifications", "Adaptation suggestions", "Change history"]
    )

    with notifications_tab:
        st.subheader("Changes made by related fractal units")
        st.caption(
            "A new notification is created automatically when a sibling unit or this "
            "unit's immediate child changes its 4EM model. For example, a change in "
            "1.1.1 notifies 1.1.2, 1.1.3, … and the parent unit 1.1."
        )
        status_filter = st.multiselect(
            "Notification statuses",
            ["UNREAD", "READ", "ANALYSED", "ADAPTING", "ADAPTED", "DISMISSED"],
            default=["UNREAD", "READ", "ADAPTING"],
            key="notification_status_filter",
        )
        notifications = repo.list_notifications(
            target_unit_id=unit_id,
            statuses=status_filter,
            limit=400,
        )
        if not notifications:
            st.info("No change notifications match the selected statuses.")
        else:
            notification_id = st.selectbox(
                "Notification",
                options=[n["id"] for n in notifications],
                format_func=lambda item: (
                    f"[{next(n for n in notifications if n['id'] == item).get('status')}] "
                    f"{next(n for n in notifications if n['id'] == item).get('change', {}).get('operation')} · "
                    f"{next(n for n in notifications if n['id'] == item).get('change', {}).get('entity_code')} · "
                    f"from {next(n for n in notifications if n['id'] == item).get('source_unit_name')}"
                ),
                key="selected_change_notification",
            )
            notification = repo.get_notification(notification_id, unit_id)
            if notification:
                change = notification.get("change", {})
                source_unit = notification.get("source_unit", {})
                st.markdown(
                    f"<span class='status-pill'>{notification.get('status')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"### {change.get('operation', '')} · "
                    f"{change.get('entity_code', '')} · {change.get('entity_name', '')}"
                )
                cols = st.columns(3)
                cols[0].metric("Source unit", source_unit.get("name", ""))
                cols[1].metric("4EM area", change.get("submodel", ""))
                cols[2].metric("Changed", change.get("created_at", ""))
                st.markdown(f"**Change rationale:** {change.get('driver', '')}")
                st.caption(
                    "You were notified because your unit is either a sibling of the "
                    "source unit under the same parent, or the immediate parent of the source unit."
                )
                with st.expander("Complete source change, including before/after snapshots"):
                    st.json(change)

                existing_suggestion = notification.get("suggestion") or {}
                if existing_suggestion:
                    st.info(
                        f"An adaptation analysis already exists: "
                        f"{existing_suggestion.get('title', 'Untitled')} "
                        f"[{existing_suggestion.get('status', '')}]. "
                        "Open the Adaptation suggestions tab to review it."
                    )

                mark_col, manual_col, analyse_col, dismiss_col = st.columns(4)
                if mark_col.button(
                    "Mark as read",
                    key=f"read_notification_{notification_id}",
                    disabled=notification.get("status") in {"ANALYSED", "ADAPTING", "ADAPTED", "DISMISSED"},
                ):
                    try:
                        repo.update_notification_status(notification_id, unit_id, "READ")
                        st.success("Notification marked as read.")
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)

                if manual_col.button(
                    "Adapt manually",
                    key=f"manual_notification_{notification_id}",
                    disabled=notification.get("status") == "DISMISSED",
                ):
                    try:
                        repo.update_notification_status(
                            notification_id, unit_id, "ADAPTING"
                        )
                        st.session_state.manual_notification_id = notification_id
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)

                if analyse_col.button(
                    "Ask OpenAI for adaptation suggestions",
                    type="primary",
                    key=f"analyse_notification_{notification_id}",
                    disabled=(not llm.configured or bool(existing_suggestion)),
                ):
                    try:
                        target_elements = repo.list_elements(
                            unit_id=unit_id,
                            limit=MAX_TARGET_ELEMENTS_FETCH,
                        )
                        target_relationships = repo.list_relationships(
                            element_unit_id=unit_id,
                            limit=MAX_TARGET_RELATIONSHIPS_FETCH,
                        )
                        source_change = repo.get_change(change.get("id", ""))
                        if not source_change:
                            raise ValueError("The source change could not be loaded.")
                        source_unit_for_analysis = (
                            repo.get_unit(source_unit.get("id", "")) or source_unit
                        )
                        with st.spinner(
                            "Comparing the source change with this unit's current 4EM model…"
                        ):
                            analysis = llm.analyse_change_for_unit(
                                source_change,
                                source_unit_for_analysis,
                                current_unit,
                                target_elements,
                                target_relationships,
                            )
                            saved = repo.save_suggestion(
                                change_id=change.get("id", ""),
                                source_unit_id=source_unit.get("id", ""),
                                target_unit_id=unit_id,
                                analysis=analysis,
                                notification_id=notification_id,
                            )
                        state = (
                            "adaptation recommended"
                            if saved.get("relevant")
                            else "no adaptation recommended"
                        )
                        st.success(f"Analysis complete: {state}.")
                        with st.expander("OpenAI result"):
                            st.json(analysis)
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)

                if dismiss_col.button(
                    "Dismiss notification",
                    key=f"dismiss_notification_{notification_id}",
                    disabled=notification.get("status") in {"ANALYSED", "ADAPTING", "ADAPTED"},
                ):
                    try:
                        repo.update_notification_status(
                            notification_id, unit_id, "DISMISSED"
                        )
                        st.success("Notification dismissed without requesting an LLM analysis.")
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)

                if not llm.configured:
                    st.warning(
                        "Set OPENAI_API_KEY at the top of the script or as an environment "
                        "variable to request adaptation suggestions. Notifications work without OpenAI."
                    )
                manual_open = (
                    st.session_state.get("manual_notification_id") == notification_id
                    or notification.get("status") == "ADAPTING"
                )
                if manual_open:
                    st.divider()
                    render_manual_adaptation_workspace(
                        repo, notification, current_unit, actor_name, units
                    )


    with suggestions_tab:
        st.subheader("LLM suggestions requested by this unit")
        st.caption(
            "These suggestions exist only because a user in this unit selected a change "
            "notification and explicitly requested an OpenAI analysis."
        )
        suggestion_statuses = st.multiselect(
            "Suggestion statuses",
            ["PENDING", "NOT_RECOMMENDED", "ADOPTED", "MODIFIED", "REJECTED", "REVIEWED"],
            default=["PENDING", "NOT_RECOMMENDED"],
            key="adaptation_suggestion_status_filter",
        )
        suggestions = repo.list_suggestions(
            target_unit_id=unit_id,
            statuses=suggestion_statuses,
            limit=400,
        )
        if not suggestions:
            st.info("No adaptation suggestions match the selected statuses.")
        else:
            suggestion_id = st.selectbox(
                "Adaptation suggestion",
                options=[s["id"] for s in suggestions],
                format_func=lambda item: (
                    f"[{next(s for s in suggestions if s['id'] == item).get('status')}] "
                    f"{next(s for s in suggestions if s['id'] == item).get('title')} — "
                    f"source: {next(s for s in suggestions if s['id'] == item).get('source_unit_name')}"
                ),
                key="selected_adaptation_suggestion",
            )
            suggestion = repo.get_suggestion(suggestion_id)
            if suggestion:
                st.markdown(
                    f"<span class='status-pill'>{suggestion.get('status')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {suggestion.get('title', '')}")
                cols = st.columns(3)
                cols[0].metric(
                    "Candidate fit", suggestion.get("candidate_fit", "LEGACY")
                )
                cols[1].metric(
                    "Source unit", suggestion.get("source_unit", {}).get("name", "")
                )
                cols[2].metric(
                    "Source operation", suggestion.get("change", {}).get("operation", "")
                )
                st.write(
                    suggestion.get("target_obligation_summary")
                    or suggestion.get("summary", "")
                )
                candidate_id = suggestion.get("candidate_target_element_id", "")
                if candidate_id:
                    st.markdown(f"**Selected local counterpart:** `{candidate_id}`")
                candidate_explanation = suggestion.get("candidate_fit_explanation", "")
                if candidate_explanation:
                    st.markdown(f"**Candidate assessment:** {candidate_explanation}")
                st.markdown(f"**Applicability rationale:** {suggestion.get('rationale', '')}")
                st.markdown(
                    f"**Concrete local effect:** {suggestion.get('target_impact', '')}"
                )
                with st.expander("Source change and rationale"):
                    st.json(suggestion.get("change", {}))

                actions_default = suggestion.get("proposed_actions_json", "[]")
                try:
                    original_actions = json.loads(actions_default)
                    if not isinstance(original_actions, list):
                        original_actions = []
                except json.JSONDecodeError:
                    original_actions = []
                local_elements = repo.list_elements(unit_id=unit_id, limit=1500)
                # Candidate-first analysis is scoped to the target unit; keep
                # relationship endpoints within that boundary during review.
                all_elements = local_elements
                unit_names = {u["id"]: u["name"] for u in units}
                st.markdown("#### Review and manually adapt the proposed changes")
                edited_actions = render_structured_action_editor(
                    suggestion_id,
                    original_actions,
                    local_elements,
                    all_elements,
                    unit_names,
                )
                resolution_note = st.text_area(
                    "Decision note",
                    key=f"resolution_note_{suggestion_id}",
                )
                apply_col, reject_col, review_col = st.columns(3)
                unresolved = suggestion.get("status") in {"PENDING", "NOT_RECOMMENDED"}

                if apply_col.button(
                    "Apply reviewed actions",
                    type="primary",
                    key=f"apply_{suggestion_id}",
                    disabled=not unresolved,
                ):
                    try:
                        reviewed_original = [
                            public_action(normalise_adaptation_action(action))
                            for action in original_actions
                        ]
                        applied, errors = apply_suggestion_actions(
                            repo, suggestion, unit_id, actor_name, edited_actions
                        )
                        for message in applied:
                            st.success(message)
                        for message in errors:
                            st.error(message)
                        if not errors:
                            status = (
                                "ADOPTED"
                                if edited_actions == reviewed_original
                                else "MODIFIED"
                            )
                            repo.resolve_suggestion(
                                suggestion_id,
                                unit_id,
                                status,
                                resolution_note,
                            )
                            st.success(f"Suggestion marked {status.lower()}.")
                            st.rerun()
                        else:
                            st.warning(
                                "The suggestion remains unresolved because one or more "
                                "actions failed validation."
                            )
                    except Exception as exc:
                        show_error(exc)

                if reject_col.button(
                    "Reject suggestion",
                    key=f"reject_{suggestion_id}",
                    disabled=not unresolved,
                ):
                    try:
                        repo.resolve_suggestion(
                            suggestion_id,
                            unit_id,
                            "REJECTED",
                            resolution_note
                            or "Rejected by the notified organisational unit.",
                        )
                        st.success("Suggestion rejected.")
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)

                if review_col.button(
                    "Mark reviewed, no change",
                    key=f"review_{suggestion_id}",
                    disabled=not unresolved,
                ):
                    try:
                        repo.resolve_suggestion(
                            suggestion_id,
                            unit_id,
                            "REVIEWED",
                            resolution_note
                            or "Reviewed; no local model change was made.",
                        )
                        st.success("Suggestion marked reviewed.")
                        st.rerun()
                    except Exception as exc:
                        show_error(exc)

    with history_tab:
        changes = repo.list_changes(unit_id=unit_id, limit=500)
        render_change_table(changes)
        if changes:
            selected_id = st.selectbox(
                "Inspect complete before/after snapshots",
                options=[c["id"] for c in changes],
                format_func=lambda item: (
                    f"{next(c for c in changes if c['id'] == item).get('operation')} · "
                    f"{next(c for c in changes if c['id'] == item).get('entity_code')} · "
                    f"{next(c for c in changes if c['id'] == item).get('created_at')}"
                ),
                key="history_change_inspect",
            )
            detail = repo.get_change(selected_id)
            if detail:
                before_col, after_col = st.columns(2)
                with before_col:
                    st.markdown("**Before**")
                    try:
                        st.json(json.loads(detail.get("before_json", "{}")))
                    except json.JSONDecodeError:
                        st.code(detail.get("before_json", ""))
                with after_col:
                    st.markdown("**After**")
                    try:
                        st.json(json.loads(detail.get("after_json", "{}")))
                    except json.JSONDecodeError:
                        st.code(detail.get("after_json", ""))



def render_organisation(
    repo: GraphRepository,
    current_unit: dict[str, Any],
    units: list[dict[str, Any]],
) -> None:
    st.subheader("Organisational-unit hierarchy")
    st.graphviz_chart(build_hierarchy_dot(units), use_container_width=True)
    st.dataframe(
        [
            {
                "Name": u.get("name"),
                "Level": u.get("level"),
                "Parent": u.get("parent_name") or "—",
                "Description": u.get("description"),
                "ID": u.get("id"),
            }
            for u in units
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Add a direct child unit")
    st.caption(
        "The new unit receives an empty local 4EM view and appears as a separate no-password login option."
    )
    with st.form("create_child_unit_form", clear_on_submit=True):
        name = st.text_input("Unit name")
        level = st.text_input("Organisational level", value="Department")
        description = st.text_area("Description")
        submitted = st.form_submit_button("Create child unit", type="primary")
    if submitted:
        if not name.strip():
            st.error("A unit name is required.")
        else:
            try:
                created = repo.create_unit(
                    name, level, description, parent_id=current_unit["id"]
                )
                st.success(f"Created {created.get('name')}.")
                st.rerun()
            except Exception as exc:
                show_error(exc)


# Application bootstrap


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="◫",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    if is_placeholder(NEO4J_URI) or is_placeholder(NEO4J_PASSWORD):
        st.title(APP_TITLE)
        st.error("Neo4j AuraDB is not configured.")
        st.code(
            "\n".join(
                [
                    'NEO4J_URI = "neo4j+s://<instance>.databases.neo4j.io"',
                    'NEO4J_USERNAME = "neo4j"',
                    'NEO4J_PASSWORD = "<AuraDB password>"',
                    'NEO4J_DATABASE = "neo4j"',
                    'OPENAI_API_KEY = "<OpenAI API key>"',
                ]
            ),
            language="python",
        )
        st.stop()

    try:
        driver = cached_driver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        repo = GraphRepository(driver, NEO4J_DATABASE)
        repo.bootstrap_schema()
    except Exception as exc:
        st.title(APP_TITLE)
        st.error("Could not connect to or initialise Neo4j AuraDB.")
        st.exception(exc)
        st.stop()

    llm = LLMService(OPENAI_API_KEY, OPENAI_MODEL)

    if not st.session_state.get("current_unit_id"):
        render_login(repo)

    current_unit = repo.get_unit(st.session_state.current_unit_id)
    if not current_unit:
        st.session_state.pop("current_unit_id", None)
        st.rerun()

    units = repo.list_units()
    render_sidebar(current_unit, llm)

    st.markdown('<div class="app-kicker">Knowledge graph · 4EM · rationale-aware propagation</div>', unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption(
        f"Logged in as {current_unit.get('name')} · local edits are isolated to this unit's subgraph."
    )

    overview_tab, model_tab, federated_tab, propagation_tab, org_tab = st.tabs(
        [
            "Overview",
            "My 4EM model",
            "Federated view",
            "Change notifications",
            "Organisation",
        ]
    )
    with overview_tab:
        render_overview(repo, current_unit, units)
    with model_tab:
        render_my_model(repo, current_unit, units)
    with federated_tab:
        render_federated_view(repo, current_unit, units)
    with propagation_tab:
        render_change_propagation(repo, llm, current_unit, units)
    with org_tab:
        render_organisation(repo, current_unit, units)


if __name__ == "__main__":
    main()
