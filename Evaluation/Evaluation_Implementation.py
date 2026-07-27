"""Evaluate LLM-proposed 4EM model changes for curriculum propagation.

The evaluator makes one OpenAI Responses API call for each fixed
scenario-target pair. It scores only the concrete model effect:

- impacted existing element IDs;
- model operation types; and
- exact agreement on both sets.

It does not score ADOPT, MODIFY, REJECT, confidence, or explanation quality.
It reads the current target models from Neo4j and never writes changes back.

Install:
    pip install openai neo4j

Run:
    python run_model_change_evaluation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from neo4j import GraphDatabase
except ImportError as exc: 
    raise SystemExit("Install Neo4j first: pip install neo4j") from exc

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Install OpenAI first: pip install openai") from exc


# Configuration

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "30000"))

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_CHANGES_PATH = SCRIPT_DIR / "data" / "source_changes.csv"
EXPECTED_CHANGES_PATH = SCRIPT_DIR / "data" / "expected_model_changes.csv"
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(SCRIPT_DIR / "results"))).expanduser()

# Fetch broadly for deterministic ranking, but keep the prompt context bounded.
MAX_TARGET_ELEMENTS_FETCH = int(os.getenv("MAX_TARGET_ELEMENTS_FETCH", "5000"))
MAX_TARGET_RELATIONSHIPS_FETCH = int(os.getenv("MAX_TARGET_RELATIONSHIPS_FETCH", "5000"))
MAX_LLM_ELEMENTS = int(os.getenv("MAX_LLM_ELEMENTS", "160"))
MAX_LLM_RELATIONSHIPS = int(os.getenv("MAX_LLM_RELATIONSHIPS", "240"))
CANDIDATE_SHORTLIST_SIZE = int(os.getenv("CANDIDATE_SHORTLIST_SIZE", "8"))
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "0")) or None
RESUME_COMPLETED_PAIRS = os.getenv("RESUME_COMPLETED_PAIRS", "true").lower() == "true"
STOP_ON_ERROR = os.getenv("STOP_ON_ERROR", "true").lower() == "true"


# 4EM vocabulary

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

ALL_ELEMENT_TYPES = sorted({item for values in ELEMENT_TYPES.values() for item in values})

SUBMODEL_TO_UPDATE_OPERATION = {
    "Goals": "UPDATE_GOAL",
    "Business Rules": "UPDATE_RULE",
    "Concepts": "UPDATE_CONCEPT",
    "Business Processes": "UPDATE_PROCESS",
    "Actors & Resources": "UPDATE_ACTOR_RESOURCE",
    "Technical Components & Requirements": "UPDATE_TECHNICAL_COMPONENT",
}

SUBMODEL_TO_CREATE_OPERATION = {
    "Goals": "CREATE_GOAL",
    "Business Rules": "CREATE_RULE",
    "Concepts": "CREATE_CONCEPT",
    "Business Processes": "CREATE_PROCESS",
    "Actors & Resources": "CREATE_ACTOR_RESOURCE",
    "Technical Components & Requirements": "CREATE_TECHNICAL_COMPONENT",
}

ACTUAL_HEADERS = [
    "scenario_id",
    "target_unit_id",
    "impacted_element_ids_json",
    "model_operations_json",
]

SOURCE_HEADERS = [
    "scenario_id",
    "scenario_name",
    "programme",
    "source_unit_id",
    "source_level",
    "scenario_description",
    "trigger_evidence",
    "change_id",
    "operation",
    "entity_kind",
    "affected_element_id",
    "changed_property",
    "before_json",
    "after_json",
    "driver",
]

EXPECTED_HEADERS = [
    "scenario_id",
    "target_unit_id",
    "expected_impacted_element_ids_json",
    "expected_model_operations_json",
]

INSTRUCTIONS = r"""
You are a 4EM change-propagation analyst. Evaluate exactly one source change
against exactly one target organisational unit and its current local 4EM model.
Your scored output is only the concrete model effect: impacted existing element
IDs and model operation types. Select the target correspondence before choosing
an action, and return the smallest action set that is actually necessary.

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
proximity, programme membership and notification receipt are also insufficient.

STEP 2 — APPLICABILITY
A change applies only when the target performs, owns, consumes, produces, depends
on, or is constrained by an operative clause in the source after-state. Do not
reject merely because local terminology differs when a functional analogue exists.
Do reject source-domain-specific obligations when the target has no corresponding
activity or dependency.

STEP 3 — SMALLEST SCORED ACTION SET
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


# Utilities


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def parse_json_list(value: str, field: str) -> list[Any]:
    parsed = json.loads((value or "[]").strip() or "[]")
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must contain a JSON list")
    return parsed


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
    """Deterministic structural-first shortlist for exact target-ID selection."""
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

        ranked.append({
            "id": element.get("id"),
            "submodel": element.get("submodel"),
            "element_type": element.get("element_type"),
            "title": element.get("title"),
            "description": element.get("description"),
            "heuristic_score": round(score, 6),
            "matching_signals": signals,
        })

    ranked.sort(key=lambda item: (
        -float(item.get("heuristic_score", 0.0)),
        str(item.get("submodel", "")),
        str(item.get("element_type", "")),
        str(item.get("id", "")),
    ))
    return ranked[:CANDIDATE_SHORTLIST_SIZE]


def prioritise_elements_for_context(
    target_elements: list[dict[str, Any]],
    candidate_shortlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Guarantee shortlist candidates are present in the bounded full context."""
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


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
        return list(reader.fieldnames), rows


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    materialised = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialised)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(record)
    return records


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "data"):
        try:
            return plain(value.data())
        except Exception:
            pass
    if hasattr(value, "iso_format"):
        try:
            return value.iso_format()
        except Exception:
            pass
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def pair_key(scenario_id: str, target_unit_id: str) -> str:
    return f"{scenario_id}::{target_unit_id}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_configuration() -> None:
    required = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        raise ValueError("Set environment variables: " + ", ".join(missing))
    for path in (SOURCE_CHANGES_PATH, EXPECTED_CHANGES_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    if MAX_PAIRS is not None and MAX_PAIRS < 1:
        raise ValueError("MAX_PAIRS must be blank/0 or at least 1")
    positive_limits = {
        "MAX_TARGET_ELEMENTS_FETCH": MAX_TARGET_ELEMENTS_FETCH,
        "MAX_TARGET_RELATIONSHIPS_FETCH": MAX_TARGET_RELATIONSHIPS_FETCH,
        "MAX_LLM_ELEMENTS": MAX_LLM_ELEMENTS,
        "MAX_LLM_RELATIONSHIPS": MAX_LLM_RELATIONSHIPS,
        "CANDIDATE_SHORTLIST_SIZE": CANDIDATE_SHORTLIST_SIZE,
        "MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
    }
    invalid_limits = [name for name, value in positive_limits.items() if value < 1]
    if invalid_limits:
        raise ValueError("These limits must be at least 1: " + ", ".join(invalid_limits))
    if CANDIDATE_SHORTLIST_SIZE > MAX_LLM_ELEMENTS:
        raise ValueError("CANDIDATE_SHORTLIST_SIZE cannot exceed MAX_LLM_ELEMENTS")
    if MAX_LLM_ELEMENTS > MAX_TARGET_ELEMENTS_FETCH:
        raise ValueError("MAX_LLM_ELEMENTS cannot exceed MAX_TARGET_ELEMENTS_FETCH")
    if MAX_LLM_RELATIONSHIPS > MAX_TARGET_RELATIONSHIPS_FETCH:
        raise ValueError("MAX_LLM_RELATIONSHIPS cannot exceed MAX_TARGET_RELATIONSHIPS_FETCH")
    if REASONING_EFFORT not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported REASONING_EFFORT")


# Input data


def load_dataset() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    source_headers, source_rows = load_csv(SOURCE_CHANGES_PATH)
    expected_headers, expected_rows = load_csv(EXPECTED_CHANGES_PATH)

    if source_headers != SOURCE_HEADERS:
        raise ValueError(
            f"source_changes.csv must have exactly these columns: {SOURCE_HEADERS}"
        )
    if expected_headers != EXPECTED_HEADERS:
        raise ValueError(
            f"expected_model_changes.csv must have exactly these columns: {EXPECTED_HEADERS}"
        )

    sources: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        scenario_id = row["scenario_id"]
        if not scenario_id:
            raise ValueError("A source change lacks scenario_id")
        if scenario_id in sources:
            raise ValueError(f"Duplicate source scenario: {scenario_id}")
        row["before_state"] = parse_json_object(row["before_json"])
        row["after_state"] = parse_json_object(row["after_json"])
        sources[scenario_id] = row

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in expected_rows:
        scenario_id = row["scenario_id"]
        target_unit_id = row["target_unit_id"]
        key = (scenario_id, target_unit_id)
        if key in seen:
            raise ValueError(f"Duplicate expected pair: {key}")
        seen.add(key)
        if scenario_id not in sources:
            raise ValueError(f"Unknown source scenario in expected data: {scenario_id}")
        if not target_unit_id:
            raise ValueError(f"Expected pair {scenario_id} lacks target_unit_id")
        row["expected_impacted_element_ids"] = sorted(
            str(item) for item in parse_json_list(
                row["expected_impacted_element_ids_json"],
                "expected_impacted_element_ids_json",
            )
        )
        row["expected_model_operations"] = sorted(
            str(item) for item in parse_json_list(
                row["expected_model_operations_json"],
                "expected_model_operations_json",
            )
        )
        pairs.append(row)

    pairs.sort(key=lambda row: (row["scenario_id"], row["target_unit_id"]))
    return sources, pairs


# Neo4j


class GraphRepository:
    def __init__(self, uri: str, username: str, password: str, database: str):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            query,
            parameters_=parameters,
            database_=self.database,
        )
        return [plain(record.data()) for record in records]

    def get_unit(self, unit_id: str) -> dict[str, Any]:
        rows = self.run(
            """
            MATCH (u:OrgUnit {id: $unit_id})
            OPTIONAL MATCH (u)-[:PART_OF]->(parent:OrgUnit)
            RETURN properties(u) AS unit,
                   parent.id AS parent_id,
                   parent.name AS parent_name
            """,
            unit_id=unit_id,
        )
        if not rows:
            raise ValueError(f"Neo4j has no OrgUnit {unit_id!r}")
        return {
            **(rows[0].get("unit") or {}),
            "parent_id": rows[0].get("parent_id"),
            "parent_name": rows[0].get("parent_name"),
        }

    def get_elements(self, element_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not element_ids:
            return {}
        rows = self.run(
            """
            UNWIND $element_ids AS element_id
            OPTIONAL MATCH (e:Element {id: element_id})
            RETURN element_id, properties(e) AS element
            """,
            element_ids=element_ids,
        )
        return {
            row["element_id"]: row["element"]
            for row in rows
            if row.get("element")
        }

    def list_elements(self, unit_id: str) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (e:Element {unit_id: $unit_id})
            WHERE coalesce(e.active, true) = true
            RETURN properties(e) AS element
            ORDER BY e.submodel, e.element_type, e.code, e.id
            LIMIT $limit
            """,
            unit_id=unit_id,
            limit=MAX_TARGET_ELEMENTS_FETCH,
        )
        return [row["element"] for row in rows]

    def list_relationships(self, unit_id: str) -> list[dict[str, Any]]:
        rows = self.run(
            """
            MATCH (source:Element)-[r:MODEL_RELATION]->(target:Element)
            WHERE (source.unit_id = $unit_id OR target.unit_id = $unit_id)
              AND coalesce(r.active, true) = true
            RETURN properties(r) AS relation,
                   source.id AS source_id,
                   source.title AS source_title,
                   source.submodel AS source_submodel,
                   source.element_type AS source_element_type,
                   source.unit_id AS source_unit_id,
                   target.id AS target_id,
                   target.title AS target_title,
                   target.submodel AS target_submodel,
                   target.element_type AS target_element_type,
                   target.unit_id AS target_unit_id
            ORDER BY r.created_at DESC, r.id
            LIMIT $limit
            """,
            unit_id=unit_id,
            limit=MAX_TARGET_RELATIONSHIPS_FETCH,
        )
        return [
            {
                **(row.get("relation") or {}),
                "source_id": row.get("source_id"),
                "source_title": row.get("source_title"),
                "source_submodel": row.get("source_submodel"),
                "source_element_type": row.get("source_element_type"),
                "source_unit_id": row.get("source_unit_id"),
                "target_id": row.get("target_id"),
                "target_title": row.get("target_title"),
                "target_submodel": row.get("target_submodel"),
                "target_element_type": row.get("target_element_type"),
                "target_unit_id": row.get("target_unit_id"),
            }
            for row in rows
        ]


# OpenAI


def build_output_schema() -> dict[str, Any]:
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


OUTPUT_SCHEMA = build_output_schema()


def build_run_signature() -> str:
    payload = {
        "instructions": INSTRUCTIONS,
        "schema": OUTPUT_SCHEMA,
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "max_target_elements_fetch": MAX_TARGET_ELEMENTS_FETCH,
        "max_target_relationships_fetch": MAX_TARGET_RELATIONSHIPS_FETCH,
        "max_elements": MAX_LLM_ELEMENTS,
        "max_relationships": MAX_LLM_RELATIONSHIPS,
        "source_data_sha256": file_sha256(SOURCE_CHANGES_PATH),
        "expected_data_sha256": file_sha256(EXPECTED_CHANGES_PATH),
        "evaluation_contract": "candidate_first_action_primary_model_changes_v4",
        "candidate_shortlist_size": CANDIDATE_SHORTLIST_SIZE,
    }
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


class LLMService:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, max_retries=0)

    def analyse(
        self,
        source_change: dict[str, Any],
        source_element: dict[str, Any],
        source_unit: dict[str, Any],
        target_unit: dict[str, Any],
        target_elements: list[dict[str, Any]],
        target_relationships: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source_payload = {
            **source_change,
            "affected_element": source_element,
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
        elements = [
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
        # Exclude relationships with hidden endpoints and rank candidate context first.
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
                0 if (
                    str(relation.get("source_id", "")) in candidate_ids
                    or str(relation.get("target_id", "")) in candidate_ids
                ) else 1,
                str(relation.get("source_id", "")),
                str(relation.get("target_id", "")),
            ),
        )[:MAX_LLM_RELATIONSHIPS]
        relationships = [
            {
                "source_id": relation.get("source_id"),
                "source_title": relation.get("source_title"),
                "source_submodel": relation.get("source_submodel"),
                "source_element_type": relation.get("source_element_type"),
                "source_unit_id": relation.get("source_unit_id"),
                "target_id": relation.get("target_id"),
                "target_title": relation.get("target_title"),
                "target_submodel": relation.get("target_submodel"),
                "target_element_type": relation.get("target_element_type"),
                "target_unit_id": relation.get("target_unit_id"),
                "kind": relation.get("kind") or relation.get("relationship_kind"),
                "rationale": relation.get("rationale"),
            }
            for relation in prioritised_relationships
        ]
        context = {
            "method": {
                "purpose": "Candidate-first, action-consistent model-change propagation",
                "unit_of_analysis": "one source change and one target unit",
                "organisational_relation": organisational_relation(source_unit, target_unit),
                "scored_contract": {
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
            "target_4em_elements": elements,
            "target_4em_relationships": relationships,
            "allowed_element_types": ELEMENT_TYPES,
            "allowed_relationships": ALL_RELATIONSHIP_KINDS,
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=json_text(context),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "candidate_first_model_change_propagation",
                    "schema": OUTPUT_SCHEMA,
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
            raise RuntimeError(f"OpenAI returned no structured output; status={status!r}")
        analysis = json.loads(output_text)
        metadata = {
            "response_id": getattr(response, "id", ""),
            "response_model": getattr(response, "model", self.model),
            "usage": plain(getattr(response, "usage", {})),
            "status": status,
            "candidate_shortlist": candidate_shortlist,
            "context_element_count": len(elements),
            "context_relationship_count": len(relationships),
        }
        return analysis, metadata


# Action validation and projection


def allowed_relationship_kinds(
    source: dict[str, Any],
    target: dict[str, Any],
) -> list[str]:
    source_model = str(source.get("submodel", ""))
    target_model = str(target.get("submodel", ""))
    if source_model == target_model:
        return INTRA_MODEL_RELATIONSHIPS.get(source_model, ["RELATES_TO"])
    return CROSS_MODEL_RELATIONSHIPS.get((source_model, target_model), ["RELATES_TO"])


def validate_and_project(
    analysis: dict[str, Any],
    target_unit_id: str,
    target_elements: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    actions = analysis.get("proposed_actions")
    if not isinstance(actions, list) or not actions or not all(isinstance(a, dict) for a in actions):
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
        notes.append("Ignored evidence IDs absent from target model: " + ", ".join(invalid_evidence))
    analysis["target_evidence_ids"] = sorted(evidence_ids & target_ids)

    no_change = [a for a in actions if str(a.get("action", "")).upper() == "NO_CHANGE"]
    concrete = [
        a for a in actions
        if str(a.get("action", "")).upper()
        in {"CREATE_ELEMENT", "UPDATE_ELEMENT", "CREATE_RELATION"}
    ]

    # Derive the scored effect from validated actions; the relevance flag is advisory.
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

    impacted: set[str] = set()
    operations: set[str] = set()
    validated: list[dict[str, Any]] = []

    if not relevant:
        operations.add("NO_MODEL_CHANGE")
        validated.append(dict(selected_actions[0]))
    else:
        for raw in selected_actions:
            action = dict(raw)
            kind = str(action.get("action", "")).upper()
            declared_submodel = str(action.get("submodel", ""))
            declared_type = str(action.get("element_type", ""))
            driver = str(action.get("driver", "")).strip()
            if not driver:
                raise ValueError(f"{kind} requires a driver")

            if kind == "CREATE_ELEMENT":
                if declared_submodel not in ELEMENT_TYPES:
                    raise ValueError(f"Invalid submodel: {declared_submodel!r}")
                if declared_type not in ELEMENT_TYPES[declared_submodel]:
                    raise ValueError(
                        f"Invalid element type {declared_type!r} for {declared_submodel!r}"
                    )
                if not str(action.get("title", "")).strip():
                    raise ValueError("CREATE_ELEMENT requires a title")
                operations.add(SUBMODEL_TO_CREATE_OPERATION[declared_submodel])

            elif kind == "UPDATE_ELEMENT":
                element_id = str(action.get("existing_element_id", "")).strip()
                if element_id not in target_ids:
                    raise ValueError(f"UPDATE_ELEMENT uses unknown target ID: {element_id!r}")
                impacted.add(element_id)
                actual_element = target_by_id[element_id]
                actual_submodel = str(actual_element.get("submodel", ""))
                if actual_submodel not in SUBMODEL_TO_UPDATE_OPERATION:
                    raise ValueError(
                        f"Target element {element_id!r} has unsupported submodel {actual_submodel!r}"
                    )
                if declared_submodel != actual_submodel:
                    notes.append(
                        f"Used actual submodel {actual_submodel} for UPDATE_ELEMENT {element_id}; "
                        f"model declared {declared_submodel}"
                    )
                operations.add(SUBMODEL_TO_UPDATE_OPERATION[actual_submodel])

            elif kind == "CREATE_RELATION":
                source_id = str(action.get("relation_source_id", "")).strip()
                target_id = str(action.get("relation_target_id", "")).strip()
                if source_id not in target_ids or target_id not in target_ids:
                    raise ValueError("CREATE_RELATION endpoints must be target element IDs")
                source_element = target_by_id[source_id]
                target_element = target_by_id[target_id]
                if str(source_element.get("unit_id", "")) != target_unit_id:
                    raise ValueError("CREATE_RELATION source endpoint must belong to target unit")
                relationship_kind = str(action.get("relationship_kind", ""))
                if relationship_kind not in allowed_relationship_kinds(source_element, target_element):
                    raise ValueError(
                        f"Invalid relationship {relationship_kind!r} between "
                        f"{source_element.get('submodel')} and {target_element.get('submodel')}"
                    )
                impacted.update({source_id, target_id})
                operations.add("CREATE_RELATION")

            validated.append(action)

    projected = {
        **analysis,
        "relevant": relevant,
        "projection_notes": notes,
        "proposed_actions": validated,
        "projected_impacted_element_ids": sorted(impacted),
        "projected_model_operations": sorted(operations),
    }
    actual = {
        "scenario_id": "",
        "target_unit_id": target_unit_id,
        "impacted_element_ids_json": json.dumps(sorted(impacted), ensure_ascii=False),
        "model_operations_json": json.dumps(sorted(operations), ensure_ascii=False),
    }
    return projected, actual


# Metrics


def set_metrics(expected: set[str], actual: set[str]) -> dict[str, float | int]:
    true_positive = len(expected & actual)
    precision = 1.0 if not expected and not actual else safe_div(true_positive, len(actual))
    recall = 1.0 if not expected and not actual else safe_div(true_positive, len(expected))
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1(precision, recall), 6),
        "exact_match": int(expected == actual),
    }


def evaluate(
    expected_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {
        (row["scenario_id"], row["target_unit_id"]): row
        for row in expected_rows
    }
    actual = {
        (row["scenario_id"], row["target_unit_id"]): row
        for row in actual_rows
    }
    if len(actual) != len(actual_rows):
        raise ValueError("Duplicate actual result pair")
    if set(actual) != set(expected):
        raise ValueError("Actual results do not match the selected expected pairs")

    detail: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for key in sorted(expected):
        exp = expected[key]
        act = actual[key]
        expected_impacted = set(exp["expected_impacted_element_ids"])
        actual_impacted = set(parse_json_list(act["impacted_element_ids_json"], "actual impacted IDs"))
        expected_operations = set(exp["expected_model_operations"])
        actual_operations = set(parse_json_list(act["model_operations_json"], "actual operations"))
        impacted_metrics = set_metrics(expected_impacted, actual_impacted)
        operation_metrics = set_metrics(expected_operations, actual_operations)
        row = {
            "scenario_id": key[0],
            "target_unit_id": key[1],
            "expected_impacted_element_ids_json": json.dumps(sorted(expected_impacted), ensure_ascii=False),
            "actual_impacted_element_ids_json": json.dumps(sorted(actual_impacted), ensure_ascii=False),
            "impacted_element_precision": impacted_metrics["precision"],
            "impacted_element_recall": impacted_metrics["recall"],
            "impacted_element_f1": impacted_metrics["f1"],
            "impacted_element_exact_match": impacted_metrics["exact_match"],
            "expected_model_operations_json": json.dumps(sorted(expected_operations), ensure_ascii=False),
            "actual_model_operations_json": json.dumps(sorted(actual_operations), ensure_ascii=False),
            "operation_precision": operation_metrics["precision"],
            "operation_recall": operation_metrics["recall"],
            "operation_f1": operation_metrics["f1"],
            "operation_exact_match": operation_metrics["exact_match"],
            "model_change_exact_match": int(
                impacted_metrics["exact_match"] and operation_metrics["exact_match"]
            ),
        }
        detail.append(row)
        grouped[key[0]].append(row)

    def average(rows: list[dict[str, Any]], field: str) -> float:
        return round(safe_div(sum(float(row[field]) for row in rows), len(rows)), 6)

    by_scenario: list[dict[str, Any]] = []
    for scenario_id, rows in sorted(grouped.items()):
        by_scenario.append({
            "scenario_id": scenario_id,
            "evaluated_target_pairs": len(rows),
            "impacted_element_macro_precision": average(rows, "impacted_element_precision"),
            "impacted_element_macro_recall": average(rows, "impacted_element_recall"),
            "impacted_element_macro_f1": average(rows, "impacted_element_f1"),
            "impacted_element_exact_match_rate": average(rows, "impacted_element_exact_match"),
            "operation_macro_precision": average(rows, "operation_precision"),
            "operation_macro_recall": average(rows, "operation_recall"),
            "operation_macro_f1": average(rows, "operation_f1"),
            "operation_exact_match_rate": average(rows, "operation_exact_match"),
            "model_change_exact_match_rate": average(rows, "model_change_exact_match"),
        })

    impacted_f1 = average(detail, "impacted_element_f1")
    operation_f1 = average(detail, "operation_f1")
    summary = {
        "evaluated_target_pairs": len(detail),
        "model_change_macro_f1": round((impacted_f1 + operation_f1) / 2, 6),
        "impacted_element_macro_precision": average(detail, "impacted_element_precision"),
        "impacted_element_macro_recall": average(detail, "impacted_element_recall"),
        "impacted_element_macro_f1": impacted_f1,
        "impacted_element_exact_match_rate": average(detail, "impacted_element_exact_match"),
        "operation_macro_precision": average(detail, "operation_precision"),
        "operation_macro_recall": average(detail, "operation_recall"),
        "operation_macro_f1": operation_f1,
        "operation_exact_match_rate": average(detail, "operation_exact_match"),
        "model_change_exact_match_rate": average(detail, "model_change_exact_match"),
    }
    return summary, detail, by_scenario


# Checkpoints and outputs


def load_completed(path: Path, run_signature: str) -> dict[str, dict[str, Any]]:
    if not RESUME_COMPLETED_PAIRS:
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        if record.get("status") != "SUCCESS" or record.get("run_signature") != run_signature:
            continue
        scenario_id = str(record.get("scenario_id", ""))
        target_unit_id = str(record.get("target_unit_id", ""))
        if scenario_id and target_unit_id:
            completed[pair_key(scenario_id, target_unit_id)] = record
    return completed


def compact_jsonl(path: Path, records: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False, sort_keys=True, default=str) + "\n")


def write_actual_results(records: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows = [records[key]["actual_result"] for key in sorted(records)]
    write_csv(RESULTS_DIR / "actual_results.csv", rows, ACTUAL_HEADERS)
    return rows


# Entry point


def main() -> int:
    try:
        validate_configuration()
        sources, all_pairs = load_dataset()
    except Exception as exc:
        print(f"Configuration/data error: {exc}", file=sys.stderr)
        return 2

    pairs = all_pairs[:MAX_PAIRS] if MAX_PAIRS is not None else all_pairs
    selected_keys = {pair_key(row["scenario_id"], row["target_unit_id"]) for row in pairs}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_signature = build_run_signature()
    checkpoint_path = RESULTS_DIR / "model_outputs.jsonl"
    errors_path = RESULTS_DIR / "errors.jsonl"

    try:
        completed = load_completed(checkpoint_path, run_signature)
    except Exception as exc:
        print(f"Checkpoint error: {exc}", file=sys.stderr)
        return 2

    repo: GraphRepository | None = None
    llm = LLMService(OPENAI_API_KEY, OPENAI_MODEL)
    unit_cache: dict[str, dict[str, Any]] = {}
    element_cache: dict[str, list[dict[str, Any]]] = {}
    relationship_cache: dict[str, list[dict[str, Any]]] = {}
    source_element_cache: dict[str, dict[str, Any]] = {}

    try:
        repo = GraphRepository(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE)
        total = len(pairs)
        for index, pair in enumerate(pairs, start=1):
            scenario_id = pair["scenario_id"]
            target_unit_id = pair["target_unit_id"]
            key = pair_key(scenario_id, target_unit_id)
            if key in completed:
                print(f"[{index}/{total}] resume {scenario_id} -> {target_unit_id}")
                continue

            source_change = sources[scenario_id]
            source_unit_id = source_change["source_unit_id"]
            affected_id = source_change["affected_element_id"]

            if source_unit_id not in unit_cache:
                unit_cache[source_unit_id] = repo.get_unit(source_unit_id)
            if target_unit_id not in unit_cache:
                unit_cache[target_unit_id] = repo.get_unit(target_unit_id)
            if target_unit_id not in element_cache:
                element_cache[target_unit_id] = repo.list_elements(target_unit_id)
            if target_unit_id not in relationship_cache:
                relationship_cache[target_unit_id] = repo.list_relationships(target_unit_id)
            if affected_id not in source_element_cache:
                source_element_cache.update(repo.get_elements([affected_id]))
            source_element = source_element_cache.get(affected_id)
            if not source_element:
                raise ValueError(f"Neo4j has no source element {affected_id!r}")

            print(f"[{index}/{total}] {scenario_id} -> {target_unit_id}")
            try:
                raw_analysis, response_metadata = llm.analyse(
                    source_change,
                    source_element,
                    unit_cache[source_unit_id],
                    unit_cache[target_unit_id],
                    element_cache[target_unit_id],
                    relationship_cache[target_unit_id],
                )
                analysis, actual_result = validate_and_project(
                    raw_analysis,
                    target_unit_id,
                    element_cache[target_unit_id],
                )
                actual_result["scenario_id"] = scenario_id
                record = {
                    "status": "SUCCESS",
                    "run_signature": run_signature,
                    "generated_at": utc_now(),
                    "scenario_id": scenario_id,
                    "target_unit_id": target_unit_id,
                    "source_change_id": source_change["change_id"],
                    "analysis": analysis,
                    "actual_result": actual_result,
                    "response_metadata": response_metadata,
                }
                append_jsonl(checkpoint_path, record)
                completed[key] = record
                write_actual_results({k: v for k, v in completed.items() if k in selected_keys})
            except Exception as exc:
                error = {
                    "status": "ERROR",
                    "run_signature": run_signature,
                    "generated_at": utc_now(),
                    "scenario_id": scenario_id,
                    "target_unit_id": target_unit_id,
                    "source_change_id": source_change["change_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(errors_path, error)
                print(f"Pair failed: {scenario_id} -> {target_unit_id}: {exc}", file=sys.stderr)
                if STOP_ON_ERROR:
                    raise
    except Exception as exc:
        print(f"Evaluation run failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if repo is not None:
            repo.close()

    selected_records = {key: completed[key] for key in selected_keys if key in completed}
    missing = sorted(selected_keys - set(selected_records))
    if missing:
        print(f"Missing successful outputs for {len(missing)} pair(s): {missing[:10]}", file=sys.stderr)
        return 1

    compact_jsonl(checkpoint_path, selected_records)
    actual_rows = write_actual_results(selected_records)
    summary, detail, by_scenario = evaluate(pairs, actual_rows)

    detail_headers = [
        "scenario_id", "target_unit_id",
        "expected_impacted_element_ids_json", "actual_impacted_element_ids_json",
        "impacted_element_precision", "impacted_element_recall", "impacted_element_f1",
        "impacted_element_exact_match",
        "expected_model_operations_json", "actual_model_operations_json",
        "operation_precision", "operation_recall", "operation_f1", "operation_exact_match",
        "model_change_exact_match",
    ]
    scenario_headers = [
        "scenario_id", "evaluated_target_pairs",
        "impacted_element_macro_precision", "impacted_element_macro_recall",
        "impacted_element_macro_f1", "impacted_element_exact_match_rate",
        "operation_macro_precision", "operation_macro_recall", "operation_macro_f1",
        "operation_exact_match_rate", "model_change_exact_match_rate",
    ]
    write_csv(RESULTS_DIR / "evaluation_detail.csv", detail, detail_headers)
    write_csv(RESULTS_DIR / "evaluation_by_scenario.csv", by_scenario, scenario_headers)
    (RESULTS_DIR / "evaluation_summary.json").write_text(json_pretty(summary) + "\n", encoding="utf-8")

    manifest = {
        "status": "PARTIAL" if MAX_PAIRS is not None else "COMPLETE",
        "generated_at": utc_now(),
        "run_signature": run_signature,
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "evaluated_target_pairs": len(pairs),
        "source_changes_sha256": file_sha256(SOURCE_CHANGES_PATH),
        "expected_model_changes_sha256": file_sha256(EXPECTED_CHANGES_PATH),
        "summary": summary,
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json_pretty(manifest) + "\n", encoding="utf-8")

    print(json_pretty(summary))
    print(f"Results written to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
