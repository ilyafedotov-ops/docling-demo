import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    smolvlm_picture_description,
)
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import BadRequestError, OpenAI
except ImportError:
    OpenAI = None
    BadRequestError = Exception

if tiktoken is not None:
    try:
        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except KeyError:
        _TOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4o")
else:
    _TOKEN_ENCODING = None

PASS1_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "Pass1Extraction",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "clause",
                                    "tariff_table",
                                    "condition",
                                    "obligation",
                                    "definition",
                                    "cross_reference",
                                    "schedule",
                                    "annex",
                                    "financial_clause",
                                    "termination_clause",
                                    "sla",
                                    "possible_tariff",
                                    "other",
                                ],
                                "default": "other",
                            },
                            "title": {"type": "string"},
                            "quote": {"type": "string"},
                            "page_hint": {"type": ["string", "null"]},
                            "char_span": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "needs_follow_up": {"type": "boolean"},
                            "notes": {"type": "string"},
                            "context_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": [
                            "chunk_id",
                            "type",
                            "quote",
                            "page_hint",
                            "char_span",
                            "confidence",
                            "needs_follow_up",
                        ],
                    },
                    "default": [],
                }
            },
            "required": ["items"],
        },
    },
}

PASS1_INSTRUCTIONS = """Task: Identify legal artifacts in the provided text. Do not infer—only cite content that is explicitly present.
Output JSON only (no commentary) following the supplied schema.
Guidelines:
1. chunk_id must match the provided metadata.
2. type ∈ {clause, tariff_table, condition, obligation, definition, cross_reference, schedule, annex, financial_clause, termination_clause, sla, possible_tariff, other}. Use possible_tariff for borderline detections.
3. char_span values are 0-based [start, end) indices relative to CHUNK TEXT.
4. confidence must be between 0 and 1. Use lower confidence for tentative matches.
5. context_ids should list any related chunk_ids (neighbors, referenced sections). Leave empty if none.
6. needs_follow_up=true when deeper drilldown is recommended (financial terms, nested obligations, unclear tables, etc.).
If nothing is present, return {"items": []}."""

PASS2_INSTRUCTIONS = """You are a senior legal analyst performing a deep extraction on a previously flagged item. Use only the provided materials.
Requirements:
1. Preserve verbatim quotations inside the structured fields.
2. Every scalar must include provenance (chunk_id, page, char_span when possible).
3. Monetary values must include value, currency (if stated), unit (e.g., per month), and derived flag when inferred.
4. Dates must be provided as ISO-8601 ranges when possible; include raw text.
5. Durations should use ISO-8601 format when deducible.
6. For tables, normalise headers and emit each row with provenance and interpretation notes for ranges/footnotes.
7. Generate deterministic IDs: if clause_id/table_id is missing, use "<chunk_id>#<increment>".
8. Populate qa_report.extracted_clause_ids / extracted_tariff_ids with the IDs you produced.
9. If you suspect missing information, add entries to qa_report.needs_review with reason and char_span.
10. name fields must be plain strings (use an empty string when no explicit name is provided).
Respond strictly in JSON conforming to the provided schema."""

_PASS1_SCHEMA_TEXT = json.dumps(PASS1_RESPONSE_FORMAT, ensure_ascii=False, indent=2)

PASS1_SYSTEM = "\n\n".join(
    [
        "You are a legal contracts analyst who extracts verbatim evidence and flags items that require deeper review.",
        "Extraction policy (duplicated to keep responses consistent and enable prompt caching):",
        PASS1_INSTRUCTIONS.strip(),
        "Structured output schema reference (copy 1):\n" + _PASS1_SCHEMA_TEXT,
        "Structured output schema reference (copy 2):\n" + _PASS1_SCHEMA_TEXT,
    ]
)

PASS2_SYSTEM_BASE = (
    "You are a senior legal analyst who converts clauses into structured JSON while preserving verbatim quotations and citations."
)

def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "section"


def _estimate_column_count(doc_items: List[Any]) -> int:
    centers: List[float] = []
    for item in doc_items or []:
        provs = list(getattr(item, "prov", []) or [])
        if not provs:
            continue
        bbox = getattr(provs[0], "bbox", None)
        if not bbox:
            continue
        left = getattr(bbox, "l", None)
        right = getattr(bbox, "r", None)
        if left is None or right is None:
            continue
        centers.append((float(left) + float(right)) / 2.0)
    if len(centers) <= 1:
        return 1
    centers.sort()
    clusters = [[centers[0]]]
    for center in centers[1:]:
        if center - clusters[-1][-1] > 120.0:
            clusters.append([center])
        else:
            clusters[-1].append(center)
    return min(len(clusters), 4)


PROVENANCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chunk_id": {"type": "string"},
        "page": {"type": ["integer", "null"]},
        "char_span": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "required": ["chunk_id"],
}

MONEY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "value": {"type": "number"},
        "currency": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "text": {"type": "string"},
        "derived": {"type": "boolean"},
        "prov": PROVENANCE_SCHEMA,
    },
    "required": ["value", "text", "prov"],
}

DATE_RANGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "start": {"type": ["string", "null"]},
        "end": {"type": ["string", "null"]},
        "text": {"type": "string"},
        "prov": PROVENANCE_SCHEMA,
    },
    "required": ["text", "prov"],
}

CROSS_REF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_text": {"type": "string"},
        "ref_section": {"type": ["string", "null"]},
        "prov": PROVENANCE_SCHEMA,
    },
    "required": ["target_text", "prov"],
}

CLAUSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clause_id": {"type": "string"},
        "name": {"type": ["string", "null"]},
        "verbatim": {"type": "string"},
        "parties": {"type": "array", "items": {"type": "string"}},
        "amounts": {"type": "array", "items": MONEY_SCHEMA},
        "effective": DATE_RANGE_SCHEMA,
        "duration": {"type": ["string", "null"]},
        "termination_triggers": {"type": "array", "items": {"type": "string"}},
        "penalties": {"type": "array", "items": MONEY_SCHEMA},
        "cross_refs": {"type": "array", "items": CROSS_REF_SCHEMA},
        "open_issues": {"type": "array", "items": {"type": "string"}},
        "prov": PROVENANCE_SCHEMA,
    },
    "required": ["clause_id", "verbatim", "prov"],
}

TARIFF_ROW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headers": {"type": "array", "items": {"type": "string"}},
        "values": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "prov": PROVENANCE_SCHEMA,
    },
    "required": ["values", "prov"],
}

TARIFF_TABLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "table_id": {"type": "string"},
        "title": {"type": ["string", "null"]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": TARIFF_ROW_SCHEMA},
        "footnotes": {"type": "array", "items": {"type": "string"}},
        "interpretation_notes": {"type": "array", "items": {"type": "string"}},
        "prov": PROVENANCE_SCHEMA,
    },
    "required": ["table_id", "rows", "prov"],
}

NEEDS_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reason": {"type": "string"},
        "char_span": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "prov": {
            "type": ["object", "null"],
            "properties": PROVENANCE_SCHEMA["properties"],
            "additionalProperties": False,
        },
    },
    "required": ["reason"],
}

QA_REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "extracted_clause_ids": {"type": "array", "items": {"type": "string"}},
        "extracted_tariff_ids": {"type": "array", "items": {"type": "string"}},
        "needs_review": {"type": "array", "items": NEEDS_REVIEW_SCHEMA},
    },
    "required": ["extracted_clause_ids", "extracted_tariff_ids", "needs_review"],
}

PASS2_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clauses": {"type": "array", "items": CLAUSE_SCHEMA},
        "tariffs": {"type": "array", "items": TARIFF_TABLE_SCHEMA},
        "qa_report": QA_REPORT_SCHEMA,
    },
    "required": ["clauses", "tariffs", "qa_report"],
}


PASS2_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "Pass2Extraction",
        "schema": PASS2_RESPONSE_SCHEMA,
    },
}

_PASS2_SCHEMA_TEXT = json.dumps(PASS2_RESPONSE_FORMAT, ensure_ascii=False, indent=2)

PASS2_SYSTEM = "\n\n".join(
    [
        PASS2_SYSTEM_BASE,
        "Deep-extraction policy (duplicated to keep responses consistent and enable prompt caching):",
        PASS2_INSTRUCTIONS.strip(),
        "Structured output schema reference (copy 1):\n" + _PASS2_SCHEMA_TEXT,
        "Structured output schema reference (copy 2):\n" + _PASS2_SCHEMA_TEXT,
    ]
)


def count_tokens(text: str) -> int | None:
    if _TOKEN_ENCODING is None:
        return None
    return len(_TOKEN_ENCODING.encode(text))


class MarkdownSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def extract(pdf_path: Path):
    opts = PdfPipelineOptions(do_table_structure=True)
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = False
    picture_opts = smolvlm_picture_description
    if hasattr(smolvlm_picture_description, "model_copy"):
        picture_opts = smolvlm_picture_description.model_copy(deep=True)
    opts.do_picture_description = True
    opts.picture_description_options = picture_opts
    opts.picture_description_options.prompt = (
        "Describe graphics, charts, and text embedded in images verbatim."
    )
    opts.generate_picture_images = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = converter.convert(str(pdf_path)).document
    return doc, doc.export_to_markdown(), doc.model_dump(mode="json")


def chunk_document(doc, source_id: str):
    chunker = HierarchicalChunker(serializer_provider=MarkdownSerializerProvider())
    chunks = []
    section_counts: Dict[str, int] = {}
    for idx, chunk in enumerate(chunker.chunk(doc)):
        meta = getattr(chunk, "meta", None)
        headings = list(getattr(meta, "headings", []) or [])
        doc_items = list(getattr(meta, "doc_items", []) or [])
        label = None
        page_no = None
        bbox = None
        if doc_items:
            first_item = doc_items[0]
            label_attr = getattr(first_item, "label", None)
            label = label_attr.name if hasattr(label_attr, "name") else label_attr
            prov = list(getattr(first_item, "prov", []) or [])
            if prov:
                first_prov = prov[0]
                page_no = getattr(first_prov, "page_no", None)
                bbox_obj = getattr(first_prov, "bbox", None)
                if bbox_obj is not None:
                    bbox = {
                        "l": float(getattr(bbox_obj, "l", 0.0)),
                        "t": float(getattr(bbox_obj, "t", 0.0)),
                        "r": float(getattr(bbox_obj, "r", 0.0)),
                        "b": float(getattr(bbox_obj, "b", 0.0)),
                    }
        level = len(headings)
        if headings:
            slug = "-".join(_slugify(h) for h in headings if _slugify(h))
            section_key = slug or "section"
        else:
            section_key = "section"
        section_counts.setdefault(section_key, 0)
        count = section_counts[section_key]
        section_counts[section_key] += 1
        section_id = f"{source_id}-{section_key}-{count}"
        column_count = _estimate_column_count(doc_items)
        chunks.append(
            {
                "chunk_id": f"{source_id}-chunk-{idx}",
                "text": chunk.text,
                "context": chunker.contextualize(chunk),
                "headings": headings,
                "label": label,
                "page_hint": page_no,
                "bbox": bbox,
                "section_id": section_id,
                "level": level,
                "column_count": column_count,
                "neighbors": [],
            }
        )
    for idx, chunk in enumerate(chunks):
        neighbor_ids: List[str] = []
        if idx > 0:
            neighbor_ids.append(chunks[idx - 1]["chunk_id"])
        if idx < len(chunks) - 1:
            neighbor_ids.append(chunks[idx + 1]["chunk_id"])
        chunk["neighbors"] = neighbor_ids
    return chunks


def _neighbor_context(
    chunk: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    limit: int = 240,
) -> str:
    snippets: List[str] = []
    for neighbor_id in chunk.get("neighbors", []):
        neighbor = chunk_map.get(neighbor_id)
        if not neighbor:
            continue
        snippet = neighbor["text"].strip().replace("\n", " ")
        if len(snippet) > limit:
            snippet = snippet[:limit].rstrip() + "..."
        snippets.append(f"{neighbor_id}: {snippet}")
    return "\n".join(snippets) if snippets else "None"


def _build_pass1_prompt(
    chunk: Dict[str, Any], chunk_map: Dict[str, Dict[str, Any]], question: str | None
) -> str:
    meta_parts = [
        f"Chunk ID: {chunk['chunk_id']}",
        f"Page hint: {chunk.get('page_hint')}",
        f"Label: {chunk.get('label')}",
        f"Section: {chunk.get('section_id')} (level={chunk.get('level')}, columns={chunk.get('column_count')})",
    ]
    if chunk.get("headings"):
        meta_parts.append(f"Headings: {' > '.join(chunk['headings'])}")
    meta = "\n".join(part for part in meta_parts if part and "None" not in str(part))
    neighbors_str = _neighbor_context(chunk, chunk_map)
    focus_line = f"Focus: {question.strip()}\n\n" if question else ""
    return (
        f"{focus_line}"
        f"{meta}\nNeighbors:\n{neighbors_str}\n\n"
        "Reminder: char_span indices reference CHUNK TEXT below. "
        "Only use context_ids from the provided neighbor list or other explicit references.\n\n"
        f"CHUNK TEXT:\n{chunk['text']}"
    )


def _build_pass2_prompt(
    chunk: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    pass1_item: Dict[str, Any],
    question: str | None,
) -> str:
    meta_parts = [
        f"Chunk ID: {chunk['chunk_id']}",
        f"Page hint: {chunk.get('page_hint')}",
        f"Label: {chunk.get('label')}",
        f"Section: {chunk.get('section_id')} (level={chunk.get('level')}, columns={chunk.get('column_count')})",
    ]
    if chunk.get("headings"):
        meta_parts.append(f"Headings: {' > '.join(chunk['headings'])}")
    meta = "\n".join(part for part in meta_parts if part and "None" not in str(part))
    neighbors_str = _neighbor_context(chunk, chunk_map)
    focus_line = f"Additional focus: {question.strip()}\n\n" if question else ""
    neighbor_text_block = ""
    if neighbors_str and neighbors_str != "None":
        neighbor_text_block = "\n\nNEIGHBOR SNIPPETS:\n" + neighbors_str
    return (
        f"{focus_line}"
        f"{meta}"
        f"{neighbor_text_block}\n\n"
        "PASS-1 ITEM:\n"
        f"{json.dumps(pass1_item, ensure_ascii=False, indent=2)}\n\n"
        f"CHUNK TEXT:\n{chunk['text']}\n"
    )

LLM_CACHE_ENABLED = os.getenv("ENABLE_LLM_CACHE", "1").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
LLM_CACHE_PREFIX = os.getenv("LLM_CACHE_PREFIX", "docling-demo")


def _chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format: Dict[str, Any],
    prompt_cache_key: str | None = None,
) -> Tuple[str, Dict[str, Any] | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            prompt_cache_key=prompt_cache_key,
        )
    except BadRequestError as exc:
        if LLM_CACHE_ENABLED and "cache" in str(exc).lower():
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                prompt_cache_key=prompt_cache_key,
            )
        else:
            raise
    message = response.choices[0].message
    content = message.content or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    usage = response.usage.model_dump() if response.usage else None
    return content, usage


def _update_usage(
    usage_totals: Dict[str, int], usage: Dict[str, Any] | None
) -> None:
    if not usage:
        return
    key_aliases = {
        "input_tokens": ["input_tokens", "prompt_tokens"],
        "output_tokens": ["output_tokens", "completion_tokens"],
        "total_tokens": ["total_tokens"],
    }
    for agg_key, aliases in key_aliases.items():
        for alias in aliases:
            if alias in usage:
                usage_totals[agg_key] = usage_totals.get(agg_key, 0) + int(usage[alias])
                break
    detail = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if isinstance(detail, dict) and "cached_tokens" in detail:
        usage_totals["cached_tokens"] = usage_totals.get("cached_tokens", 0) + int(
            detail["cached_tokens"] or 0
        )


def run_two_pass_extraction(
    chunks: List[Dict[str, Any]],
    question: str,
    pass1_output_path: Path,
    precision_output_path: Path,
    model: str = "gpt-4o-mini",
    max_chunks: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai`.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    cache_prefix = f"{LLM_CACHE_PREFIX}:{model}"
    pass1_cache_key = f"{cache_prefix}:pass1:v1" if LLM_CACHE_ENABLED else None
    pass2_cache_key = f"{cache_prefix}:pass2:v1" if LLM_CACHE_ENABLED else None
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    pass1_records: List[Dict[str, Any]] = []
    pass1_usage_totals: Dict[str, int] = {}
    pass2_usage_totals: Dict[str, int] = {}

    processing_chunks = chunks[:max_chunks] if max_chunks else chunks

    for chunk in processing_chunks:
        prompt = _build_pass1_prompt(chunk, chunk_map, question)
        content, usage_dump = _chat_json(
            client,
            model,
            PASS1_SYSTEM,
            prompt,
            PASS1_RESPONSE_FORMAT,
            prompt_cache_key=pass1_cache_key,
        )
        _update_usage(pass1_usage_totals, usage_dump)
        data = json.loads(content)
        for item in data.get("items", []):
            if not item.get("chunk_id"):
                item["chunk_id"] = chunk["chunk_id"]
            if item.get("page_hint") is None:
                item["page_hint"] = chunk.get("page_hint")
            item.setdefault("notes", "")
            item.setdefault("type", "other")
            item.setdefault("context_ids", [])
            pass1_records.append(item)

    pass1_output_path.write_text(
        json.dumps({"items": pass1_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    targeted_types = {
        "clause",
        "tariff_table",
        "condition",
        "obligation",
        "financial_clause",
        "termination_clause",
        "sla",
    }
    flagged_items = [
        (idx, item)
    for idx, item in enumerate(pass1_records)
        if item.get("needs_follow_up")
        or item.get("type") in targeted_types
        or question
    ]

    pass2_items: List[Dict[str, Any]] = []
    for idx, item in flagged_items:
        chunk = chunk_map.get(item["chunk_id"])
        if not chunk:
            continue
        prompt = _build_pass2_prompt(chunk, chunk_map, item, question)
        content, usage_dump = _chat_json(
            client,
            model,
            PASS2_SYSTEM,
            prompt,
            PASS2_RESPONSE_FORMAT,
            prompt_cache_key=pass2_cache_key,
        )
        _update_usage(pass2_usage_totals, usage_dump)
        data = json.loads(content)
        qa_report = data.setdefault("qa_report", {})
        qa_report.setdefault("extracted_clause_ids", [])
        qa_report.setdefault("extracted_tariff_ids", [])
        qa_report.setdefault("needs_review", [])
        clauses = data.get("clauses", [])
        tariffs = data.get("tariffs", [])
        page_hint = chunk.get("page_hint")
        try:
            page_number = int(page_hint) if page_hint is not None else None
        except (TypeError, ValueError):
            page_number = None
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            clause.setdefault("prov", {})
            clause["prov"].setdefault("chunk_id", chunk["chunk_id"])
            clause["prov"].setdefault("page", page_number)
        for table in tariffs:
            if not isinstance(table, dict):
                continue
            table.setdefault("prov", {})
            table["prov"].setdefault("chunk_id", chunk["chunk_id"])
            table["prov"].setdefault("page", page_number)
        data["source_chunk_id"] = chunk["chunk_id"]
        data["source_pass1_index"] = idx
        pass2_items.append(data)

    precision_output_path.write_text(
        json.dumps({"items": pass2_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return pass1_usage_totals, pass2_usage_totals


def build_document_graph(
    segments_data: List[Dict[str, Any]],
    pass1_data: Dict[str, Any],
    pass2_data: Dict[str, Any],
    output_path: Path,
) -> Path:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    chunk_to_section: Dict[str, str] = {}
    for segment in segments_data:
        section_id = segment.get("section_id") or f"{segment['chunk_id']}-section"
        chunk_to_section[segment["chunk_id"]] = section_id
        if section_id not in nodes:
            nodes[section_id] = {
                "id": section_id,
                "type": "section",
                "level": segment.get("level"),
                "page_hint": segment.get("page_hint"),
                "label": segment.get("label"),
                "headings": segment.get("headings"),
            }
        chunk_id = segment["chunk_id"]
        if chunk_id not in nodes:
            nodes[chunk_id] = {
                "id": chunk_id,
                "type": "chunk",
                "section_id": section_id,
                "page_hint": segment.get("page_hint"),
                "label": segment.get("label"),
                "headings": segment.get("headings"),
            }

    clause_nodes: Dict[str, Dict[str, Any]] = {}
    tariff_nodes: Dict[str, Dict[str, Any]] = {}
    text_ref_nodes: Dict[str, str] = {}

    for extraction in pass2_data.get("items", []):
        source_chunk = extraction.get("source_chunk_id")
        section_id = chunk_to_section.get(source_chunk)
        for clause in extraction.get("clauses", []):
            clause_id = clause.get("clause_id") or f"{source_chunk}#clause"
            clause_nodes[clause_id] = {
                "id": clause_id,
                "type": "clause",
                "name": clause.get("name"),
                "parties": clause.get("parties"),
                "prov": clause.get("prov"),
                "amounts": clause.get("amounts"),
                "duration": clause.get("duration"),
            }
            if clause_id not in nodes:
                nodes[clause_id] = clause_nodes[clause_id]
            if section_id:
                edges.append(
                    {
                        "source": clause_id,
                        "target": section_id,
                        "type": "within_section",
                    }
                )
            for cross in clause.get("cross_refs", []) or []:
                target_section = cross.get("ref_section")
                target_text = cross.get("target_text", "").strip()
                target_id = None
                if target_section and target_section in nodes:
                    target_id = target_section
                else:
                    normalized = target_text.lower()
                    for possible_id, node in clause_nodes.items():
                        name = (node.get("name") or "").lower()
                        if name and name == normalized:
                            target_id = possible_id
                            break
                    if not target_id and target_text:
                        target_id = text_ref_nodes.setdefault(
                            normalized or f"text-{len(text_ref_nodes)}",
                            f"textref::{len(text_ref_nodes)}",
                        )
                        if target_id not in nodes:
                            nodes[target_id] = {
                                "id": target_id,
                                "type": "reference_text",
                                "value": target_text,
                            }
                if target_id:
                    edges.append(
                        {
                            "source": clause_id,
                            "target": target_id,
                            "type": "references",
                            "prov": cross.get("prov"),
                        }
                    )

        for table in extraction.get("tariffs", []) or []:
            table_id = table.get("table_id") or f"{source_chunk}#tariff"
            tariff_nodes[table_id] = {
                "id": table_id,
                "type": "tariff_table",
                "title": table.get("title"),
                "columns": table.get("columns"),
                "prov": table.get("prov"),
            }
            if table_id not in nodes:
                nodes[table_id] = tariff_nodes[table_id]
            if section_id:
                edges.append(
                    {
                        "source": table_id,
                        "target": section_id,
                        "type": "within_section",
                    }
                )

    name_map: Dict[str, List[str]] = {}
    for clause_id, node in clause_nodes.items():
        raw_name = node.get("name")
        if isinstance(raw_name, list):
            raw_name = " ".join(str(part) for part in raw_name)
        if raw_name is None:
            raw_name = ""
        name = str(raw_name).lower()
        if name.strip():
            name_map.setdefault(name, []).append(clause_id)
    for ids in name_map.values():
        if len(ids) > 1:
            ids_sorted = sorted(ids)
            for i in range(len(ids_sorted)):
                for j in range(i + 1, len(ids_sorted)):
                    edges.append(
                        {
                            "source": ids_sorted[i],
                            "target": ids_sorted[j],
                            "type": "possible_duplicate",
                        }
                    )
                    amounts_i = {
                        (amt.get("value"), amt.get("currency"))
                        for amt in (clause_nodes[ids_sorted[i]].get("amounts") or [])
                    }
                    amounts_j = {
                        (amt.get("value"), amt.get("currency"))
                        for amt in (clause_nodes[ids_sorted[j]].get("amounts") or [])
                    }
                    if amounts_i and amounts_j and amounts_i != amounts_j:
                        edges.append(
                            {
                                "source": ids_sorted[i],
                                "target": ids_sorted[j],
                                "type": "conflict",
                                "metadata": {
                                    "reason": "Different monetary values for similarly named clauses"
                                },
                            }
                        )

    tariff_title_map: Dict[str, List[str]] = {}
    for table_id, node in tariff_nodes.items():
        raw_title = node.get("title")
        if isinstance(raw_title, list):
            raw_title = " ".join(str(part) for part in raw_title)
        if raw_title is None:
            raw_title = ""
        title = str(raw_title).lower()
        if title.strip():
            tariff_title_map.setdefault(title, []).append(table_id)
    for ids in tariff_title_map.values():
        if len(ids) > 1:
            ids_sorted = sorted(ids)
            for i in range(len(ids_sorted)):
                for j in range(i + 1, len(ids_sorted)):
                    edges.append(
                        {
                            "source": ids_sorted[i],
                            "target": ids_sorted[j],
                            "type": "possible_duplicate_tariff",
                        }
                    )

    detection_nodes: Dict[str, Dict[str, Any]] = {}
    for item in pass1_data.get("items", []):
        chunk_id = item.get("chunk_id")
        section_id = chunk_to_section.get(chunk_id)
        if section_id:
            edges.append(
                {
                    "source": chunk_id,
                    "target": section_id,
                    "type": "chunk_in_section",
                }
            )
        item_type = item.get("type", "other")
        det_id = f"detection::{item_type}"
        if det_id not in detection_nodes:
            detection_nodes[det_id] = {
                "id": det_id,
                "type": "detection_category",
                "category": item_type,
            }
            nodes[det_id] = detection_nodes[det_id]
        edges.append(
            {
                "source": chunk_id,
                "target": det_id,
                "type": "pass1_detection",
                "metadata": {
                    "confidence": item.get("confidence"),
                    "needs_follow_up": item.get("needs_follow_up"),
                    "char_span": item.get("char_span"),
                },
            }
        )

    graph = {
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Insurance PDF extraction with Docling")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--json-out", type=Path, help="Path to write Docling JSON dump")
    parser.add_argument("--markdown-out", type=Path, help="Path to write Docling Markdown")
    parser.add_argument("--chunks-out", type=Path, help="Path to write hierarchical chunks JSON")
    parser.add_argument(
        "--precision-out", type=Path, help="Path to write structured extraction JSON"
    )
    parser.add_argument(
        "--pass1-out", type=Path, help="Path to write pass-1 identification JSON"
    )
    parser.add_argument(
        "--question",
        default="Extract all policy information with verbatim citations.",
        help="Prompt guiding the LLM deep extraction (default runs full stack).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model to use for extraction (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-pass1-chunks",
        type=int,
        help="Optional limit for number of chunks processed in pass 1 (for debugging)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM passes (only Docling/VLM outputs).",
    )
    args = parser.parse_args()

    if not args.skip_llm and load_dotenv:
        load_dotenv()

    start_time = time.perf_counter()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.pdf.stem

    doc, markdown, doc_json = extract(args.pdf)

    markdown_path = args.markdown_out or output_dir / f"{stem}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    json_path = args.json_out or output_dir / f"{stem}.docling.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8")

    chunks = chunk_document(doc, stem)
    chunks_path = args.chunks_out or output_dir / f"{stem}.chunks.json"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    segments_path = output_dir / f"{stem}.segments.json"
    segments_data = []
    for chunk in chunks:
        segments_data.append(
            {
                "chunk_id": chunk["chunk_id"],
                "section_id": chunk.get("section_id"),
                "level": chunk.get("level"),
                "page_hint": chunk.get("page_hint"),
                "bbox": chunk.get("bbox"),
                "headings": chunk.get("headings"),
                "label": chunk.get("label"),
                "column_count": chunk.get("column_count"),
                "neighbors": chunk.get("neighbors", []),
                "text": chunk["text"],
            }
        )
    segments_path.write_text(
        json.dumps({"segments": segments_data}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pass1_path = None
    precision_path = None
    graph_path = None
    pass1_usage: Dict[str, int] = {}
    pass2_usage: Dict[str, int] = {}
    precision_tokens = None
    if not args.skip_llm:
        pass1_path_override = args.pass1_out or output_dir / f"{stem}.pass1.json"
        precision_path_override = (
            args.precision_out or output_dir / f"{stem}.precision.json"
        )
        pass1_path_override.parent.mkdir(parents=True, exist_ok=True)
        precision_path_override.parent.mkdir(parents=True, exist_ok=True)

        pass1_usage, pass2_usage = run_two_pass_extraction(
            chunks,
            args.question,
            pass1_path_override,
            precision_path_override,
            model=args.llm_model,
            max_chunks=args.max_pass1_chunks,
        )

        pass1_path = pass1_path_override

        precision_path = precision_path_override
        precision_tokens = count_tokens(precision_path.read_text(encoding="utf-8"))
        graph_path = build_document_graph(
            segments_data,
            json.loads(pass1_path.read_text(encoding="utf-8")),
            json.loads(precision_path.read_text(encoding="utf-8")),
            output_dir / f"{stem}.graph.json",
        )

    duration = time.perf_counter() - start_time
    markdown_tokens = count_tokens(markdown)
    chunks_token_total = (
        sum(count_tokens(item["text"]) or 0 for item in chunks)
        if markdown_tokens is not None
        else None
    )

    saved_paths = [
        ("markdown", markdown_path),
        ("docling_json", json_path),
        ("chunks_json", chunks_path),
        ("segments_json", segments_path),
    ]
    if pass1_path:
        saved_paths.append(("pass1_json", pass1_path))
    if precision_path:
        saved_paths.append(("precision_json", precision_path))
    if graph_path:
        saved_paths.append(("graph_json", graph_path))

    for label, path in saved_paths:
        print(f"{label}: {path.resolve()}")

    print(f"execution_seconds: {duration:.2f}")
    if markdown_tokens is not None:
        print(f"markdown_tokens_cl100k: {markdown_tokens}")
        print(f"chunk_tokens_cl100k: {chunks_token_total}")
        if precision_tokens is not None:
            print(f"precision_tokens_cl100k: {precision_tokens}")
    else:
        print("token_stats: install tiktoken for token estimates")

    if pass1_usage:
        print(
            "pass1_usage_tokens: "
            f"in={pass1_usage.get('input_tokens', 0)}, "
            f"out={pass1_usage.get('output_tokens', 0)}, "
            f"total={pass1_usage.get('total_tokens', 0)}, "
            f"cached={pass1_usage.get('cached_tokens', 0)}"
        )
    if pass2_usage:
        print(
            "pass2_usage_tokens: "
            f"in={pass2_usage.get('input_tokens', 0)}, "
            f"out={pass2_usage.get('output_tokens', 0)}, "
            f"total={pass2_usage.get('total_tokens', 0)}, "
            f"cached={pass2_usage.get('cached_tokens', 0)}"
        )


if __name__ == "__main__":
    main()
