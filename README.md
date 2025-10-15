# Insurance PDF Extraction Demo

Docling-first pipeline that produces detailed Markdown, JSON, and hierarchical chunk artifacts for insurance policies. All outputs land under `outputs/` unless you override the destinations.

1. Install deps: `pip install -r requirements.txt`.
2. (Optional) copy `.env.example` to `.env` and set `OPENAI_API_KEY`, or export it in your shell when you want structured LLM extraction.
3. Extract a policy (full stack by default): `python insurance_demo.py "samples/your_policy.pdf"`. The run emits:
   - `outputs/<name>.md` (Markdown with VLM picture descriptions)
   - `outputs/<name>.docling.json` (raw Docling document)
   - `outputs/<name>.chunks.json` (HierarchicalChunker output with Markdown tables)
   - `outputs/<name>.segments.json` (Pass-0 segmentation: section ids, levels, neighbors, verbatim text)
   - CLI summary reports total runtime and cl100k token counts (requires `tiktoken`, already in requirements).
4. Configure extraction behaviour:
   - Override the LLM prompt: `python insurance_demo.py "samples/your_policy.pdf" --question "Focus on dental coverage clauses."`
     * Pass 1 (`outputs/<name>.pass1.json`) flags clauses/tables per chunk.
     * Pass 2 (`outputs/<name>.precision.json`) performs deep extraction of the flagged items with verbatim quotes, structured ontology, and citations.
     * Pass 3 builds `outputs/<name>.graph.json`, a document graph linking sections, clauses, tariffs, cross references, and duplicate/conflict signals.
   - `--max-pass1-chunks N` limits how many chunks run through the LLM passes (handy for quick experiments).
   - `--skip-llm` keeps only the Docling/VLM outputs (skips Pass 1/2/3).
   - Path overrides still work (e.g., `--markdown-out custom/path.md`).

The two-hop approach uses Docling chunks as context, identifies candidate clauses, then reprocesses each target with a focused prompt to capture parties, amounts, dates, termination triggers, table rows, and issues for legal/compliance review. OpenAI usage totals for each pass and aggregate token counts are printed so you can tune prompts and model choice.

## Pipeline Overview

- **Pass 0 – Segmentation**
  - Docling + VLM conversion feeds a hierarchical chunker.
  - Each chunk receives `section_id`, heading level, column count, neighbors, and verbatim text.
  - Output: `outputs/<name>.segments.json`.

- **Pass 1 – Identification (coarse)**
  - Lightweight prompt only classifies artifacts (no interpretation) per chunk.
  - Returns `type`, `quote`, `char_span`, `confidence`, `context_ids`, `needs_follow_up`.
  - Output: `outputs/<name>.pass1.json`.

- **Pass 2 – Drilldown (fine)**
  - For flagged items (`needs_follow_up` or high-value types) the LLM extracts full clause/tariff structures using a strict JSON schema.
  - Includes verbatim evidence, monetary/date normalization, cross-references, table rows, and a QA report.
  - Output: `outputs/<name>.precision.json`.

- **Pass 3 – Document Graph**
  - Reconciles Pass 0–2 data into `outputs/<name>.graph.json` linking sections, chunks, clauses, tariffs, detection categories, duplicate/conflict hints, and references.

## Prompting & Tuning Tips

- Prompts enforce verbatim text, ISO-formatted dates/durations, and provenance (`chunk_id`, `page`, `char_span`).
- QA tail lists extracted clause/tariff IDs and records follow-up issues under `qa_report.needs_review`.
- `--llm-model` lets you choose a heavier model (e.g., `gpt-4o`) for dense legalese; `--max-pass1-chunks` helps debug small subsets.
- CLI telemetry shows Docling runtime plus per-pass token usage so you can monitor cost.

## Storage Model Snapshot

| File | Purpose |
|------|---------|
| `*.segments.json` | Pass 0 segmentation (sections, neighbors, headings, bbox). |
| `*.pass1.json` | Classified artifacts with confidences and spans. |
| `*.precision.json` | Deep extraction (clauses, tariffs, QA report). |
| `*.graph.json` | Reconciliation graph (sections ↔ clauses ↔ tariffs, conflicts, references). |

These outputs form a minimal audit trail for compliance, search, or downstream graph analytics.
