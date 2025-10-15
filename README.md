# Docling Insurance Policy Extraction Demo

End-to-end reference pipeline that turns any PDF (insurance policies are just the running example) into rich structured artifacts you can use for exploration, compliance review, or downstream analytics. The script is built around Docling’s PDF pipeline, a hierarchical chunker, and an optional two-pass LLM workflow that captures detailed clause and tariff information.

## Features at a Glance

- High-fidelity Docling conversion with Markdown, JSON, picture captions, and table structure.
- Hierarchical chunking with section metadata, neighbor context, per-chunk column estimates, and serialized tables.
- Optional multi-pass LLM extraction with caching, verbatim provenance, clause/tariff schemas, and token usage telemetry.
- Document graph output that stitches sections, chunks, detections, clauses, tariffs, and potential conflicts.
- Sensible defaults plus CLI hooks for custom prompts, output paths, and quick experimentation.

## Requirements

- Python 3.10+ recommended.
- `pip install -r requirements.txt`
  - `openai` enables the LLM passes.
  - `tiktoken` adds token counting in the CLI summary.
  - `python-dotenv` (optional) loads `.env` files for local experimentation.
- OpenAI API key (`OPENAI_API_KEY`) if you plan to run the LLM passes.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Docling only (no LLM passes):
python demo.py samples/sample_policy.pdf --skip-llm

# Run the full two-pass extraction:
OPENAI_API_KEY=sk-... python demo.py samples/sample_policy.pdf
```

Outputs are written to `outputs/` by default. Specify alternative paths with the CLI flags documented below.

## Command Reference

```bash
python demo.py <pdf-path> [options]
```

| Option | Description |
| ------ | ----------- |
| `--json-out PATH` | Override the Docling JSON export path. |
| `--markdown-out PATH` | Override the Markdown export path. |
| `--chunks-out PATH` | Override the hierarchical chunks JSON path. |
| `--pass1-out PATH` | Override the pass 1 identification JSON path. |
| `--precision-out PATH` | Override the pass 2 deep extraction JSON path. |
| `--question TEXT` | Prompt that steers the LLM drill-down (defaults to a comprehensive extraction request). |
| `--llm-model NAME` | OpenAI model name (default `gpt-4o-mini`). |
| `--max-pass1-chunks N` | Limit how many chunks the LLM processes—great for fast iterations. |
| `--skip-llm` | Skip both LLM passes; emit only Docling/VLM artifacts. |

All CLI options can be combined; path overrides are handy when you want to keep multiple experiments side by side.

## Output Files

| File | Purpose |
| ---- | ------- |
| `<stem>.md` | Markdown rendition of the PDF with image captions produced by Docling + smolVLM. |
| `<stem>.docling.json` | Full Docling structured export (pages, blocks, tables, figures). |
| `<stem>.chunks.json` | Hierarchical chunker output with headings, neighbor IDs, contextual snippets, and table markdown. |
| `<stem>.segments.json` | Pass 0 snapshot capturing section IDs, levels, coordinates, and raw text. |
| `<stem>.pass1.json` | LLM pass 1 detections (clauses, tariffs, obligations, definitions, etc.) with confidences and char spans. |
| `<stem>.precision.json` | LLM pass 2 drill-down results: structured clauses, tariff tables, QA report. |
| `<stem>.graph.json` | Reconciled document graph linking sections, chunks, detections, clauses, tariffs, duplicates, and conflicts. |

Paths use the PDF filename stem unless you override them with CLI flags.

## Pipeline Walkthrough

1. **Docling Conversion**  
   `demo.py` configures Docling’s PDF pipeline in high-accuracy mode, enables table structure detection, and asks smolVLM to describe images. The Markdown and JSON exports form the base artifacts.
2. **Hierarchical Chunking**  
   The Docling document flows into a `HierarchicalChunker`, producing chunks attached to heading hierarchy, neighbor context, page hints, table markdown, and inferred column counts.
3. **Segmentation Snapshot**  
   Chunk metadata is normalized into `segments.json`, giving you a tidy “pass 0” inventory of the document with text for traditional NLP pipelines or search indexing.
4. **Pass 1 – Detection (optional)**  
   Each chunk is scored by an OpenAI model (default `gpt-4o-mini`) using a strict JSON schema that identifies legal artifacts and flags items that deserve deeper review. Token usage is accumulated for telemetry.
5. **Pass 2 – Precision Extraction (optional)**  
   Flagged chunks (or all chunks when you provide a question) receive a detailed prompt. The model emits structured clauses and tariff tables with verbatim evidence, normalised monetary values, ISO date ranges, QA flags, and provenance.
6. **Graph Assembly**  
   Pass 0–2 outputs are woven into a graph that surfaces duplicate clauses/tariffs, conflicting monetary values, detection categories, and references back to the original sections.

This architecture makes it straightforward to inspect policies interactively, push data into a graph database, or run audits that require traceable citations.

## Environment Variables

- `OPENAI_API_KEY` – required for pass 1/2 if you are not skipping the LLM stages.
- `ENABLE_LLM_CACHE` – defaults to `1`; set to `0` to disable OpenAI response caching.
- `LLM_CACHE_PREFIX` – string prefix used when caching prompts/responses (`docling-demo` by default).

You can drop these into a `.env` file (loaded automatically when `python-dotenv` is installed) or export them in your shell.

## Tips for Effective Experiments

- Start with `--skip-llm` to verify Docling outputs and chunk structure before incurring LLM cost.
- Narrow focus with `--question "Focus on dental coverage clauses."` or similar prompts; the second pass will honour your intent and still emit full provenance.
- Use `--max-pass1-chunks` when you want to debug the pipeline on just a few sections; the CLI still writes all static Docling artifacts.
- Track cost via the printed token telemetry for each pass. If `tiktoken` is installed, you also receive markdown/chunk token counts for the base artifacts.

## Repository Layout

```
.
├── demo.py            # Main entry point for the pipeline described above
├── requirements.txt   # Python dependencies
├── samples/           # Example policy PDFs for quick trials
└── outputs/           # Default output directory (created on demand)
```

Feel free to customize `demo.py` for your own ontology, different LLM vendors, or alternative downstream targets—the pipeline is modular enough to drop in new serializers or schemas as needed.

## Known Issues

- Throughput is currently slower than we would like, especially on large documents; performance work is planned.
- LLM caching occasionally misbehaves, so you may see duplicate calls or cache misses—we will address this in a future update.
