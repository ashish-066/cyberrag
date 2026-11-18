# whentrue-cti MCP server guide

`whentrue-cti` exposes the CyberRAG verification service to any MCP client (LM Studio with a local Qwen, Claude Desktop, MCP Inspector, other agents). Verdicts come from the evidence corpus and an NLI model, never from the client LLM.

## 1. One-time setup

```bash
pip install -r requirements.lock
python scripts/download_models.py     # the only step that needs the network
python scripts/build_manifest.py      # only after intentionally editing files in data/
```

After this the server runs fully offline (`CTI_OFFLINE` defaults to true).

## 2. Run it

| Transport | Command | Use for |
|---|---|---|
| stdio | `python -m mcp_server.server --transport stdio` | LM Studio, Claude Desktop |
| Streamable HTTP | `python -m mcp_server.server --transport http --port 8765` (endpoint `http://127.0.0.1:8765/mcp`) | MCP Inspector, remote agents, tests |

* HTTP binds to `127.0.0.1`. Binding any other host **requires** `CTI_MCP_TOKEN`; clients then send `Authorization: Bearer <token>`.
* Models load in a background thread, so the server answers `initialize` and `tools/list` immediately. The first tool call waits until loading finishes (about 5 s warm, up to ~90 s when the embedding cache has to be rebuilt).
* Logs are JSON on stderr (stdout carries the protocol).

## 3. Tools

| Tool | Purpose | Key inputs |
|---|---|---|
| `search_evidence` | Ranked, de-duplicated passages with provenance. No verdict. | `query`, `top_k` (1-50), `date_from`, `date_to`, `publishers`, `session_id` |
| `verify_claim` | Verify one claim. Returns `status`, `support_tier`, supporting / contradicting evidence, conflict analysis. | `subject`, `relation` (USES, TARGETS, EXPLOITS, USES_TECHNIQUE), `object`, `as_of_date`, `session_id` |
| `verify_question` | Plan a question into sub-claims and verify each. Which/what questions return a graph lookup. | `question`, `session_id` |
| `get_entity` | Canonical name, MITRE id, aliases, mentions, first/last seen, graph neighbours. | `name` |
| `get_timeline` | Dated evidence for a subject-object pair, marked supports / contradicts / neutral. | `subject`, `object`, `relation` |
| `ingest_report_preview` | Step 1: parse a report, preview entities and relations. **Nothing is indexed.** | `text` (up to 20 MB), `title`, `publisher`, `pub_date`, `session_id` |
| `ingest_report_confirm` | Step 2: index the previewed report into that session only. | `confirm_token`, `session_id` |
| `export_stix` | STIX 2.1 bundle (validated) from earlier `verify_claim` results. | `verification_ids` |

Resources: `cti://corpus/stats`, `cti://report/{report_id}`, `cti://trace/{verification_id}`. Prompt: `verify_cti_statement`.

### Reading a result
* `status`: `SUPPORTED`, `CONTRADICTED`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`.
* `support_tier`: `NONE`, `SINGLE_SOURCE`, `CORROBORATED` (needs two or more independent corpus publishers), `CONFLICTING`. One publisher can never be `CORROBORATED`. User uploads never count as independent corroboration.
* `hedged_only`: true when every supporting passage says suspected / alleged / unconfirmed. Treat the support as tentative.
* `conflict`: `type` (for example `TEMPORAL_SUPERSEDENCE`), `direction` (`contradiction_later` / `support_later` / `concurrent`), `delta_days`.
* `as_of_date_applied`: `publication_date_filter` in Stage 1 (it filters by publication date, not event date). Undated evidence is excluded when a date filter is set and reported in `warnings`.
* Errors: `isError=true` with `{code, message, hint}`; codes are `INVALID_INPUT`, `UNKNOWN_RELATION`, `ENTITY_NOT_FOUND`, `VERIFIER_UNAVAILABLE`, `UPLOAD_REJECTED`, `NO_EXTRACTABLE_TEXT`, `TOKEN_EXPIRED`, `RATE_LIMITED`, `INTERNAL`.

## 4. Security model

* Report text appears only in fields named `untrusted_text`; every result carries `content_notice` reminding the client to treat it as data. Tool and parameter descriptions are static strings in code.
* No file-system or shell tools. Ingest takes text only; the client reads files.
* Writes only through preview then confirm. Tokens expire after 10 minutes and are bound to the session; uploads live in that session only and expire after 2 hours.
* Responses are capped at 200 KB (truncation is flagged), passages at 2,000 characters, `top_k` at 50. Over HTTP: 60 calls per minute per session.
* Every call is logged (tool, argument hash, session, duration, outcome) without report text.

## 5. Clients

### LM Studio (local Qwen, offline)
1. Load a tool-capable model (for example Qwen2.5-3B/7B-Instruct or Qwen3-4B).
2. Open **Program → Install → Edit mcp.json** and add the server from `docs/mcp_clients/lm_studio_mcp.json` (replace `<REPO>` with the absolute path of this repository).
3. Ask: *"Does APT28 use Zebrocy? Verify with the tool and cite evidence ids."* The model should call `verify_claim` and quote `evidence_id` values. Turn Wi-Fi off and repeat: it still works.

Small models call tools unevenly. If a 3B model skips the tool, select the `verify_cti_statement` prompt or use a larger model; the verdict never depends on the client model.

### Claude Desktop
Add `docs/mcp_clients/claude_desktop_config.json` to your Claude Desktop config (`mcpServers` section) and restart the app.

### VS Code (Copilot agent mode)
Save `docs/mcp_clients/vscode_mcp.json` as `.vscode/mcp.json` in your workspace (note the top-level key is `servers`, not `mcpServers`), replace `<REPO>`, then use Copilot Chat in Agent mode and confirm the 8 tools are listed.

### Antigravity
Agent panel, MCP servers menu, Manage, "View raw config", then merge `docs/mcp_clients/antigravity_mcp_config.json` into `mcp_config.json` (menu names can differ between versions). VS Code and Antigravity are untested; if a client fails to connect, check its MCP log (this server logs JSON to stderr). Hosted models send tool results, including evidence text, to the provider, so "fully offline" only applies with a local model.

### MCP Inspector
```bash
npx @modelcontextprotocol/inspector python -m mcp_server.server --transport stdio
# or against HTTP: start the HTTP server, then connect to http://127.0.0.1:8765/mcp
```
All eight tools, three resources and the prompt should be listed with valid schemas.

### Python SDK
```python
from mcp.client import Client
from mcp_server.server import ServiceHolder, build_server
# in-memory: Client(build_server(ServiceHolder(service=svc), settings))
```
See `tests/mcp/test_mcp_contract.py` for complete examples.

## 6. Configuration

All settings are `CTI_<NAME>` environment variables or keys in `config.yaml` (see `config.example.yaml`). Bad values print a readable error and exit with code 2. Useful ones: `CTI_MODE` (`demo|eval|prod`), `CTI_LLM_URL`, `CTI_LLM_MODEL`, `CTI_TOP_K`, `CTI_SESSION_TTL_SECS`, `CTI_MCP_TOKEN`, `CTI_MCP_PORT`, `CTI_LOAD_DEMO_OVERLAY` (adds one record labelled `DEMO – SYNTHETIC`; refused when `CTI_MODE=eval`).

## 7. Stage 2 compatibility
`as_of_date`, `support_tier` and `conflict.type` already exist in the schemas and hold Stage 1 values (publication-date filter, publisher-count tier, resolver taxonomy). Stage 2 replaces how they are computed without changing the interface.
