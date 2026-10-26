# Ambactos Brain

Memory engine for agents.

Documents go in. The user decides the ontology. Passages are labelled.
The engine reconstructs, per entity and per period, with cited facts,
disagreements, and « not established ». Each decided entity gets one page.
Search is composite and carries provenance. Synchronisation follows a
regime per family of documents.

A hosted version is operated by Ambactos.

Code comments and messages are in French.

## What it does

1. **Ingest documents** into a tenant deposit (files, mail, Drive-style
   connectors). Each piece is registered, routed by family, and read.
2. **Decide an ontology** with the user: entity types, entities and their
   keys, relations, concepts, families of pieces. The engine proposes;
   every answer is a dated decision. Nothing is erased. No search is
   deployed without at least a v1 ontology.
3. **Label passages** with those decided entities at ingest time. The
   filter later comes from this structure, never from the meaning of the
   question.
4. **Guided reconstruction** walks pieces entity by entity through time
   and writes facts, rules, disagreements and « not established », each
   with the exact source line. The census of units comes from the
   ontology, not from a model.
5. **One Markdown page per entity** (compiled state, rules, open points,
   disagreements with every version and its lines, chronology, identities,
   cited pieces, related entities), plus an index, in a diffable Git
   deposit that belongs to the tenant.
6. **Composite search with provenance**: reconstructed objects by sense,
   then raw passages filtered by the entities of those objects, inside a
   fixed budget. A piece that is authoritative for an entity is cited
   before derived pieces.
7. **Synchronisation regimes** per family: *continuous* (each new piece
   rebuilds the touched unit), *on demand* (first question on an entity
   triggers reconstruction, cached, invalidated by sync), *never* (the
   piece speaks for itself — statutes, filings, leases, statements).

## Architecture (six steps)

| Step | Artefact | Owner |
| --- | --- | --- |
| 1. Sources | piece register, routing by family | the tenant |
| 2. Ontology | `ontologie.vN.json`, decision log, open questions | the user; the agent proposes |
| 3. Data model | types, relations, concepts, treatments, regimes | decided, rarely |
| 4. Constitution | labels, reconstructed units, pages, index, embeddings | the engine, under a cap |
| 5. Access | per-tenant API behind a token; the same context for every agent | the engine |
| 6. Feedback and sync | pass log: detected, routed, invalidated, open questions | the engine; reviewed by the user |

A fact an agent learns while acting goes back through steps 1 and 2. It
never enters constitution directly.

The design is recorded in
[`docs/adr/0027-agent-memory-ontology-model-then-search.md`](docs/adr/0027-agent-memory-ontology-model-then-search.md).
Behavioural proof (what a test must make visible) is in
[`docs/preuve-comportementale.md`](docs/preuve-comportementale.md).

## Installation

Python 3.10+. From a clone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH="scripts:src"
export BANC_VAR="${BANC_VAR:-./var}"
```

`BANC_VAR` is the working directory of the bench (gold, corpus, caches).
It defaults to `./var`.

## Synthetic bench: GreenLoop and MOREAU

The public bench ships two synthetic dossiers only:

- **GreenLoop** — English business dossier (`gl-q…` questions).
- **MOREAU** — French legal dossier (`MOR-Q…` questions).

Place each dossier under `$BANC_VAR/dossiers/<name>/{gold,documents,aire}`.
Run the offline tests:

```bash
PYTHONPATH=scripts:src python -m pytest tests -q
```

The bench remains the judge of new retrieval lines: equal budget, human-
validated gold, gold never ingested. Call caps are explicit.

## HTTP API and MCP

The v0 HTTP API (`brain.api_brain`) exposes ontology, entities, pages,
composite search, on-demand reconstruction and synchronisation, one tenant
at a time, behind a bearer token.

```bash
PYTHONPATH=scripts:src python -m banc_realite.run_api_brain
```

The thin MCP server (`brain.mcp_api`) calls that API. Raw HTTP binds
loopback only.

```bash
PYTHONPATH=scripts:src python -m banc_realite.run_mcp_api --tenant demo --transport stdio
```

## Published contracts

### Pieces and citations

A fact carries the piece filename and the line numbers of the quote. A
citation is the exact span that supports the fact, not a model paraphrase.
Disagreements keep every version and its lines. « not established » is a
first-class status, not a missing value.

### `BRAIN_TENANTS`

JSON object mapping a tenant name to paths and caps, for example:

```json
{
  "demo": {
    "racine": "./var/dossiers/greenloop",
    "reconstitution": "./var/dossiers/greenloop/reconstitution",
    "ontologie": "./var/dossiers/greenloop/ontologie/ontologie.v1.json",
    "dossier_banc": "greenloop",
    "plafonds": {
      "embeddings_questions": 10000,
      "embeddings_passages": 3000,
      "reconstitution_appels": 80
    },
    "regimes": {"statuts": "jamais"}
  }
}
```

The access token is the environment variable
`BRAIN_TENANT_<TENANT>_JETON` (non-alphanumerics become `_`).
`BRAIN_TENANTS` is the register; it is not a secret.

Optional keys: `depot_brut`, `registre_pieces`, `pieces`, `dsn`,
`reconstitution_a_la_demande`. A tenant with a raw deposit and no
`plafonds.embeddings_passages` is rejected (« plafond non décidé »).

### Raw deposit spec v1

A connector writes a local tree the engine reads without a network:

- files under the deposit root, stable identifiers (no `..`, no `/` in ids);
- Gmail-style messages as JSON plus attachments under
  `gmail/pieces-jointes/<message-id>/`;
- a cursor `depot-brut.curseur.json`.

`synchroniser` detects added, changed and removed pieces, routes them,
invalidates only what changed, and appends a pass to the journal.

## Paper

The measurements that fixed this order of work (ontology and data model
first, then search and reconstruction) are in the paper **Reporting
reality** (English), public deposit
[`rendre-compte-de-la-realite`](https://github.com/guillaume-hestia-projekt/rendre-compte-de-la-realite).

## License

Copyright 2026 Hestia Projekt. Licensed under the GNU Affero General
Public License v3.0 only (AGPL-3.0-only), see `LICENSE`. If you run a
modified version as a network service, the AGPL requires you to offer
its source to its users. A hosted version and commercial licenses
(use without the AGPL obligations) are offered by Ambactos: contact
guillaume@ambactos.ai.
