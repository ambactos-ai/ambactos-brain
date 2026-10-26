# ADR 0027 — Agent memory: stable ontology and data model first, then search and reconstruction, one shared workflow

| | |
|---|---|
| **Status** | Proposed — 2026-10-14. Decision of the project owner, Hestia Projekt (13–14/10/2026): every agent must have a memory, and for that it must live in an environment with a stable ontology and a structured data model; once that is in place, a RAG and a reconstruction can be deployed; the workflow must be structured, because it will cross every agent; a synchronisation workflow to update continuously is still missing. |
| **Track** | brain#158 (Brain from a tenant Drive, PR #155/#159), brain#165 (this ADR, PR #167), brain#166 (continuous synchronisation, parent issue). Product track: ambactos#381 (GTP Brain), ambactos#382 (product ADR), ambactos#384 (ingest → Brain), ambactos#388 (federated search). |
| **Related decisions** | ADR 0021 (fine reconstruction), ADR 0022 (coarse-to-fine claiming), ADR 0023 (comparative bench), **ADR 0024 (porting the bench chain to production: the bench remains the judge, budget read as a share of the corpus)**, ADR 0025 and 0026 (situations); shared policy in `AGENTS.md` § canonical coordination (Brain = the place of memory, Memory Kernel = Workspace–Brain seam), repeated in the generated block of this repository's `AGENTS.md`; interim paper of 10–11/10/2026 (public deposit `rendre-compte-de-la-realite`). |
| **Implementation** | `scripts/banc_realite/` (Drive reading, routing, tenant ontology, labelling, guided reconstruction, pages, composite search, bench), `src/brain/api_brain/` (v0 API per tenant); tenant data lives outside the repository (`BANC_VAR`, default `./var`). |

## Context

An agent that acts for a company must remember what the company knows: its
pieces, its commitments, its figures, its contradictions. Since September
2026 the « reporting reality » bench has measured more than twenty ways of
giving that memory to a model, on three dossiers (GreenLoop, MOREAU, and a
third production dossier not published here), with gold written or validated
by a human, never calling a model at query time. Three results hold on all
three dossiers:

1. **RAG alone does not surface contradictions** (14/28 disputes on
   GreenLoop, 0/8 on the third dossier): it returns a single side.
2. **Reconstruction at ingest is worthless alone and changes everything in
   composition**: alone, 15/34 on the third dossier; composed with filtered
   RAG, 24/34, the best line everywhere.
3. **The filter does the work, and it must come from the structure of the
   content, never from the question.** A perfect filter holds 113/141;
   filters deduced from the question lose against no filter.

October measurements added what structure contributes when it is **decided
by the user** and placed **before** reconstruction:

| Line (third dossier, 34 questions, 12 000 characters) | All holds | Contradictions | Citations |
|---|---|---|---|
| Naive RAG | 3 | 0/8 | 1/6 |
| Composite RAG filtered by ontology v1, no reconstruction | 5 | 0/8 | 1/6 |
| Reconstruction alone (census by model) | 15 | 3/8 | 0/6 |
| Reconstruction **guided by the decided ontology**, alone | 20 | 7/8 | 0/6 |
| Objects by sense then filtered RAG (reference) | 24 | 6/8 | 2/6 |
| Guided reconstruction, objects by sense then filtered RAG | **25** | 6/8 | 2/6 |
| … + reserved share for pieces in plain reading, by sense | 25 | 6/8 | 2/6 |
| … + reserved share filtered by the entities of the question | 25 | 6/8 | 2/6 |

Reading: the decided ontology (38 entities, two mergers settled in
conversation) lifts reconstruction-alone from 15 to 20; ontology alone,
without reconstruction, does nothing (5). Missing citations are not a
recall hole but a **provenance** hole: the right value is in the context,
carried by a derived piece, while gold requires the line of the piece that
is authoritative. Sense and keywords do not distinguish that piece (one
entity key occurs in 117 passages); only the entity label placed at ingest
does.

These findings meet, by experience, what practitioners report in
production: vector search alone loses exact terms and multi-factor
questions; a model-generated knowledge graph degrades (duplicates,
uncontrolled entities, context that explodes at three relations); a
file-based « second brain » fits a person, not a company (governance,
rights); what holds is a structure decided at ingest, a filter by that
structure, then matching by sense. The difference we keep: the structure is
proposed by the content and **decided by the user**, and the **versions of
the same fact** (disagreements, not established) are kept, not frozen into
one value per variable.

## Decision

### D1. Every agent has a memory, and it is shared

A Bureau agent has no private memory. It reads and writes in its tenant's
Brain through a single interface (Brain API, `rechercher`, `entites`,
`ontologie`). Two agents of the same tenant see the same context for the
same question. A fact an agent learns does not enter memory except through
the workflow below.

### D2. Ontology and data model come before any search

- **The tenant ontology** (families of pieces, entity types, entities and
  their keys, relations, concepts) is **proposed from the content** and
  **decided in conversation** with the user (mergers, splits, missing
  types, pieces that are authoritative alone). Each answer is a dated
  decision; nothing is erased.
- It is **frozen in use**: one version labels, filters and guides; it
  **evolves between versions** by decision. The core (types, relations,
  families) changes rarely; instances (entities) often.
- **The data model** is the stable core: entity types, relations, concepts,
  and for each family of pieces its **treatment** (reconstruction per
  entity, plain reading, OCR, register only) and its **regime** (D4).
- No search is deployed for a tenant without a decided ontology at least
  in v1.

### D3. Search composes, it does not guess

- The filter comes **from the structure**: ontology keys found literally in
  the question, labels placed at ingest, families and treatments. **No
  filter is deduced from the sense of the question.**
- Context is **composed**: objects reconstructed by sense, then raw
  passages filtered by the entities of those objects, inside a budget
  fixed in advance. On the bench that budget is 12 000 characters so that
  lines compare equally; in production the default budget of a tenant is
  read as a share of its corpus, as ADR 0024 (decision 5) requires, and
  the context report always says which share of the dossier it poured. One
  form of search per family of pieces, composed; not a router by question.
- A question about an entity whose piece **is authoritative** receives the
  line of that piece before any derived piece (provenance rule, to be
  measured: brain#166 or a dedicated issue).
- A single embedding model; what varies is the embedded object: passage,
  reconstructed object, entity page.

### D4. Reconstruction has a regime, decided by family

Reconstruction (reread pieces entity by entity through time, write facts,
rules, disagreements and not established with the exact line) is **guided
by the ontology**: the census of units comes from decided entities, not
from a model. Its regime is decided at routing, by family:

| Regime | When | For whom |
|---|---|---|
| **continuous** | each new piece rebuilds the touched unit; disagreements are formalised before any question | dossiers where contradiction is the deliverable (litigation, counsel, audit) |
| **on demand** | the first question on an entity triggers its reconstruction, under a cap; cached; invalidated by synchronisation | ordinary company memory |
| **never** | the piece is authoritative alone; plain reading, labelled, cited | statutes, company filings, leases, statements |

Held fragility: « surface disputes at search time » has not worked from
raw passages (0/8). On demand means *reconstruct on the first question*,
not *do not reconstruct*.

### D5. Outputs are readable and belong to the tenant

One **Markdown page per decided entity** (compiled state, rules, not
established, disagreements with every version and its lines, chronology by
period, identities, cited pieces, related entities), an index, all in a
tenant Git deposit, diffable, exportable. Each fact carries its piece and
its lines; no sentence drafted by a model at this step.

### D6. The shared workflow, in six steps

| Step | Artefact | Owner | Measured |
|---|---|---|---|
| 1. Sources | piece register, routing by family | the tenant, in the Bureau | Drive only |
| 2. Ontology | `ontologie.vN.json`, decision log, open questions | the user; the agent proposes | yes |
| 3. Data model | types, relations, concepts, treatments, regimes | decision, rarely | partially |
| 4. Constitution | labels, reconstructed units, pages, index, embeddings | Brain, under a cap | yes |
| 5. Access | per-tenant API behind a token; the same context for every agent | Brain | yes |
| 6. Feedback and synchronisation | pass log: detected, routed, invalidated, open questions | Brain; reviewed by the user | **no, brain#166** |

Step 6 is the missing workflow: **detect** (new, changed, removed piece;
live data such as a stock or an amended contract), **route** (family,
treatment; unknown entity = ontology question, never an error),
**invalidate** in cascade only what changes (unit → page → vectors, by
packet fingerprint), **trace** each pass. A fact an agent learns while
acting goes back through 1 and 2, never straight into 4.

### D7. The bench remains the authority

Every new line is measured at equal budget against the reference, on
validated gold, without touching the judge or the gold; a negative result
is kept and recorded. Gold is used only to measure, never to ingest. Call
and embedding caps are decided by the project owner and recorded.

## Consequences

- A tenant without a decided ontology has no memory agent: the ontology
  conversation is the first step of every go-live (ambactos#384).
- The hard-coded dossier register (`DOSSIERS`) must give way to per-tenant
  configuration (`BRAIN_TENANTS`: root, reconstruction, ontology, passage
  store) before any exposure outside the bench.
- Workflow steps (reading, routing, ontology, reconstruction, pages,
  embeddings) are **ported into production code** (`src/brain`), with
  journal and cap, as ADR 0024 decided for search; the scripts in
  `scripts/banc_realite/` remain the measuring instrument, replayed
  independently, and are never modified to improve their own score
  (ADR 0021 § 7, ADR 0024 decision 6). The v0 API exposes only access
  (step 5) and decisions (step 2).
- Continuous synchronisation (step 6) is a track of its own (brain#166),
  a prerequisite for live data (CRM, mail, stocks).
- Still to measure: the provenance rule (authoritative piece first), the
  « on demand » regime (cost and latency of the first question), user
  intent and end-to-end latency.

## Alternatives set aside

- **Memory per agent** (memories extracted by the model, self-edited
  blocks): measured in ADR 0023 against mem0 and Letta; does not render
  dossiers with tables and sources that contradict each other.
- **Knowledge graph extracted by the model**: uncontrolled entities,
  duplicates, no time and no disagreement; our graph is the decided
  ontology plus `cite` relations.
- **File-based second brain, per agent** (GBrain / LLM Wiki style): a
  good output form (Markdown, Git, one page per entity, night cycle),
  which we take up in D5; not a model of company memory (single user, no
  decided ontology, no gold).
- **Filter deduced from the question**: loses everywhere on the bench;
  forbidden.
