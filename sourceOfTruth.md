Good — that's the full original paper. It's very clear: the paper only describes global search (community summary map-reduce). Local search with entity-first vector retrieval is described in the implementation, not this paper. Let me verify the official implementation's local search spec now.I now have primary sources for all three systems. Let me write the strict verification.

---

## Strict Verification: Retrieval Flows

Three systems, sourced directly from: the Edge et al. (2024) paper + official `microsoft.github.io/graphrag` docs; the Guo et al. EMNLP 2025 paper (full PDF); and the CircleMind DeepWiki codebase analysis + GitHub README.

---

## System 1 — Microsoft GraphRAG

### Indexing-Time Structures Built

Entity descriptions are embedded for entity-based retrieval in local search. Community summaries are embedded for global search retrieval. Text unit content is also embedded for basic semantic search. Embeddings enable the semantic similarity searches that serve as entry points into the knowledge graph during query time.

So three distinct embedding namespaces exist at rest:
- Entity description embeddings → vector store
- Community report embeddings → vector store
- Text unit embeddings → vector store
- Graph topology (entities as nodes, relationships as edges, Leiden community hierarchy) → graph store (Parquet/LanceDB/Neo4j)

---

### Query-Time Flow: Global Search

**Source: Edge et al. (2024) paper, Section 3.1.6 — primary source, full text verified above.**

Given a user query, community summaries are randomly shuffled and divided into chunks of pre-specified token size. Intermediate answers are generated in parallel — the LLM is also asked to generate a score between 0–100 indicating how helpful the generated answer is in answering the target question. Answers with score 0 are filtered out. Intermediate community answers are sorted in descending order of helpfulness score and iteratively added into a new context window until the token limit is reached. This final context is used to generate the global answer returned to the user.

**First retrieval signal (Global):** None. There is no vector similarity or entity lookup. **All community summaries at the chosen level are sent to the LLM unconditionally.** The "retrieval" is a fan-out over all community reports, not a targeted lookup. The LLM scores each one for helpfulness and filters post-hoc.

**Chunk selection (Global):** Chunks are **not retrieved at all** in global search. The unit of context is community summaries, not source text chunks. The paper is explicit: "these community summaries provide global descriptions and insights over the corpus."

---

### Query-Time Flow: Local Search

**Source: Official `microsoft.github.io/graphrag/query/local_search/` — verified above.**

Given a user query and optionally conversation history, the local search method identifies a set of entities from the knowledge graph that are semantically related to the user input. These entities serve as access points into the knowledge graph, enabling the extraction of further relevant details such as connected entities, relationships, entity covariates, and community reports. Additionally, it also extracts relevant text chunks from the raw input documents that are associated with the identified entities. These candidate data sources are then prioritized and filtered to fit within a single context window of pre-defined size.

The official dataflow diagram (verified above) shows:

```
User Query
    │
    └─[1]─→ Entity Description Embedding (vector search against entity store)
                │
                └─[2]─→ Extracted Entities
                           ├── Entity-Text Unit Mapping      → Candidate Text Units
                           ├── Entity-Report Mapping         → Candidate Community Reports
                           ├── Entity-Entity Relationships   → Candidate Entities
                           └── Entity-Covariate Mappings     → Candidate Covariates
                                        │
                                        └─[3]─→ Ranking + Filtering → Context Window → LLM Response
```

When a user submits a query, it is embedded using the openai-text-embedding-3-small model. This embedding is used to search for semantically similar entities in the vector store. The vector store returns the top-k (default 10) semantically relevant entities based on descriptions. Retrieved entities are mapped to the entities in the original entities DataFrame and converted into Entity objects for further use.

**Verified pipeline for local search:**

| Step | Action | Signal |
|---|---|---|
| 1 | Embed query | Query embedding |
| 2 | ANN search against **entity description embeddings** | Vector search — entity-first |
| 3 | Retrieve top-k entities (default 10) | Entity set |
| 4 | Follow entity→text unit mappings | Graph lookup (not vector) |
| 5 | Follow entity→relationship mappings | Graph lookup |
| 6 | Follow entity→community report mappings | Graph lookup |
| 7 | Rank + filter all candidates by token budget | Priority scoring |
| 8 | Assemble context, call LLM | Generation |

**First retrieval hop: vector search on entity description embeddings.** Not on raw chunk embeddings. Chunks are never directly vector-searched in local search — they are reached via graph traversal from entity seeds.

---

### Critical Distinction: Paper vs. Implementation

The original Edge et al. (2024) paper **does not describe local search at all.** The paper only formalizes global search (community summary map-reduce). Local search was added in the implementation and is documented in the official codebase, not the paper. Many blog posts incorrectly attribute both modes to the paper.

You'll even learn a search method that the paper didn't mention at all (Local Search).

---

### Summary Table — Microsoft GraphRAG

| Dimension | Global Search | Local Search |
|---|---|---|
| First retrieval signal | None (all communities broadcast) | Vector search on entity embeddings |
| Vector search used? | Not for retrieval — LLM scoring post-hoc | Yes — step 1, entity embeddings only |
| Entity extraction at query time? | No | Yes — ANN top-k entities used as graph seeds |
| How chunks are retrieved | Not retrieved | Via entity→text unit graph mapping |
| Role of entities | Not used | Graph seeds, access points to all other data |
| Role of relationships | Not used directly | Retrieved via entity→entity mapping |
| Role of communities | Central unit of context (all sent to LLM) | Fetched via entity→report mapping (subset) |
| Chunk selection | Not applicable | Token-budget priority ranking of all candidates |

**Confidence: High.** Based on the primary paper (full text verified) and the official implementation docs (dataflow diagram sourced directly).

---

## System 2 — LightRAG (arXiv:2410.05779, EMNLP 2025)

### Indexing-Time Structures Built

**Source: Guo et al. EMNLP 2025, Section 3.1 — full PDF verified above.**

LightRAG employs a LLM-powered profiling function P(·) to generate a text key-value pair (K, V) for each entity node and relation edge. Each index key is a word or short phrase that enables efficient retrieval, while the corresponding value is a text paragraph summarizing relevant snippets from external data to aid in text generation. Entities use their names as the sole index key, whereas relations may have multiple keys derived from LLM enhancements that include global themes from connected entities.

Structures at rest:
- Knowledge graph: entity nodes + relationship edges (deduplicated)
- Vector index over **entity-level keys** (entity names/descriptions) — low-level
- Vector index over **relation-level global theme keys** — high-level
- Original chunk text (retained for context assembly)

No community detection. No pre-computed community summaries. No separate community store.

---

### Query-Time Retrieval Flow

**Source: Guo et al. EMNLP 2025, Section 3.2 — full PDF verified above.**

For a given query q, the retrieval algorithm of LightRAG begins by extracting both local query keywords k(l) and global query keywords k(g). The algorithm uses an efficient vector database to match local query keywords with candidate entities and global query keywords with relations linked to global keys. To enhance the query with higher-order relatedness, LightRAG further gathers neighboring nodes within the local subgraphs of the retrieved graph elements. This involves the set {vi | vi ∈ V ∧ (vi ∈ Nv ∨ vi ∈ Ne)}, where Nv and Ne represent the one-hop neighboring nodes of the retrieved nodes v and edges e, respectively.

For each query, we first utilize the LLM to generate relevant keywords. Similar to current RAG systems, our retrieval mechanism relies on vector-based search. However, instead of retrieving chunks as in conventional RAG, we concentrate on retrieving entities and relationships. This approach markedly reduces retrieval overhead compared to the community-based traversal method used in GraphRAG.

**Verified pipeline:**

| Step | Action | Signal |
|---|---|---|
| 1 | LLM call: extract local keywords k(l) AND global keywords k(g) from query | LLM keyword extraction |
| 2 | Vector search: match k(l) against entity name/description index | Low-level vector search |
| 3 | Vector search: match k(g) against relation global theme key index | High-level vector search |
| 4 | Graph expansion: collect 1-hop neighbors of retrieved entities and relations | Graph traversal |
| 5 | Assemble context: entity descriptions + relationship descriptions + original chunk excerpts | Concatenation |
| 6 | LLM generates answer | Generation |

**First retrieval hop: LLM keyword extraction, followed immediately by dual-level vector search.** The LLM runs first to produce keywords, then those keywords are matched via vector similarity — this is not a standard query embedding approach. Critically, **there is no direct chunk vector search** at any point. Chunks appear in the final context only because they are the "values" associated with retrieved entity/relation keys.

---

### Key Design Points Verified from the Paper

**Dual-level = two parallel vector searches on two different indexes, not a two-stage sequential filter.** Low-level targets entity nodes; high-level targets relation edges (which carry global theme keys generated by LLM at indexing time). Both run in parallel.

**Graph traversal role:** One-hop neighbor expansion after vector matches. This is not deep multi-hop traversal — it is a shallow expansion to enrich the retrieved subgraph with adjacent entities and edges.

**No community detection, confirmed explicitly:** The paper explicitly contrasts LightRAG with GraphRAG: "This approach markedly reduces retrieval overhead compared to the community-based traversal method used in GraphRAG." There is no Leiden step, no community summaries.

---

### Summary Table — LightRAG

| Dimension | LightRAG |
|---|---|
| First retrieval signal | LLM keyword extraction from query |
| Vector search used? | Yes — dual parallel searches (entity index + relation index) |
| Query entities extracted? | Yes — via LLM keyword extraction (not embedding-based entity matching) |
| How keywords are used | As queries into two separate vector indexes |
| How chunks are retrieved | Not directly — chunks are the "value" paired with entity/relation keys; never ANN-searched independently |
| Role of entities | Primary index targets; retrieved by low-level keyword match |
| Role of relationships | Secondary index targets; retrieved by high-level keyword match |
| Role of communities | **Not present** |
| Graph traversal role | 1-hop neighbor expansion after vector retrieval |
| Final context assembly | Entity descriptions + relation descriptions + associated chunk excerpts |

**Confidence: High.** Based on the full EMNLP 2025 paper (complete PDF verified).

---

### Common Blog Misconception About LightRAG

Many blog posts describe LightRAG as doing "low-level retrieval for specific queries and high-level retrieval for abstract queries" as if the system routes between them. **The paper does not describe routing.** Section 3.2 describes both retrieval levels running together for any given query. The paper says "To accommodate diverse query types, the LightRAG employs two distinct retrieval strategies within the dual-level retrieval paradigm" — but does not specify that only one runs per query. Whether the implementation applies both always or routes is **not explicitly specified in the paper**; the formal description implies both run in parallel.

---

## System 3 — Fast GraphRAG (CircleMind AI, 2024)

### Indexing-Time Structures Built

**Source: CircleMind DeepWiki codebase analysis + GitHub README — verified above.**

Fast GraphRAG has no published academic paper. All findings below are sourced from the official GitHub repository, README, and DeepWiki codebase documentation.

fast-graphrag constructs an explicit knowledge graph that captures entities and their relationships, enabling more interpretable and accurate retrieval through graph traversal algorithms. Key Innovation: fast-graphrag uses Personalized PageRank to explore the knowledge graph, starting from entities mentioned in the query and propagating relevance scores through relationships to chunks.

Structures at rest:
- Knowledge graph: entity nodes + relationship edges (stored as igraph sparse matrix)
- HNSW vector index over entity description embeddings (`HNSWVectorStorage`)
- Chunk store (original text chunks, stored separately)
- Entities-to-relationships sparse matrix (for propagation)
- Relationships-to-chunks sparse matrix (for propagation)

No community detection. No community summaries.

---

### Query-Time Retrieval Flow

**Source: CircleMind DeepWiki query pipeline page — full text verified above.**

The query pipeline transforms a natural language query into a structured response by: (1) Extracting key entities and concepts from the query; (2) Retrieving relevant entities, relationships, and chunks from storage using multi-stage scoring; (3) Truncating the context to fit within token limits; (4) Generating a final response using an LLM with the assembled context.

**Phase 1 — Query Entity Extraction:**

The extract_entities_from_query() method prompts the LLM to identify two types of entities: Named Entities (specific proper nouns mentioned, e.g., ["Scrooge"]) and Generic Entities (general concepts or types, e.g., ["person", "character"]). The LLM interaction uses structured output with Pydantic models to ensure consistent formatting.

**Phase 2 — Multi-Stage Scoring:**

Entity scoring combines two complementary approaches. Vector Similarity Scoring: Convert query string, named entities, and generic entities to embeddings; use HNSW index to find similar entities with different top_k settings — named entities: top_k=1 (exact matches); generic entities + query: top_k=20 (broad search); threshold filtering at 0.7 for named, 0.5 for generic. Graph Structure Scoring: applies graph algorithms (typically PageRank) to the initial vector similarity scores, allowing highly connected entities to gain importance. Input: vector similarity scores as sparse matrix; Process: graph_storage.score_nodes() applies PageRank-based weighting; Output: Graph-weighted entity scores.

Relationships are scored by propagating entity scores through the entities-to-relationships sparse matrix: relationship_scores = entity_scores × E→R_matrix. Chunks are scored by propagating relationship scores through the relationships-to-chunks sparse matrix.

**Verified pipeline:**

| Step | Action | Signal |
|---|---|---|
| 1 | LLM call: extract named entities + generic entities from query | LLM entity extraction |
| 2 | Encode query, named entities, generic entities → embeddings | Query embedding |
| 3 | HNSW vector search against entity embeddings (top-k=1 for named, top-k=20 for generic+query) | Vector search — entity-first |
| 4 | Filter by similarity threshold (0.7 named, 0.5 generic) | Threshold filter |
| 5 | Apply PageRank: propagate vector scores through graph topology | Graph structure scoring |
| 6 | Propagate entity scores → relationship scores (via E→R sparse matrix) | Matrix multiplication |
| 7 | Propagate relationship scores → chunk scores (via R→C sparse matrix) | Matrix multiplication |
| 8 | Token-budget truncation of top-scored entities + relationships + chunks | Ranking |
| 9 | Assemble context, call LLM | Generation |

**First retrieval hop: LLM entity extraction from query, then vector search on entity embeddings.** Vector search is used to identify seed entities; PageRank then re-weights those seeds by graph topology. Chunks are reached purely through score propagation — they are **never directly ANN-searched.**

---

### Summary Table — Fast GraphRAG

| Dimension | Fast GraphRAG |
|---|---|
| First retrieval signal | LLM entity extraction → vector search on entity embeddings |
| Vector search used? | Yes — HNSW on entity embeddings only |
| Query entities extracted? | Yes — by LLM (named + generic), then embedded for vector search |
| How chunks are retrieved | Via score propagation through sparse matrices: entities → relationships → chunks |
| Role of entities | Primary vector search targets; PageRank seeds |
| Role of relationships | Intermediate propagation step (E→R matrix) |
| Role of communities | **Not present** |
| PageRank's exact role | Re-weights entity scores after initial vector search; not a standalone retrieval step |
| Chunk selection | Scored by propagation from relationships; top-ranked chunks fill token budget |

**Confidence: High for architecture; Medium for precise internal thresholds** (sourced from codebase docs, not a peer-reviewed paper — no arXiv paper exists for Fast GraphRAG at time of verification).

---

## Cross-System Comparison: The Core Ambiguity Resolved

**Is retrieval vector-first, entity-first, or hybrid with routing?**

| System | Mode | Verdict | Detail |
|---|---|---|---|
| MS GraphRAG — Global | Community-broadcast | **Neither** | No vector or entity lookup. All community summaries sent to LLM. |
| MS GraphRAG — Local | **Entity-first via vector** | Hybrid | Vector search is the *entry point into entity space*, not chunk space |
| LightRAG | **Keyword-then-dual-vector** | Hybrid (parallel) | LLM extracts keywords first; two parallel vector searches on entity and relation indexes |
| Fast GraphRAG | **Entity-first via vector + graph** | Hybrid (sequential) | LLM extracts entities; HNSW finds them; PageRank re-weights; chunks reached by propagation |

**Critical shared finding: none of the three systems do direct chunk vector search as the primary retrieval signal.** In all three, the first meaningful retrieval act targets the *entity or relation layer*, not the raw chunk layer. Chunks are always downstream of entity/relation retrieval — either via graph traversal (MS local, Fast), via sparse matrix propagation (Fast), or as associated values in the key-value structure (LightRAG).

**The correct characterization is: all three are entity-first systems where the entity layer is accessed via vector similarity.** The distinction from naive RAG is precisely this: in naive RAG, the vector search targets chunks directly. In all three GraphRAG variants, the vector search targets entity or relation embeddings, and chunks are retrieved via graph-structural hops from those entity seeds.

---

## Unresolved Ambiguities (Not Explicitly Specified in Sources)

1. **MS GraphRAG local — how are candidate entities ranked/filtered before context assembly?** The dataflow shows "Ranking + Filtering" but the official docs do not specify the exact scoring formula (e.g., is it purely by description similarity score, or does graph degree also factor in?). *Not explicitly specified in source.*

2. **LightRAG — does dual-level retrieval always run both levels, or does it route based on query type?** The paper describes two levels but the exact triggering logic (always-both vs. classifier-based routing) is *not explicitly specified in the paper.* The GitHub implementation may differ from the paper description.

3. **Fast GraphRAG — exact PageRank damping factor and convergence criteria** used at query time. The codebase docs describe the mechanism but don't surface these hyperparameters. *Not explicitly specified in source.*

4. **MS GraphRAG global — the paper does not describe any vector search on community embeddings.** The official implementation docs note that community embeddings *are* created at index time and *can* be used for retrieval, but the original paper's global search algorithm uses no vector filtering — it sends all community reports at a given level to the LLM. Whether the production implementation deviates from the paper here is *not fully reconciled across sources.*
