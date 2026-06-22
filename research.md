# GraphRAG: Comprehensive Technical Research Document
*Prepared for ZetaBox AI Engineering Internship — Presentation Research*
*Sources: arXiv papers, Microsoft Research, GitHub repositories, EMNLP 2025 proceedings — June 2026*

---

## 1. RAG Landscape Overview

### 1.1 Naive / Basic RAG

**How it works**
Naive RAG is a linear pipeline: a user query is embedded into a vector, then compared (via cosine similarity) against pre-computed embeddings of text chunks stored in a vector database. The top-k most similar chunks are retrieved and concatenated into a prompt context, which is passed to an LLM for generation. The pipeline has three fixed stages: offline indexing (chunk → embed → store) and online retrieval (embed query → ANN search → generate).

**Strengths**
- Extremely simple to implement and deploy
- Low latency and very low per-query cost (~$0.001)
- Works well for simple, single-document factual lookups
- No structural dependency on data quality beyond embedding quality

**Weaknesses**
- Vector similarity ≠ relevance; retrieves noisy or redundant chunks frequently
- Cannot handle multi-hop reasoning (answering requires combining facts from multiple unrelated chunks)
- No query understanding or transformation — garbage-in, garbage-out
- Lost-in-the-middle problem: LLMs degrade on long retrieved contexts
- Treats all information as isolated chunks — relationship between entities is invisible
- Production benchmarks show ~44% factual accuracy without advanced techniques (CRAG Benchmark, 2024)

**Best for**
Single-document Q&A, simple factual lookups over a small, clean corpus where queries have direct lexical/semantic matches to stored content.

---

### 1.2 Advanced RAG (Enhanced RAG)

**How it works**
Advanced RAG adds optimization stages around the naive pipeline without fundamentally changing its vector-search backbone. **Pre-retrieval optimizations** include query rewriting, query decomposition, HyDE (Hypothetical Document Embedding), and metadata filtering. **Retrieval-time optimizations** include hybrid search (dense vector + BM25 sparse), ensemble retrieval, and multi-index routing. **Post-retrieval optimizations** include cross-encoder reranking, contextual compression, and iterative/multi-hop retrieval passes. State-of-the-art Advanced RAG reaches ~63% factual accuracy vs 44% for naive RAG.

**Strengths**
- Significantly higher precision than naive RAG with modest engineering effort
- Hybrid retrieval captures both semantic and keyword relevance
- Reranking and compression dramatically reduce noise passed to the LLM
- Still relatively low cost and latency compared to graph-based approaches
- Production-proven across a wide range of domains

**Weaknesses**
- Still fundamentally chunk-based — cannot reason about entity relationships
- Multi-hop retrieval is heuristic and brittle; no guaranteed coverage across hops
- Increased pipeline complexity — more components to tune, monitor, and maintain
- Reranking models add latency (~100–300ms per query)
- Cannot answer thematic/summarization queries over large corpora ("what are the main themes across these 5,000 documents?")

**Best for**
Most production RAG applications: customer support, document Q&A, enterprise search. The recommended starting point before considering more complex architectures.

---

### 1.3 Agentic RAG

**How it works**
Agentic RAG replaces the fixed retrieval pipeline with an LLM-driven control loop. The LLM acts as an agent that dynamically decides: whether to retrieve at all, what to retrieve, whether the retrieved context is sufficient, and whether to retry with a refined query or a different tool. Patterns include ReAct (Reason + Act), CRAG (Corrective RAG), Self-RAG (self-reflection on retrieval quality), and multi-agent workflows where specialized agents handle different retrieval tasks. Frameworks: LangGraph, LlamaIndex Agents, AutoGen, CrewAI.

**Strengths**
- Self-correcting: can detect poor retrieval and re-query
- Handles multi-hop queries natively through iterative planning
- Can combine multiple retrieval tools (vector search, SQL, web search, APIs) in one workflow
- Highest answer quality ceiling — DSPy/ReAct benchmarks show +8.3 F1 on HotpotQA vs iterative non-agentic retrieval
- Flexible: adapts retrieval strategy to query complexity at runtime

**Weaknesses**
- Non-deterministic: agent loops are hard to debug and reproduce
- High latency (multiple LLM calls per query — typically 3–10x naive RAG)
- Cost: 5–10x more expensive than naive RAG per query
- Overkill for simple factual lookups — same answer as naive RAG at much higher cost
- Harder to operate in production: requires observability tooling (Langfuse, LangSmith)

**Best for**
Complex, multi-step research tasks; queries that span multiple sources or require tool use; workflows where answer quality justifies latency and cost, such as financial analysis, medical research assistance, or software engineering agents.

---

### 1.4 GraphRAG

**How it works**
GraphRAG replaces the flat chunk index with a structured knowledge graph (KG). Documents are processed offline to extract entities and relationships, which are assembled into a graph. Graph algorithms (community detection, path traversal, subgraph extraction) structure this knowledge hierarchically. At query time, retrieval is graph-aware: it can traverse relationships across entities rather than comparing query embeddings against chunk embeddings. GraphRAG answers two fundamentally different query classes — local (entity-centric) and global (thematic/summarization) — that vector RAG cannot serve well simultaneously.

**Strengths**
- Multi-hop reasoning: follows relationship chains across entities natively
- Global summarization: can answer thematic questions across an entire corpus
- Structured relationship traversal: explicit entity-relationship context vs. implicit semantic similarity
- Explainability: reasoning paths through the graph are traceable
- 72–83% comprehensiveness on global queries vs. vector RAG (Microsoft Research, 2024); 3.4x accuracy improvement in enterprise scenarios (Diffbot benchmark)

**Weaknesses**
- Indexing cost: original Microsoft GraphRAG cost ~$33K to index a 5GB dataset (2024); LazyGraphRAG (Nov 2024) reduced this to 0.1% of that cost
- Graph construction quality is LLM-dependent — poor models produce noisy graphs
- Storage complexity: requires both a graph database and a vector store
- Inference latency: graph traversal + LLM summarization adds 2–3× latency over vector RAG
- Challenging with frequently updated data — graph rebuilds are expensive
- Underperforms on simple factual lookups (+4.5% reasoning depth on HotpotQA at 2.3× higher latency, arXiv:2506.05690)

**Best for**
Entity-rich, interconnected document corpora; queries requiring multi-hop reasoning or global thematic understanding; domains like biomedical research, legal analysis, financial intelligence, and enterprise knowledge management.

---

## 2. GraphRAG — Deep Dive

### 2.1 What is GraphRAG?

**Core idea and motivation**
Standard RAG treats knowledge as a bag of chunks — semantically related by embedding distance, but structurally disconnected. This fails when answers require combining facts across entities: "How is Company A connected to the risk flagged in Document B?" Vector similarity cannot answer this reliably. GraphRAG's motivation is to make the implicit relational structure of a document corpus *explicit* and queryable. By representing entities as nodes and relationships as edges, GraphRAG enables graph traversal — following chains of facts — rather than proximity-based retrieval.

**How it differs from vector-only RAG**

| Dimension | Vector RAG | GraphRAG |
|---|---|---|
| Data structure | Flat embeddings in vector DB | Nodes + edges in knowledge graph |
| Retrieval mechanism | ANN similarity search | Graph traversal + community-based search |
| Multi-hop reasoning | Heuristic / brittle | Native (path traversal) |
| Global summarization | Fails at scale | Supported via community summaries |
| Explainability | Low (black-box similarity) | High (reasoning paths) |
| Indexing cost | Very low | High (but decreasing rapidly) |

**The role of knowledge graphs**
The KG acts as a structured index of facts. Rather than finding "what chunks are semantically similar to this query," the system asks "what entities are related to the query entity, and how?" This factual constraint also reduces hallucination: if a relationship does not exist in the graph, the system will not fabricate it (Ontotext/Elastic research; structural grounding effect).

---

### 2.2 GraphRAG Architectures

#### Microsoft GraphRAG (Reference Implementation)

**What makes it distinct**
The canonical GraphRAG approach from Microsoft Research (Edge et al., April 2024; open-sourced July 2024, 29,800+ GitHub stars by Dec 2024). It is the only architecture that performs *pre-computed hierarchical community summarization* — making it uniquely capable of answering global thematic queries but uniquely expensive to index.

**Graph construction**
1. Corpus is chunked into text units
2. GPT-4 extracts `(entity, relationship, entity)` triples and entity descriptions from each chunk
3. Entities are deduplicated via embedding similarity; edge weights reflect co-occurrence frequency
4. Leiden community detection clusters entities into hierarchically nested communities (~50–500 communities for a typical corpus)
5. LLM generates multi-level "community reports" (summaries) for each cluster at every hierarchy level

**Query pipeline**
- **Local search**: Entity-centric. Embeds the query, retrieves the most relevant entities from the graph, expands to their neighborhood (related entities, relationships, source chunks), and synthesizes with an LLM. Best for specific, focused questions.
- **Global search**: Community-centric. Aggregates all community summaries (map-reduce style) to answer thematic questions. Best for "what are the main themes / patterns across the entire corpus?" — questions vector RAG fundamentally cannot answer.
- **DRIFT search** (Oct 2024): Hybrid. Starts with global search to form a hypothesis, then uses local search to verify and refine. Combines breadth and precision.

**Trade-offs**
- Indexing is extremely LLM-heavy: every entity, relationship, and community requires LLM calls — original cost ~$33K for a large corpus
- Storage: graph database (or NetworkX/Parquet for smaller corpora) + vector database for embeddings
- Not designed for incremental updates — graph rebuilds are expensive

---

#### LightRAG (HKUDS, October 2024 — EMNLP 2025)

**What makes it distinct**
LightRAG (Guo et al., arXiv:2410.05779) is a lightweight alternative that skips Microsoft GraphRAG's expensive community clustering step entirely. It uses a *dual-level retrieval* architecture — low-level (entity-specific) and high-level (concept-level) — with both a KG and a vector index. The key design decision: instead of pre-computing community summaries, it stores (entity, relationship) pairs as both graph elements and key-value entries queryable by vector similarity.

**Graph construction**
LLM extracts entities and relationships from chunks; both are stored with dual-level keys (specific entity name + high-level concept). No Leiden clustering. Supports incremental indexing — new documents can be added without full reindex.

**Query pipeline**
Query keywords are extracted; the system retrieves matching entities and relationships via vector search over the dual-level index, then assembles context combining graph elements and original text chunks.

**Trade-offs**
- ~60% lower indexing token cost vs Microsoft GraphRAG; ~2× lower query latency (arXiv:2506.05939)
- Consistently outperforms NaiveRAG, RQ-RAG, HyDE, and Microsoft GraphRAG across agriculture, CS, legal, and mixed domains (EMNLP 2025 evaluation)
- Does not support global thematic summarization as well as Microsoft GraphRAG (no community hierarchy)
- Best for corpora where you need multi-hop accuracy without the full community-detection overhead

---

#### Fast GraphRAG (CircleMind AI, 2024)

**What makes it distinct**
Fast GraphRAG prioritizes inference speed and incremental updates. It uses a personalized PageRank-based retrieval mechanism over the entity graph rather than community summaries, which eliminates the need for expensive pre-computed summaries entirely. Designed for production workloads where the document corpus changes frequently.

**Graph construction**
LLM extracts entities from chunks; edges are co-occurrence-weighted. No community detection step. Graph is stored to support efficient PageRank traversal.

**Query pipeline**
Entities in the query are identified; PageRank propagation over the knowledge graph identifies the most relevant nodes. Top-ranked entities and their neighborhoods are assembled into context.

**Trade-offs**
- Very fast indexing (no LLM summarization overhead beyond entity extraction)
- Supports incremental updates natively
- Lower global summarization quality than Microsoft GraphRAG
- Less battle-tested in production at scale than LightRAG or Microsoft GraphRAG

---

#### LlamaIndex PropertyGraphIndex

**What makes it distinct**
PropertyGraphIndex is LlamaIndex's framework-native GraphRAG implementation. Rather than a standalone system, it is a composable abstraction that lets you build GraphRAG pipelines using LlamaIndex's existing document loaders, LLM wrappers, and retriever interfaces. It supports custom `KGExtractors` (for defining your own entity/relationship extraction logic) and pluggable graph stores (in-memory, Neo4j, Nebula).

**Graph construction**
Configurable: you define triplet extraction prompts or use built-in extractors. Follows the same extract → build graph → detect communities → generate summaries pattern, but all steps are modular and swappable. Supports both knowledge graph triplets and property graphs (nodes/edges with arbitrary metadata).

**Query pipeline**
`GraphRAGQueryEngine`: retrieves entities matching query, looks up their community IDs, fetches community summaries, generates partial answers per community, then aggregates. Highly customizable — you can swap any component.

**Trade-offs**
- Maximum flexibility and LlamaIndex ecosystem integration
- Requires more engineering to get production-ready vs. the Microsoft CLI-based approach
- Performance depends heavily on extractor and store quality — no opinionated defaults
- Best for teams already on LlamaIndex who want GraphRAG as one retrieval strategy among many

---

### 2.3 Microsoft GraphRAG — Detailed Pipeline

#### Indexing Pipeline

```
Raw Documents
    │
    ▼
[1] Chunking
    Text split into overlapping units (~600 tokens)
    │
    ▼
[2] Entity & Relationship Extraction  ← LLM call (GPT-4)
    For each chunk: extract (entity_name, type, description)
    and (source, relation, target, description) triples
    │
    ▼
[3] Entity Deduplication & Graph Construction
    Entities merged via embedding similarity + name matching
    Edges weighted by co-occurrence frequency
    │
    ▼
[4] Community Detection — Leiden Algorithm
    Graph partitioned into hierarchically nested communities
    Guarantees well-connected communities (vs Louvain's weakness)
    Produces multiple levels: leaf → mid → root
    │
    ▼
[5] Community Report Generation  ← LLM call (per community, per level)
    LLM summarizes each community's entities, relationships, and covariates
    Reports stored as "pre-computed answers" to thematic queries
    │
    ▼
[6] Embedding
    Entities, relationships, and community reports embedded for vector search
```

**Role of LLMs in the pipeline:**
- Entity extraction: one LLM call per chunk (the dominant cost driver)
- Relationship extraction: bundled with entity extraction
- Community summarization: one LLM call per community per hierarchy level
- Query-time generation: one or more LLM calls for answer synthesis

**Why indexing is expensive:**
Every chunk requires an LLM call for extraction. Every community at every level requires an LLM call for summarization. For a 5GB legal corpus, this produced ~$33,000 in API costs (GPT-4, 2024 pricing). LazyGraphRAG (Nov 2024) defers LLM summarization to query time and uses lightweight NLP for indexing — reducing cost to 0.1% of full GraphRAG.

#### Query Pipeline

**Local Search** (entity-centric, specific questions)
```
User Query → Embed query → Find top-k entities via vector similarity
           → Expand to entity neighborhood (related entities, edges, source chunks)
           → Assemble tabular context
           → LLM generates answer
```

**Global Search** (community-centric, thematic questions)
```
User Query → Retrieve all community reports at a given hierarchy level
           → LLM scores each community report for relevance (map phase)
           → LLM aggregates top-scored summaries into final answer (reduce phase)
```

**DRIFT Search** (hybrid)
```
User Query → Global search → initial broad answer + follow-up sub-questions
           → Local search for each sub-question
           → Iterative refinement (typically 2 passes)
           → Final synthesized answer
```

#### Storage Requirements
- **Graph store**: NetworkX (in-memory/Parquet for small corpora), Neo4j, or Azure Cosmos DB for large-scale production
- **Vector store**: Any — pgvector, Qdrant, Weaviate, Azure AI Search
- **Relational**: Entity tables, relationship tables, community tables (Parquet or SQLite for local; Postgres for production)

---

### 2.4 Graph Types in GraphRAG

**Knowledge Graphs (KG)**
- Directed, typed triples: `(subject, predicate, object)` — e.g., `(Ibuprofen, treats, Inflammation)`
- Curated or LLM-extracted; entities have types (Person, Drug, Organization)
- Best for: domains with rich entity relationships, multi-hop queries, biomedical, legal
- Example use: Microsoft GraphRAG, LightRAG, HippoRAG

**Property Graphs**
- Nodes and edges with arbitrary key-value metadata (properties)
- More flexible than strict KGs — edges can carry rich attributes
- Native to Neo4j, Memgraph, Amazon Neptune
- Best for: enterprise data with complex, heterogeneous relationships; LlamaIndex PropertyGraphIndex uses this model

**Ontology-Based Graphs**
- Knowledge graphs governed by a formal schema (ontology) defining allowed entity types, relationship types, and axioms
- Curated manually by domain experts; higher precision, lower recall than LLM-extracted
- Best for: healthcare (SNOMED CT, UMLS), finance, regulatory compliance — domains with strict definitional requirements
- Enables logical inference (SPARQL, OWL reasoning) beyond pattern matching

---

### 2.5 Advantages of GraphRAG over Vector RAG

- **Multi-hop reasoning**: Graph traversal natively follows relationship chains — `Drug → inhibits → Enzyme → regulates → Pathway` — without brittle multi-step retrieval heuristics
- **Global summarization**: Hierarchical community summaries enable answering "what are the main themes across 10,000 documents?" — a query that causes vector RAG to either truncate context or hallucinate
- **Structured relationship traversal**: Relationships between entities are first-class facts — not lost to chunk boundaries or embedding drift
- **Handles complex queries at scale**: 80% accuracy vs 50% for traditional RAG on complex queries (Lettria/AWS, Dec 2024); 3.4× enterprise benchmark improvement (Diffbot 2023)
- **Explainability**: Reasoning paths through the graph are auditable — "this answer was derived from the path A → B → C" — critical for regulated industries
- **Hallucination reduction**: Structural grounding constrains generation — if the relationship isn't in the graph, it won't be synthesized (Ontotext/Elastic research)
- **Scalable knowledge base growth**: Graph structures allow incremental expansion without full re-indexing (especially LightRAG and Fast GraphRAG)

---

### 2.6 Disadvantages and Limitations

- **Indexing cost**: Full Microsoft GraphRAG cost ~$33K for a large corpus in 2024; reduced to ~$33 by mid-2025 with LazyGraphRAG, but still requires engineering investment
- **LLM-dependent graph quality**: Entity extraction errors compound — the same entity ("Sagar S", "Sagar Shankaran", "S Shankaran") may appear as 3 separate nodes; relationship precision is bounded by prompt and model quality
- **Storage complexity**: Requires managing both a graph store and a vector store, plus relational entity/community tables — more operational burden than pure vector RAG
- **Query latency**: Graph traversal + LLM summarization adds 2–3× latency over vector RAG (arXiv:2506.05939); global search involves map-reduce over all communities
- **Dynamic data challenges**: Graph rebuilds are expensive; incremental update support is still maturing (LightRAG and Fast GraphRAG handle this better than Microsoft GraphRAG)
- **Overkill for simple queries**: GraphRAG yields only +4.5% reasoning depth improvement on HotpotQA for simple factoid lookups at 2.3× higher latency (arXiv:2506.05690) — vector RAG is better for point lookups
- **Entity resolution is hard**: Deduplication via embedding + name-matching is imperfect at scale; requires alias rules and heuristic resolution passes
- **Super-linear storage growth**: Graph index and community summaries grow super-linearly with corpus size (arXiv:2506.05939)

---

### 2.7 When to Use GraphRAG — Decision Framework

**Use GraphRAG when:**
- Queries require multi-hop reasoning: "What are all the side effects of drugs prescribed to patients who are also on Drug X?"
- Queries require global/thematic understanding: "What are the recurring risk themes across all Q3 earnings calls?"
- Data is entity-rich and has meaningful relationships between entities
- Corpus is large (>1,000 documents) and interconnected — graph density matters
- Explainability and auditability are required (regulated industries)
- Budget allows for higher indexing cost (or you use LazyGraphRAG/LightRAG)

**Use Naive/Advanced RAG when:**
- Queries are simple factual lookups against a clean, small corpus
- Latency is critical and costs are constrained
- The corpus is not entity-rich (e.g., general prose articles with few named entities)
- Relationships between entities are not important to the query

**Use Agentic RAG when:**
- Queries are complex and multi-step but data is NOT graph-structured
- The system needs to use multiple tools (web search, SQL, code execution) in a single workflow
- Budget and latency tolerance are high

**Use Hybrid (Vector + Graph) for production:**
Most 2025–2026 production stacks route query types: simple/factual → vector RAG; multi-hop/relational → GraphRAG; mixed-complexity → DRIFT or agentic orchestration. This is the recommended production architecture.

**Quick decision tree:**

```
Is the corpus large (>1K docs) and entity-rich?
├── No  → Advanced RAG (hybrid search + reranker)
└── Yes → Do queries require multi-hop or thematic reasoning?
          ├── No  → Advanced RAG
          └── Yes → Is budget / latency constrained?
                    ├── Yes → LightRAG or LazyGraphRAG
                    └── No  → Microsoft GraphRAG (full pipeline)
```

---

## 3. GraphRAG Use Cases

### 3.1 Biomedical / Life Sciences

GraphRAG is particularly powerful in biomedical contexts because the domain is defined by entity relationships: drugs inhibit enzymes, genes regulate pathways, proteins interact with receptors.

- **Drug discovery**: KG-based GraphRAG can traverse drug-target-pathway chains to identify repurposing candidates or predict drug-drug interactions — relationships invisible to vector RAG
- **Alzheimer's research**: Cedars-Sinai built a 1.6M-edge Alzheimer's knowledge graph (Memgraph, 2024) for entity-rich research navigation
- **Clinical knowledge**: GraphRAG on EHR data enables answering complex clinical queries: "Which patients on Drug A also have Condition B and were prescribed Drug C in the last 6 months?"
- **Diabetes management**: Precina Health achieved a 1% monthly HbA1c reduction (12× faster than standard care) using GraphRAG-powered clinical decision support (2024)
- **LLM hallucination in medical QA**: Addressing accuracy and hallucination in Alzheimer's disease research through knowledge graphs (arXiv:2508.21238) — using Microsoft GraphRAG's community summaries for grounded biomedical Q&A

### 3.2 Legal Document Analysis

Legal reasoning is inherently graph-structured: statutes reference other statutes, cases cite precedents, obligations flow between parties.

- **Contract parsing**: Graph-agentic systems achieve ~90% precision in clause classification (IEEE Trans. on AI survey, 2025) — outperforming vector RAG significantly
- **Multi-jurisdiction queries**: Traversing EU, UK, and US case law graphs (72.7K documents, 295.58M tokens — arXiv:2506.05690) for cross-jurisdiction precedent mapping
- **Regulatory compliance**: Ontology-based graphs map obligations and prohibitions to a regulatory schema, enabling gap analysis and compliance verification
- **CBR-RAG for legal QA**: Case-based reasoning over legal KGs (ICCBR 2024) — retrieving precedent cases via graph similarity

### 3.3 Financial Knowledge Bases

Financial data is inherently relational: companies, executives, transactions, and market signals form a dense graph.

- **Fraud detection / AML**: Banks use GraphRAG to detect money laundering rings — "How is Customer A connected to sanctioned Entity B?" across millions of transaction records
- **Credit risk and due diligence**: GA-RAG integrates corporate hierarchies and market signals for comprehensive risk assessment; decomposed into specialized subtasks via orchestrator-worker models
- **Earnings analysis**: Global search over community summaries of quarterly reports to identify recurring risk themes across sectors
- **Portfolio intelligence**: 360-degree customer/company insights by traversing corporate ownership graphs, product relationships, and news entities
- **OmniEval benchmark**: Financial-domain RAG evaluation (EMNLP 2025) shows graph-based retrieval significantly outperforms vector RAG for KPI and forecast queries (90%+ vs 0% for vector RAG on schema-bound queries)

### 3.4 Enterprise Knowledge Management

- **Supply chain risk analysis**: Finding hidden dependencies across supplier relationships, contract terms, and external risk signals — impossible with flat vector search
- **IT/DevOps knowledge**: GraphRAG over runbooks, incident reports, and architecture documents — enables multi-hop questions like "What services were affected by all incidents involving Service A?"
- **HR and org intelligence**: Traversing organizational hierarchies, skill graphs, and project histories for HR analytics
- **Microsoft Discovery**: Anthropic's GraphRAG integrated into Microsoft Discovery, an agentic scientific research platform on Azure

### 3.5 Cybersecurity Threat Intelligence

- **Attack graph analysis**: Representing CVEs, attack techniques (MITRE ATT&CK), threat actors, and affected systems as a graph — enables multi-hop threat hunting: "What TTPs are associated with APT group X and which of our systems are exposed?"
- **Indicator of Compromise (IoC) mapping**: Traversing relationships between IP addresses, malware families, domains, and affected sectors
- **Vulnerability knowledge bases**: NVD/CVE data as a KG — multi-hop queries for transitive vulnerability impact

---

## 4. Evaluation & Benchmarks

### 4.1 Evaluation Frameworks

**RAGAS (Retrieval-Augmented Generation Assessment)**
Four core metrics:
- **Faithfulness**: Is the answer grounded in the retrieved context? (hallucination detection)
- **Answer Relevance**: Does the answer actually address the question?
- **Context Precision**: What fraction of retrieved chunks are relevant?
- **Context Recall**: Does the retrieved context contain all information needed for the answer?

Caveat: Cleanlab's 2025 benchmarks found RAGAS fails on 83.5% of production examples — it works better as a relative comparison tool than an absolute ground-truth evaluator.

**TRIAD** (used in some academic evaluations): Truthfulness, Relevance, Informativeness, Answer Definitiveness.

**LLM-as-a-Judge**: Microsoft GraphRAG's own evaluation uses GPT-4 to compare answers on comprehensiveness, diversity, empowerment, and directness — a relative pairwise comparison rather than absolute scoring.

### 4.2 Benchmark Comparisons

**HotpotQA** (multi-hop QA, 2-hop reasoning required):
- GraphRAG (local): +4.5% reasoning depth improvement over vector RAG, at 2.3× higher latency (arXiv:2506.05690)
- Agentic iterative RAG: +8.3 F1 over Iter-RetGen (arXiv state-of-the-art survey, 2026)
- Microsoft GraphRAG with GNN reranker: competitive on HotpotQA (IEEE Trans. on AI, Aug 2025)
- LightRAG: outperforms NaiveRAG, RQ-RAG, HyDE, and Microsoft GraphRAG on agriculture, CS, legal, mixed domains (EMNLP 2025)

**Microsoft GraphRAG global query evaluation** (Edge et al., 2024):
- Comprehensiveness: 72–83% vs vector RAG on global/thematic questions
- Diversity: graphRAG produces significantly more varied, multi-perspective answers

**Enterprise benchmarks**:
- 80% accuracy vs 50% for traditional RAG on complex enterprise queries (Lettria/AWS, Dec 2024)
- 3.4× improvement on enterprise benchmarks (Diffbot KG-LM Benchmark, 2023)
- 90%+ accuracy on schema-bound KPI/forecast queries vs 0% for vector RAG (Salfati Group / 2025 benchmark)

**Key insight from arXiv:2506.05690** ("When to use Graphs in RAG"):
GraphRAG benefits are *highly query-type-dependent*. On factoid questions, vector RAG is still competitive or superior. GraphRAG's gains are concentrated in multi-hop and global query types — which is precisely where vector RAG fails fundamentally.

### 4.3 Key Metrics Summary

| Metric | Measures | Relevant for |
|---|---|---|
| Faithfulness | Hallucination rate | All RAG types |
| Answer Relevance | On-topic rate | All RAG types |
| Context Precision | Retrieval noise | Naive/Advanced RAG |
| Context Recall | Retrieval coverage | Naive/Advanced RAG |
| Comprehensiveness | Breadth of answer | GraphRAG global |
| Multi-hop accuracy | Chain-of-reasoning | GraphRAG, Agentic RAG |
| Entity F1 | Named entity recall | GraphRAG extraction quality |

---

## 5. Implementation Landscape

### 5.1 Key Open-Source Tools

| Tool | Language | Graph Store | Best For |
|---|---|---|---|
| Microsoft GraphRAG | Python | NetworkX / Parquet / Azure | Full pipeline, global queries |
| LightRAG | Python | Neo4j, NetworkX, PostgreSQL | Cost-efficient, incremental |
| Fast GraphRAG | Python | In-memory | Fast iteration, dynamic corpora |
| LlamaIndex PropertyGraphIndex | Python | Neo4j, Nebula, in-memory | LlamaIndex ecosystem integration |
| LangChain graph integrations | Python | Neo4j, Amazon Neptune | Custom graph pipelines |
| Neo4j GraphRAG | Python/JS | Neo4j (native) | Enterprise, production-grade graph ops |
| HippoRAG / HippoRAG2 | Python | In-memory KG | Academic, high-accuracy multi-hop |

### 5.2 Typical Production Tech Stack

```
[Document Ingestion]
PDF/Word/HTML → Apache Tika / Unstructured → Text chunks

[Entity Extraction]
Text chunks → LLM (GPT-4o or Claude Sonnet) → (entity, relation, entity) triples

[Graph Store]
Small corpus:   NetworkX + Parquet
Medium corpus:  Neo4j (self-hosted or AuraDB)
Large corpus:   Azure Cosmos DB / Amazon Neptune / Memgraph

[Vector Store]
pgvector, Qdrant, Weaviate, Azure AI Search, Pinecone

[Orchestration]
LangChain / LlamaIndex / Microsoft GraphRAG CLI

[Query Routing]
Simple queries → vector RAG path
Complex queries → GraphRAG local/global/DRIFT

[Observability]
Langfuse, LangSmith, Azure AI Studio

[LLM]
OpenAI GPT-4o (standard), GPT-4o-mini (cost reduction)
Claude Sonnet 4.x (alternative — strong entity extraction)
Local: Llama 3.3, Mistral for cost reduction
```

### 5.3 Approximate Indexing Costs (Microsoft GraphRAG)

| Corpus Size | Approx. Cost (GPT-4, 2024) | With LazyGraphRAG (2025) |
|---|---|---|
| ~100 documents (~1M tokens) | ~$100–500 | ~$0.10–0.50 |
| ~1,000 documents (~10M tokens) | ~$1,000–5,000 | ~$1–5 |
| ~10,000 documents (5GB) | ~$33,000 | ~$33 |

**Cost drivers**: Number of entity extraction LLM calls (proportional to chunks) + number of community summary calls (proportional to graph density × hierarchy depth).

**Cost reduction strategies**:
- Use GPT-4o-mini or a local model for extraction; GPT-4o for summarization only
- Use LazyGraphRAG (deferred summarization) for one-off or exploratory use
- Use LightRAG (skips community summarization entirely) for production at scale
- Hybrid: build graph with smaller model, validate with larger model

---

## 6. Recent Research & Trends (2024–2025)

### 6.1 Most Impactful Papers

| Paper | Year | Contribution |
|---|---|---|
| Edge et al., "From Local to Global: A GraphRAG Approach to Query-Focused Summarization" (arXiv:2404.16130) | Apr 2024 | Microsoft GraphRAG — the foundational paper |
| Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation" (EMNLP 2025, arXiv:2410.05779) | Oct 2024 | Dual-level KG-RAG, 60% lower cost than MS GraphRAG |
| Microsoft Research, "LazyGraphRAG: Setting a New Standard for Quality and Cost" | Nov 2024 | Deferred summarization — 0.1% of MS GraphRAG indexing cost |
| arXiv:2506.05690, "When to use Graphs in RAG: A Comprehensive Analysis" | Jun 2025 | Systematic analysis of when GraphRAG helps vs hurts |
| arXiv:2601.05264, "Engineering the RAG Stack" | Jan 2026 | Comprehensive survey of RAG architectures and trust frameworks |
| arXiv:2501.09136, "Agentic RAG Survey" | Jan 2025 | Taxonomy and benchmarks for agentic RAG |
| arXiv:2503.06474, "ROGRAG: Robustly Optimized GraphRAG" | Mar 2025 | Improving entity extraction and retrieval robustness |

### 6.2 Key Research Directions

**Lazy / Deferred Graph Construction**
LazyGraphRAG showed that pre-computing community summaries is not always necessary. The field is moving toward *on-demand* graph construction where the graph is built incrementally or at query time, dramatically reducing upfront cost.

**Incremental / Streaming Indexing**
LightRAG supports incremental document addition. Active research: how to update graph structures (add/remove entities and relationships) without full reindex — critical for production systems with frequently updated corpora.

**Hybrid Retrieval Routing**
"Don't force every query through the same expensive pipeline" is the 2025 consensus. Active research on lightweight query classifiers that route to vector RAG, GraphRAG local, GraphRAG global, or agentic RAG based on query characteristics.

**Cost Reduction via Smaller Models**
Research showing that open-source dependency parsers for KG construction achieve performance comparable to GPT-4o on RAGAS metrics (arXiv:2507.03226) — reducing the LLM dependency in the extraction phase.

**HippoRAG 2 (2025)**
Extends HippoRAG with passage-level embeddings and Personalized PageRank over the KG — achieving similar quality to Microsoft GraphRAG at 10–30× lower cost and 6–13× lower latency.

**Multimodal GraphRAG**
Emerging research on integrating image, table, and video understanding into the graph construction phase — beyond text-only entity extraction.

**Temporal / Event-Based KGs**
Research on encoding temporal ordering and causality as first-class graph properties (arXiv:2506.05939) — enabling queries like "What events led to X after Y?"

**Agentic + Graph convergence**
KG-RAG-Agent framework (EAMCON 2025): combining GraphRAG with agentic retrieval significantly improves multi-hop accuracy over both pure vector RAG and static GraphRAG. A dedicated NeurIPS 2025 workshop ("Knowledge Graphs & Agentic Reasoning") confirmed this is a mainstream research direction.

### 6.3 Where the Field Is Heading

1. **Cost democratization**: LazyGraphRAG and LightRAG have already reduced the $33K indexing barrier to ~$33–$100 for most corpora. By 2026, GraphRAG indexing with open models is approaching vector RAG cost parity.

2. **Hybrid by default**: Production stacks increasingly route queries by type — simple → vector, complex → graph. This hybrid architecture is becoming the de facto standard.

3. **Graph + Agents convergence**: The next major paradigm shift is graph-augmented agents — where the KG serves as both a retrieval source and a persistent shared memory across agent steps.

4. **Federated graphs**: Cross-organizational knowledge graph construction — enabling multi-enterprise knowledge synthesis without data sharing, using privacy-preserving techniques.

5. **Autonomous graph curation**: LLM agents that continuously update and validate the knowledge graph as new documents arrive — closing the gap between static graph indexing and real-time knowledge.

---

## Appendix: Quick Reference Card

### RAG Paradigm Comparison

| Dimension | Naive RAG | Advanced RAG | Agentic RAG | GraphRAG |
|---|---|---|---|---|
| Indexing complexity | Low | Low-Medium | Low | High |
| Query cost | $0.001 | $0.002–0.005 | $0.005–0.01 | $0.01–0.05 |
| Latency | ~500ms | ~1s | ~5–15s | ~2–5s |
| Multi-hop reasoning | ✗ | Weak | ✓ | ✓ |
| Global summarization | ✗ | ✗ | Partial | ✓ |
| Explainability | Low | Low-Medium | Medium | High |
| Incremental updates | ✓ | ✓ | ✓ | Partial |
| Best corpus size | Any | Any | Any | >1K docs |
| Recommended stack | LangChain/LlamaIndex | Same + reranker | LangGraph | MS GraphRAG / LightRAG |

### GraphRAG Architecture Comparison

| Feature | MS GraphRAG | LightRAG | Fast GraphRAG | LlamaIndex PGI |
|---|---|---|---|---|
| Community detection | ✓ Leiden | ✗ | ✗ | ✓ (configurable) |
| Incremental updates | ✗ | ✓ | ✓ | Partial |
| Global queries | ✓ Excellent | ✓ Good | ✗ Weak | ✓ Good |
| Local queries | ✓ Good | ✓ Excellent | ✓ Good | ✓ Good |
| Indexing cost | High | ~60% lower | Very low | Medium |
| Production readiness | High | High | Medium | Medium |
| Open source | ✓ | ✓ | ✓ | ✓ |
