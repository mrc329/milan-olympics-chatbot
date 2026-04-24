# Retrieval & Chunking Review — Milan 2026 Olympics Chatbot

## What Was Built

### Chunking

| Setting | Value |
|---|---|
| Max chunk size | 300 words |
| Min chunk size | 40 words |
| Overlap | None |
| Method | Word-count truncation |
| Sentence boundary aware | No |

**Flow:**
1. RSS feeds fetched every 30 minutes
2. Articles filtered — must mention "Milano Cortina 2026" or a venue keyword
3. Each article split into 300-word chunks (simple whitespace split)
4. Chunks below 40 words discarded

### Embedding

- Model: `all-MiniLM-L6-v2` (SentenceTransformers)
- Dimensions: 384
- Used consistently across ingestion, retrieval, and evaluation

### Vector Storage (Pinecone)

Three namespaces:
- `athletes` — athlete profiles, medals, injuries
- `events` — results, upsets, country upsets
- `narratives` — RSS content, rumors, injuries, Wikipedia pages

### Retrieval

- Embed the user query with the same model
- Query all namespaces in parallel, top 7 per namespace
- Merge all results, re-rank by cosine score, return top 7
- Figure skating queries: double athlete top-K, apply 30% score boost for known star athletes

### Deduplication (ingestion)

- Before upserting, query Pinecone for nearest neighbor
- If cosine similarity >= 0.92, skip the chunk as a duplicate

### Evaluation

- Framework: RAGAS
- Metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`
- Test set: 33 questions across all sports

---

## What Went Wrong — Technical Weaknesses

### Chunking Failures

**1. No overlap**
The most critical gap. If a key fact spans the boundary between two chunks, neither chunk contains complete context. Standard practice is 10–20% overlap to preserve continuity.

**2. Word-count instead of token-count**
LLMs operate on tokens, not words. A 300-word chunk can exceed 400 tokens. The chunk size does not map accurately to the model's actual context window budget.

**3. No sentence boundary awareness**
Splitting on word count cuts mid-sentence. A chunk ending `"...Malinin landed the quad axel before"` and the next starting `"falling in the second rotation"` produces two broken, half-useless chunks.

**4. Flat document chunking**
No awareness of document structure (paragraphs, sections, headings). A structured Wikipedia article and a short RSS blurb are processed identically.

---

### Retrieval Failures

**1. No reranking**
Retrieval is purely by vector similarity. A cross-encoder reranker as a second pass significantly improves precision by scoring query-chunk relevance jointly rather than independently. This is the most commonly expected improvement.

**2. No hybrid search**
Pure dense vector search misses exact keyword matches. Hybrid search combines dense embeddings with BM25 (sparse keyword matching) to handle both semantic and lexical queries. Pinecone supports this natively.

**3. Hardcoded domain boosting**
The figure skating score boost only works for keywords and athlete names hardcoded at write time. This does not generalise. A proper solution uses a query classification layer or a routing agent.

**4. Weak embedding model**
`all-MiniLM-L6-v2` is fast and small but 384 dimensions limits semantic richness. Production-grade alternatives: `bge-large-en-v1.5`, `text-embedding-3-large`, or `e5-large-v2`.

**5. No redundancy filtering post-retrieval**
The top 7 returned chunks can contain near-duplicate content, wasting context window space. MMR (Maximum Marginal Relevance) penalises redundant results and improves diversity.

---

## What Good Looks Like

| Area | Current | Better |
|---|---|---|
| Chunking | Fixed word-count, no overlap | Sentence-aware, 10-20% overlap, token-counted |
| Embedding | all-MiniLM-L6-v2 (384-dim) | bge-large or text-embedding-3-large |
| Search | Dense vector only | Hybrid (dense + BM25) |
| Post-retrieval | Score sort only | Cross-encoder reranker + MMR |
| Query routing | Hardcoded keyword rules | Query classifier or routing agent |

---

## How to Explain It in an Interview

> "I built a RAG pipeline for a real-time Olympics chatbot. Data comes from RSS feeds every 30 minutes. I filter for relevance, chunk to 300 words, deduplicate against Pinecone by cosine similarity, then embed with all-MiniLM-L6-v2 and store across three namespaces. At query time I search all namespaces in parallel and return the top 7 chunks by score.
>
> The main weaknesses I'd fix: add chunk overlap so context isn't lost at boundaries, switch to hybrid search to handle keyword queries, and add a cross-encoder reranker as a second pass to improve precision. The embedding model choice was a speed/cost tradeoff — a larger model would improve quality."
