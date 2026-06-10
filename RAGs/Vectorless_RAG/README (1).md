# Vectorless RAG: Hierarchical Index over Financial Documents

**Course:** Applied NLP / LLM Systems
**Topic:** Retrieval-Augmented Generation without Vector Search
**Document Used:** BSE Corporate Filing (Annual Report PDF)

---

## What This Notebook Is About

This notebook teaches you a retrieval strategy called **Vectorless RAG**. Instead of converting text into embedding vectors and doing similarity search (which is the standard approach), it builds a **tree-shaped index** of the document and navigates that tree using an LLM at query time.

By the end of this notebook you will understand two complete RAG pipelines, how to compare them, and when to use each one.

---

## Complete Code Architecture

```
                        BSE Annual Report PDF
                               |
                        [ PdfReader ]
                               |
                     List of page strings
                    (one string per page)
                               |
              +----------------+----------------+
              |                                 |
     [ build_index() ]               [ build_traditional_index() ]
              |                                 |
    Hierarchical Tree                    Flat list of chunks
    (root > sections > pages)            + TF-IDF matrix
    Each node stores:                    (sklearn)
      - raw_text                                |
      - summary (by LLM)                        |
      - children list                  [ traditional_query() ]
              |                          TF-IDF similarity
    [ vectorless_query() ]               -> top-k chunks
      1. LLM picks best section          -> LLM answers
      2. LLM picks best page
      3. LLM answers from raw text
              |                                 |
              +----------------+----------------+
                               |
                     Side-by-Side Comparison
```

### Tree Structure of the Hierarchical Index

```
root  (level 0)
  |- section_1  (level 1, pages 1-5)
  |    |- page_1  (level 2)
  |    |- page_2
  |    |- page_3
  |    |- page_4
  |    |- page_5
  |- section_2  (level 1, pages 6-10)
  |    |- page_6
  |    |- ...
  |- section_3  (level 1, pages 11-15)
  |- section_4  (level 1, pages 16-20)
```

Every node (root, section, page) holds a short **LLM-generated summary**. Retrieval works by reading these summaries top-down and descending into the most relevant subtree. No vectors are stored anywhere.

---

## Complete Code Flow

```
Step 1: Install packages
        pypdf, llama-index, openai, sklearn

Step 2: Load and extract PDF
        Download PDF --> PdfReader --> pages[] (list of strings)

Step 3: Build Hierarchical Index
        For each page --> create leaf DocumentNode --> LLM summarises it
        Group N pages --> create section DocumentNode --> LLM summarises group
        Combine all sections --> create root DocumentNode --> LLM summarises all

Step 4: Vectorless Query
        Query --> LLM picks best section --> LLM picks best page --> LLM answers

Step 5: Build Traditional Index (Baseline)
        pages --> split into word-chunks --> TF-IDF vectoriser.fit_transform()

Step 6: Traditional Query
        Query --> TF-IDF similarity --> top-k chunks --> LLM answers

Step 7: Compare both approaches on same query
```

At this point you should have ~90% of what is happening in the notebook. The sections below go deeper into each code block.

---

## Block-by-Block Code Walkthrough

---

### Block 1: Package Installation and Imports

```python
%pip install pypdf llama-index llama-index-llms-openai openai
```

Installs the four core libraries. `pypdf` handles PDF reading. `openai` provides the LLM client. `llama-index` is imported but its main use here is conceptual framing (the actual index is built from scratch).

```python
import os, requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from pypdf import PdfReader
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "..."
client = OpenAI()
```

Sets up the OpenAI client that all LLM calls in this notebook go through. The `dataclass` import is used in the next block to define the node structure. The API key is hardcoded here for demo purposes; in real projects it should come from an environment variable or secrets manager.

---

### Block 2: Download and Read the PDF

```python
PDF_URL = "https://www.bseindia.com/..."
PDF_PATH = Path("annual_report.pdf")

if not PDF_PATH.exists():
    response = requests.get(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
    PDF_PATH.write_bytes(response.content)

reader = PdfReader(str(PDF_PATH))
pages = [page.extract_text() or "" for page in reader.pages]
```

Downloads the PDF only if it has not already been saved locally (caching). Then `PdfReader` reads it and `extract_text()` is called on every page. The result is `pages`, a plain Python list where `pages[0]` is the text of page 1, `pages[1]` is page 2, and so on. The `or ""` handles scanned/image pages that return `None`.

Key point: everything downstream works on this simple list of strings. No special format is needed.

---

### Block 3: DocumentNode Dataclass

```python
@dataclass
class DocumentNode:
    node_id: str
    level: int
    page_range: tuple
    raw_text: str
    summary: str = ""
    children: List["DocumentNode"] = field(default_factory=list)
    parent: Optional["DocumentNode"] = field(default=None, repr=False)
```

This is the blueprint for every node in the tree. Think of it as the fundamental unit of the index.

Key fields to understand:
- `level` tells you where in the tree this node sits: 0 is root, 1 is section, 2 is a leaf (single page).
- `summary` is what the LLM reads during retrieval. It is a 2-3 sentence description of what is in this node.
- `children` holds the list of nodes one level below.
- `parent` points upward. It is excluded from `repr` to avoid infinite loops when printing.

The tree is just `DocumentNode` objects linked to each other through `children` and `parent`.

---

### Block 4: Summarise Helper Function

```python
def summarise(text: str, max_tokens: int = 120) -> str:
    prompt = (
        "Summarise the following financial document excerpt in 2-3 sentences. "
        "Be specific about numbers, section names, and key facts.\n\n"
        f"{text[:3000]}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

This function wraps a single LLM call. It takes any piece of text, sends it to GPT-4o-mini, and returns a short factual summary.

Key points:
- `text[:3000]` clips the input to avoid token limit errors on long pages.
- `temperature=0` ensures deterministic output (same input always gives same summary).
- This function is called many times during index construction: once per page, once per section, once for the root.
- The quality of these summaries directly determines retrieval accuracy. Better prompts here improve the whole system.

---

### Block 5: build_index Function (Core of the Notebook)

```python
def build_index(pages: List[str], section_size: int = 5) -> DocumentNode:
```

This is the most important function. It builds the entire tree in three passes.

**Pass 1 (Leaf nodes):** Loop over every page, create a `DocumentNode` at `level=2`, call `summarise()` on the page text. This is the most LLM-expensive step since it makes one API call per page.

**Pass 2 (Section nodes):** Take the leaf nodes in groups of `section_size` (default 5). For each group, create a `level=1` section node. Its summary is derived by joining the summaries of its child pages and passing that to `summarise()`. Child nodes get their `parent` pointer set here.

**Pass 3 (Root node):** One single `level=0` root node is created. Its summary comes from joining all section summaries. This summary describes the entire document in a few sentences.

The tree is returned as the `root` node. Everything else hangs off it through `children`.

Key point: the index is built **bottom-up** (pages first, then sections, then root) but it will be **queried top-down** (root first, then section, then page).

---

### Block 6: print_tree Function

```python
def print_tree(node: DocumentNode, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{node}")
    print(f"{prefix}  Summary: {node.summary[:100]}...")
    for child in node.children:
        print_tree(child, indent + 1)
```

A recursive function purely for inspection. It walks the tree depth-first and prints each node with its first 100 characters of summary. Useful for verifying the index was built correctly before running queries.

---

### Block 7: select_best_child Function

```python
def select_best_child(query: str, nodes: List[DocumentNode]) -> DocumentNode:
    options = "\n".join(
        f"[{i}] {n.node_id} (pages {n.page_range}): {n.summary}"
        for i, n in enumerate(nodes)
    )
    prompt = (
        f"Query: {query}\n\nNodes:\n{options}\n\n"
        f"Reply with only the integer index (0, 1, 2 ...) of the best node."
    )
    response = client.chat.completions.create(model="gpt-4o-mini", ...)
    idx = int(idx_str) if idx_str.isdigit() else 0
    return nodes[idx]
```

This is the decision-making engine of retrieval. It presents a list of node summaries to the LLM and asks it to pick the most relevant one for the given query. The LLM replies with just a number (e.g., `2`), which is parsed and used to index into the `nodes` list.

Key points:
- It is called twice during a query: once to pick a section, once to pick a page within that section.
- The prompt asks for only a number to make parsing reliable and keep token usage low.
- The fallback `if idx_str.isdigit() else 0` prevents crashes if the LLM replies unexpectedly.
- `max(0, min(idx, len(nodes) - 1))` clamps the index to valid range.

---

### Block 8: vectorless_query Function

```python
def vectorless_query(query: str, root: DocumentNode) -> str:
    section = select_best_child(query, root.children)
    page_node = select_best_child(query, section.children)
    answer_prompt = f"Question: {query}\n\nExcerpt:\n{page_node.raw_text[:3000]}"
    response = client.chat.completions.create(...)
    return response.choices[0].message.content.strip()
```

This is the complete retrieval-and-answer pipeline in three lines of logic:

1. Ask the LLM which section is most relevant.
2. Ask the LLM which page within that section is most relevant.
3. Send the raw text of that page to the LLM with the query and get an answer.

Three LLM calls total per query. The answer is grounded in actual page text, not a vector approximation.

---

### Block 9: Sample Queries

Four example queries are run through `vectorless_query`:
- Total revenue for the financial year
- Board of Directors members
- Dividend declaration and amount per share
- Auditor observations and qualifications

These are representative of what anyone analyzing an annual report would want to know. They test different kinds of retrieval: numeric (revenue), list-based (board members), conditional (dividend), and qualitative (auditor remarks).

---

### Block 10: build_traditional_index and traditional_query (Baseline)

```python
def build_traditional_index(pages, chunk_size=300):
    # Split every page into word-chunks of 300 words each
    # Fit a TF-IDF vectoriser over all chunks
    # Return chunks list, vectoriser, and the TF-IDF matrix

def traditional_query(query, chunks, vectoriser, matrix, top_k=2):
    # Transform query into TF-IDF vector
    # Compute cosine similarity against all chunk vectors
    # Pick top-k chunks
    # Send those chunks to the LLM with the query
```

This is the standard RAG baseline. Instead of a tree, it creates a flat list of text chunks and represents each as a TF-IDF vector. Retrieval is cosine similarity between the query vector and all chunk vectors.

Key points:
- No embedding model is used here. TF-IDF is a classical keyword-frequency approach, not semantic.
- The same LLM answers the question at the end, so the only difference between the two pipelines is how the relevant text is found.
- In a production baseline you would replace TF-IDF with dense embeddings (`text-embedding-3-small`), but the structure of the pipeline stays the same.

---

### Block 11: Side-by-Side Comparison

```python
v_answer = vectorless_query(query, root)
t_answer = traditional_query(query, chunks, vectoriser, matrix)
```

Runs both approaches on the same query and prints both answers. The purpose is to let you see concretely where each approach succeeds or fails. Vectorless RAG tends to give more precise answers when the answer lives in a specific section of a structured document. Traditional RAG can sometimes surface relevant context from multiple scattered locations that the tree traversal would miss.

---

## Architecture Summary Table

| Component | Vectorless RAG | Traditional RAG |
|---|---|---|
| Index structure | Tree (root, sections, pages) | Flat list of chunks |
| What is stored | Raw text + LLM summaries | Text chunks + TF-IDF vectors |
| Retrieval method | LLM-guided tree traversal | Cosine similarity |
| LLM calls per query | 3 (section select, page select, answer) | 1 (answer only) |
| Good for | Structured documents, reports | Large unstructured corpora |
| Limitation | Does not scale to millions of pages | Loses document structure |

---

## Key Concepts to Remember

**Why vectorless retrieval works on structured documents:** Annual reports, legal filings, and technical manuals have a natural hierarchy (chapters, sections, subsections). The hierarchical index mirrors this structure, so retrieval is navigating the document's own organization rather than hunting through a flat pool of chunks.

**Why the summary quality matters so much:** The LLM during retrieval never sees the raw page text. It only sees summaries. If a summary misses a key fact, the retriever will choose the wrong node and the final answer will be wrong. The summarisation prompt in Block 4 is therefore one of the most important parts of the system to tune.

**Why three LLM calls per query is acceptable:** In production, the index (and its summaries) is built once and cached. Retrieval then costs only three fast calls to a small model (gpt-4o-mini). This is often faster and cheaper than dense embedding retrieval at scale.

**Hybrid RAG as the production path:** Use the hierarchical index for structured section-level reasoning. Fall back to vector search for broad coverage when the tree traversal confidence is low. Most real-world RAG systems are hybrid.

---

## Exercises in the Notebook

1. Increase `section_size` to 10 pages and observe if retrieval accuracy changes for the dividend query.
2. Replace TF-IDF with dense embeddings (`text-embedding-3-small`) and compare latency.
3. Add a confidence score to `select_best_child` so the system falls back to vector search when uncertain.
4. Extend the tree to three levels: document, chapter, section, page.
