# Vectorless RAG: Hierarchical Index over Financial Documents

**Course:** Applied NLP / LLM Systems  
**Topic:** Retrieval-Augmented Generation without Vector Search  
**Document:** BSE Corporate Filing (Annual Report PDF)

---

## Part 1: What is Vectorless RAG?

To understand Vectorless RAG, you need to first understand what Traditional RAG does, and where it breaks down.

### Traditional RAG: How It Works

In the standard RAG pipeline, you do three things:

1. Take your document and split it into small text chunks (say, 300 words each).
2. Pass every chunk through an embedding model. This converts text into a list of numbers (a vector) that captures the meaning of that chunk.
3. When a query comes in, embed the query too, and find the chunks whose vectors are closest to the query vector. That closeness is called cosine similarity. The top matching chunks go into the LLM prompt as context.

This works at scale. But it has two well-known failure modes for structured documents like annual reports, legal filings, or technical manuals:

**Problem 1: Context gets sliced across chunk boundaries.** If a financial table starts on page 12 and ends on page 13, your chunker splits it. Neither chunk contains the full table. The retriever picks one chunk, the LLM sees half a table, and the answer is wrong or incomplete.

**Problem 2: The retriever is blind to document structure.** It has no idea that what it retrieved is from the "Auditor's Report" section versus the "Board's Report" section. All chunks are equal flat text. There is no awareness of hierarchy, section headings, or document organization.

### Vectorless RAG: The Core Idea

Vectorless RAG solves this by doing something very different. Instead of embedding chunks and doing similarity search, it builds a tree of the document and navigates that tree using the LLM's language understanding.

The key insight is: if every section and every page has a summary, you can ask the LLM "which of these summaries is most relevant to my query?" and it will give you a sensible answer. No vectors needed. This is pure language reasoning.

The tree looks like this:

```
Root Node (summary of entire document)
    Section 1 (pages 1-5, summary)
        Page 1 (raw text, summary)
        Page 2 (raw text, summary)
        Page 3 (raw text, summary)
        Page 4 (raw text, summary)
        Page 5 (raw text, summary)
    Section 2 (pages 6-10, summary)
        Page 6 (raw text, summary)
        ...
    Section 3 (pages 11-15, summary)
        ...
    Section 4 (pages 16-20, summary)
        ...
```

At query time, instead of searching, the system traverses this tree:

Step 1: Look at all section summaries. Ask the LLM: "Which section is most relevant to the query?" The LLM picks one section index.

Step 2: Inside that section, look at all page summaries. Ask the LLM again: "Which page is most relevant?" The LLM picks one page index.

Step 3: Take the raw text of that page and ask the LLM to answer the original query using that text.

No embeddings. No vector database. No cosine similarity. Just LLM reading summaries and picking the right path down the tree.

### When to Use Which

| Dimension | Traditional RAG | Vectorless RAG | Hybrid RAG |
|---|---|---|---|
| Storage | Vector database | In-memory tree | Both |
| Retrieval | Approximate nearest neighbor | LLM-guided tree traversal | Combined |
| Handles structure | Weak | Strong | Strong |
| Scalability | High | Medium | High |
| Best for | Large open-domain corpora | Structured documents like reports | Complex production systems |

The guiding principle is: match RAG complexity to problem complexity. If your document is a structured annual report, legal filing, or manual with clear sections, Vectorless RAG is the right tool. If you have millions of unstructured documents, you need vectors. If you need both, you build Hybrid RAG.

---

## Part 2: Code Architecture Deep Dive

The notebook has four main building blocks. Here is what each one does and why.

### Building Block 1: DocumentNode (the data structure)

This is the dataclass that represents a single node in the tree. Every page, every section, and the root is a DocumentNode.

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

What each field means:

`node_id` is the human-readable name. For a page, it is something like "page_7". For a section, "section_2". For the top of the tree, "root".

`level` tells you where in the tree this node sits. Level 0 is the root, level 1 is a section, level 2 is a leaf page. This is how the code knows how to print and navigate the tree.

`page_range` is a tuple like (6, 10) meaning this node covers pages 6 through 10. For a leaf node it is (7, 7), one page only. This is important for tracing back where an answer came from.

`raw_text` is the actual content. For leaf nodes, this is the extracted text of that page from the PDF. For section nodes, it is the concatenated text of all pages in that section. For the root, it is left empty because the root only stores the high-level summary, not all the text.

`summary` is the LLM-generated summary. This is the most important field for retrieval. The entire traversal logic reads summaries, not raw text. Summaries are concise (2-3 sentences) so the LLM can read all of them in one prompt and pick the right one.

`children` is the list of child nodes. For section nodes, children are the page nodes inside them. For the root, children are the section nodes. For leaf page nodes, children is empty.

`parent` is a back-reference so you can navigate upward in the tree if needed. It is marked `repr=False` so it does not appear when you print the node, which avoids circular output.

### Building Block 2: summarise() function

This is a simple helper that calls the OpenAI API and returns a 2-3 sentence summary of any text block.

```python
def summarise(text: str, max_tokens: int = 120) -> str:
```

It takes the first 3000 characters of the text (to stay within token limits), sends it to gpt-4o-mini with temperature 0 (deterministic output), and returns the summary string.

This function is called in three different contexts: once per page when building leaf nodes, once per section (summarizing the combined child summaries), and once for the root (summarizing all section summaries). Notice that as you go up the tree, the summary is always a summary of summaries, not a summary of raw text. This is intentional: it keeps the higher-level summaries abstract and topical rather than detail-heavy.

### Building Block 3: build_index() function

This is the most important function in the notebook. It constructs the entire tree bottom-up.

```python
def build_index(pages: List[str], section_size: int = 5) -> DocumentNode:
```

It takes the list of page texts and a section size parameter (default 5 pages per section), and returns the fully built root node.

The construction has three phases:

**Phase 1 (Leaf nodes):** Loop through every page. Create a DocumentNode at level 2. Call summarise() on its text. Append to leaf_nodes list. This is where most of the LLM calls happen: one per page.

**Phase 2 (Section nodes):** Group the leaf nodes in batches of section_size. For each group, create a DocumentNode at level 1. Its raw_text is all the child pages concatenated. Its summary is generated from the combined child summaries (not raw text). Wire the parent reference on each child. Append to section_nodes list.

**Phase 3 (Root node):** Create one DocumentNode at level 0. Its children are all the section nodes. Its summary is generated from all the section summaries concatenated. Wire the parent references on section nodes. Return the root.

After this function runs, you have a complete in-memory tree. Every node knows its parent and its children. Every node has a summary. The tree is ready to be queried.

### Building Block 4: The Query Engine (select_best_child and vectorless_query)

These two functions together implement the traversal-based retrieval.

`select_best_child()` is the low-level selector. It takes a query string and a list of nodes, formats all their summaries as numbered options, and asks the LLM to return a single integer: the index of the most relevant node. It then returns that node. There is also a safety guard: if the LLM returns something that is not a digit or is out of range, it defaults to index 0.

`vectorless_query()` is the high-level orchestrator. It calls select_best_child twice: first on the root's children to pick a section, then on that section's children to pick a page. Then it builds a final answer prompt using the raw text of the chosen page and makes one more LLM call to generate the answer.

In total, answering one query makes exactly 3 LLM calls: one to pick the section, one to pick the page, one to generate the answer.

---

## Part 3: Code Block-by-Block Walkthrough

### Cell 1: Installation

```python
%pip install pypdf llama-index llama-index-llms-openai openai
```

`pypdf` is for reading PDF files and extracting text page by page.  
`openai` is the official OpenAI Python SDK for making API calls.  
`llama-index` is installed but not used in the custom implementation sections. It is available if you want to compare with LlamaIndex's built-in hierarchical retrieval.

### Cell 2: Imports and API Setup

```python
import os
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from pypdf import PdfReader
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."
client = OpenAI()
```

Standard library imports first: `os` for environment variables, `requests` for downloading the PDF, `Path` for filesystem paths, `dataclass` and `field` for the DocumentNode class, `List` and `Optional` for type hints.

Then the third-party imports: `PdfReader` from pypdf to extract page text, `OpenAI` client to make API calls.

The API key is set as an environment variable. Replace it with your own key. The `client = OpenAI()` line picks up the key automatically from the environment.

### Cell 3: PDF Download and Page Extraction

```python
PDF_URL = "https://www.bseindia.com/..."
PDF_PATH = Path("annual_report.pdf")

if not PDF_PATH.exists():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(PDF_URL, headers=headers)
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)

reader = PdfReader(str(PDF_PATH))
pages = [page.extract_text() or "" for page in reader.pages]
```

The download block checks if the file already exists before downloading. This is a caching pattern: re-running the notebook does not re-download.

The `User-Agent` header is added because BSE India's server sometimes blocks requests that look like bots without a browser-like user agent string.

`response.raise_for_status()` will throw an exception if the download failed (404, 403, etc.) instead of silently writing an empty file.

The `PdfReader` loop extracts text from each page. The `or ""` fallback handles scanned pages or image-only pages that return None from `extract_text()`. After this cell, `pages` is a Python list where `pages[0]` is page 1 text, `pages[1]` is page 2 text, and so on.

### Cell 4: DocumentNode Dataclass

Covered in architecture section above. The key thing to remember in class: this is not a LlamaIndex Node or a LangChain Document. It is a plain Python dataclass. The tree structure is built entirely with standard Python objects. This makes the concept clear without framework abstraction getting in the way.

### Cell 5: summarise() Helper

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

The prompt asks for specificity about numbers, section names, and key facts. This is important for financial documents: a vague summary like "this page discusses financial information" is useless for retrieval. A good summary says "This page contains the profit and loss statement for FY2023, reporting total revenue of Rs. 142 crore and net profit of Rs. 18 crore."

`temperature=0` ensures the summary is consistent. If you re-run the same page twice, you get the same summary. This is important because the summaries are used for navigation, not generation.

`max_tokens=120` is the default for page summaries. The root node uses `max_tokens=200` because it needs to cover the entire document.

### Cell 6: build_index() Function

Covered in the architecture section. For classroom context, point out two things:

First, this is an offline process. You run build_index once, store the tree in memory (or serialize it), and then query it multiple times. The expensive part is the page summarization: 20 pages means 20 LLM calls just to build the index.

Second, the section summarization does not re-read raw text. It reads the child summaries. This is a compression chain: page raw text gets compressed to a page summary, then multiple page summaries get compressed to a section summary, then all section summaries get compressed to a root summary. Each level is a progressive abstraction.

### Cell 7: print_tree() Helper

```python
def print_tree(node: DocumentNode, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{node}")
    print(f"{prefix}  Summary: {node.summary[:100]}...")
    for child in node.children:
        print_tree(child, indent + 1)
```

This is a recursive tree printer. It calls itself on each child with increasing indentation. It prints the node's `__repr__` (which shows node_id, level, page_range, and number of children) followed by the first 100 characters of the summary.

Run this after building the index to visually confirm the tree structure. It is a good classroom demo: students can see the hierarchy and the summaries without digging into raw text.

### Cell 8: select_best_child() Function

```python
def select_best_child(query: str, nodes: List[DocumentNode]) -> DocumentNode:
    options = "\n".join(
        f"[{i}] {n.node_id} (pages {n.page_range}): {n.summary}"
        for i, n in enumerate(nodes)
    )
    prompt = (
        f"You are navigating a document index. "
        f"Choose the single most relevant node for the query below.\n\n"
        f"Query: {query}\n\n"
        f"Nodes:\n{options}\n\n"
        f"Reply with only the integer index (0, 1, 2 ...) of the best node."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    idx_str = response.choices[0].message.content.strip()
    idx = int(idx_str) if idx_str.isdigit() else 0
    idx = max(0, min(idx, len(nodes) - 1))
    return nodes[idx]
```

The prompt lists all child nodes with their index, page range, and summary. It explicitly instructs the model to return only an integer. `max_tokens=5` enforces this: the model cannot write an explanation even if it tries.

The safety guard at the end (`max(0, min(idx, len(nodes) - 1))`) clamps the returned index to the valid range. If somehow the model returns 99 but there are only 4 nodes, this prevents an IndexError.

### Cell 9: vectorless_query() Function

```python
def vectorless_query(query: str, root: DocumentNode) -> str:
    section = select_best_child(query, root.children)
    page_node = select_best_child(query, section.children)

    answer_prompt = (
        f"Use the document excerpt below to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Excerpt (page {page_node.page_range[0]}):\n{page_node.raw_text[:3000]}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        max_tokens=300,
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

The traversal is exactly two hops: root to section, section to page. The final answer call uses the raw text of the selected page, not the summary. Summaries are only for navigation. Answers come from raw text. This is an important distinction.

The `page_node.raw_text[:3000]` truncates to 3000 characters as a safety measure for token limits. In production you would want to handle this more carefully.

### Cells 10-13: Sample Queries

Four representative queries are run against the annual report:

- Total revenue or turnover for the financial year
- Board of Directors members
- Dividend declared and amount per share
- Key observations from the auditors

Each query demonstrates a different section of a financial document. Run these in class to show live traversal: the print statements inside `vectorless_query` will show which section and which page was selected for each query, making the traversal path visible.

### Cells 14-15: Traditional RAG Baseline

```python
def build_traditional_index(pages: List[str], chunk_size: int = 300):
    # splits pages into 300-word chunks
    # fits a TF-IDF vectorizer over all chunks
    # returns chunks list, vectorizer, and TF-IDF matrix

def traditional_query(query: str, chunks, vectoriser, matrix, top_k: int = 2) -> str:
    # embeds the query with TF-IDF
    # finds top-k chunks by cosine similarity
    # sends those chunks to LLM as context for answering
```

Note that this baseline uses TF-IDF instead of dense vector embeddings. TF-IDF is keyword-based: it scores chunks by word frequency match against the query. Real production Traditional RAG would use `text-embedding-3-small` or similar. The TF-IDF version is used here to avoid a second embedding API cost and to keep the comparison focused on architecture rather than model quality.

The comparison is still valid conceptually: flat chunks with similarity search versus hierarchical tree with LLM traversal. The point being demonstrated is the structural awareness difference, not embedding quality.

### Cell 16: Side-by-Side Comparison

Runs the same query through both systems and prints both answers. This is the core classroom demo moment. Students see how the two systems pick different context (a traversed page versus a flat chunk) and may produce different quality answers depending on how well the relevant information is structured in the document.

---

## Part 4: Key Takeaways for the Classroom

**Why Vectorless RAG matters:** Most enterprise documents are structured. Annual reports, contracts, medical records, technical manuals all have a hierarchy. Traditional RAG ignores that hierarchy. Vectorless RAG exploits it.

**The tradeoff:** Building the index is expensive at construction time because you summarize every page. But querying is fast and predictable: always 3 LLM calls, always the same traversal path, always traceable.

**The scalability ceiling:** This approach works well for documents up to a few hundred pages. For very large corpora, the section-level traversal becomes too slow and the in-memory tree becomes too large. That is where Hybrid RAG comes in: use hierarchical traversal for structural reasoning on individual documents, and vector search for broad coverage across a large corpus.

**The exercises at the end of the notebook are worth doing:**

Exercise 1 asks students to change section_size from 5 to 10. Larger sections mean fewer sections, fewer choices at the section level, potentially less precise routing. Students should observe whether the dividend query still lands on the right page.

Exercise 2 asks students to replace TF-IDF with actual dense embeddings using `text-embedding-3-small`. This makes the Traditional RAG baseline more realistic and shows the latency difference between embedding-based and traversal-based retrieval.

Exercise 3 asks for a confidence score in select_best_child so the system can fall back to vector search when the LLM is uncertain. This is the conceptual step toward Hybrid RAG.

Exercise 4 asks for a three-level tree: document, chapter, section, page. This makes the traversal more powerful but also more expensive to build.
