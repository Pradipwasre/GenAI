# 📡 Multi-Agent RAG : Bharti Airtel FY2025
## Complete Code Structure & Study Guide
> *Read this file end-to-end and you will understand every concept, every line of code, and every design decision in the notebook.*

---

##  Table of Contents

1. [What is RAG? — The Core Idea](#1-what-is-rag--the-core-idea)
2. [Architecture Overview — All 5 Patterns](#2-architecture-overview--all-5-patterns)
3. [Tech Stack & Package Map](#3-tech-stack--package-map)
4. [Section 1 — Environment Setup](#4-section-1--environment-setup)
5. [Section 2 — Base RAG Pipeline](#5-section-2--base-rag-pipeline)
6. [Section 3 — Network Multi-Agent RAG](#6-section-3--network-multi-agent-rag)
7. [Section 4 — Supervisor Multi-Agent RAG](#7-section-4--supervisor-multi-agent-rag)
8. [Section 5 — Hierarchical Multi-Agent RAG](#8-section-5--hierarchical-multi-agent-rag)
9. [Section 6 — Hybrid Live + PDF RAG](#9-section-6--hybrid-live--pdf-rag)
10. [Section 7 — Student Exercises](#10-section-7--student-exercises)
11. [Pattern Comparison Cheat Sheet](#11-pattern-comparison-cheat-sheet)
12. [Common Errors & Fixes](#12-common-errors--fixes)
13. [Key Concepts Glossary](#13-key-concepts-glossary)

---

## 1. What is RAG? — The Core Idea

**RAG = Retrieval-Augmented Generation**

Without RAG, an LLM only knows what it was trained on.  
With RAG, you give it a *private knowledge base* — in our case, Airtel's annual report PDF — and let it answer questions from that source at runtime.

```
WITHOUT RAG:
  User: "What was Airtel's FY2025 EBITDA?"
  LLM:  "I don't know / makes something up" ❌

WITH RAG:
  User: "What was Airtel's FY2025 EBITDA?"
  System: 1. Search PDF for relevant passages  →  "Found: EBITDA ₹72,345 Cr..."
          2. Send those passages + question to LLM
  LLM:  "According to the FY2025 report, EBITDA was ₹72,345 Cr" ✅
```

**The 5-step RAG pipeline:**

```
STEP 1: LOAD      → Read PDF pages into Document objects
STEP 2: SPLIT     → Break pages into overlapping text chunks
STEP 3: EMBED     → Convert each chunk to a vector (numbers)
STEP 4: STORE     → Save vectors in FAISS for fast search
STEP 5: RETRIEVE  → At query time, find top-K similar chunks
                    → Pass chunks + question to LLM → Get answer
```

---

## 2. Architecture Overview — All 5 Patterns

### Pattern A: Base RAG (Section 2)
Simplest form — one PDF, one retriever, one LLM.

```
PDF ──► Chunks ──► Vectors ──► FAISS
                                 │
User Query ──► Embed Query ──► Search FAISS
                                 │
                             Top 5 Chunks
                                 │
                        [Chunks + Query] ──► LLM ──► Answer
```

---

### Pattern B: Network Multi-Agent (Section 3)
Sequential pipeline — three agents always run in order.

```
User Query
    │
    ▼
┌──────────────────┐
│  Research Agent  │  ← Retrieves raw facts from PDF (FAISS search)
│  Prompt: "Extract│
│  only facts, no  │
│  interpretation" │
└────────┬─────────┘
         │  state["raw_findings"]
         ▼
┌──────────────────┐
│  Analysis Agent  │  ← Receives raw_findings, identifies trends
│  Prompt: "Find   │
│  growth rates,   │
│  ratios, trends" │
└────────┬─────────┘
         │  state["analysis"]
         ▼
┌──────────────────┐
│   Writer Agent   │  ← Produces student-friendly final report
│  Prompt: "Write  │
│  for MBA class,  │
│  plain English"  │
└────────┬─────────┘
         │  state["final_report"]
         ▼
    Final Answer
```

**Shared State** carries data between agents:
```python
NetworkAgentState = {
    "query":        str,   # original question (never changes)
    "raw_findings": str,   # filled by Research Agent
    "analysis":     str,   # filled by Analysis Agent
    "final_report": str    # filled by Writer Agent
}
```

---

### Pattern C: Supervisor Multi-Agent (Section 4)
Intelligent router — LLM decides which specialist to call.

```
                    ┌──────────────────────────┐
                    │     Supervisor Agent      │
User Query ──────►  │  (LLM reads the query    │
                    │   and picks a specialist) │
                    └───┬──────────┬────────┬──┘
                        │          │        │
              "numbers" │  "trends"│  "sum" │
                        ▼          ▼        ▼
               ┌──────────┐ ┌──────────┐ ┌──────────┐
               │ Research │ │ Analysis │ │  Writer  │
               │Specialist│ │Specialist│ │Specialist│
               │          │ │          │ │          │
               │Tool:     │ │Tools:    │ │Tools:    │
               │retrieve_ │ │retrieve_ │ │retrieve_ │
               │financial_│ │financial_│ │financial_│
               │data      │ │data +    │ │data +    │
               │          │ │analyze_  │ │write_    │
               │          │ │trends    │ │summary   │
               └──────────┘ └──────────┘ └──────────┘
                        │          │        │
                        └────┬─────┘────────┘
                             ▼
                    Supervisor compiles
                    final answer
```

---

### Pattern D: Hierarchical Multi-Agent (Section 5)
Multi-level supervision — parent delegates to year-specific children.

```
User: "Summarize Airtel across FY2022–FY2025"
                          │
                          ▼
            ┌─────────────────────────┐
            │   PARENT SUPERVISOR     │
            │  "Delegate to each year │
            │   agent, then aggregate"│
            └──┬──────┬──────┬───┬───┘
               │      │      │   │
               ▼      ▼      ▼   ▼
           ┌──────┐┌──────┐┌──────┐┌──────┐
           │FY2022││FY2023││FY2024││FY2025│
           │Agent ││Agent ││Agent ││Agent │
           │      ││      ││      ││      │
           │Tool: ││Tool: ││Tool: ││Tool: │
           │retri-││retri-││retri-││retri-│
           │eve_  ││eve_  ││eve_  ││eve_  │
           │FY2022││FY2023││FY2024││FY2025│
           │_data ││_data ││_data ││_data │
           └──────┘└──────┘└──────┘└──────┘
               │      │      │   │
               └──────┴──────┴───┘
                          │
               Parent aggregates all 4 outputs
                          │
              Consolidated 4-Year Report
```

---

### Pattern E: Hybrid Live + PDF (Section 6)
Two data sources merged in one prompt.

```
              ┌─────────────────┐     ┌──────────────────┐
              │  PDF (RAG)      │     │   Screener.in    │
              │  Historical:    │     │   Live/Current:  │
              │  • FY25 Revenue │     │   • Stock Price  │
              │  • EBITDA       │     │   • Market Cap   │
              │  • Net Profit   │     │   • P/E Ratio    │
              │  • ARPU         │     │   • Book Value   │
              └────────┬────────┘     └────────┬─────────┘
                       │                       │
                       └──────────┬────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │    hybrid_query()        │
                    │                         │
                    │  PROMPT STRUCTURE:      │
                    │  ═══ LIVE DATA ═══      │
                    │  {live_context}         │
                    │                         │
                    │  ═══ PDF DATA ═══       │
                    │  {pdf_context}          │
                    │                         │
                    │  ═══ QUESTION ═══       │
                    │  {question}             │
                    └─────────────────────────┘
                                  │
                                  ▼
                        Groq LLM ──► Hybrid Answer
```

---

## 3. Tech Stack & Package Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    PACKAGE PURPOSE MAP                          │
├──────────────────────────┬──────────────────────────────────────┤
│ Package                  │ What it does in this project         │
├──────────────────────────┼──────────────────────────────────────┤
│ langchain >= 0.3         │ Core: prompts, chains (LCEL |pipes)  │
│ langchain-community>=0.3 │ PyPDFLoader, FAISS vector store      │
│ langchain-groq >= 0.2    │ ChatGroq LLM wrapper for Groq API    │
│ langchain-huggingface    │ HuggingFaceEmbeddings (local, free)  │
│ langchain-core >= 0.3    │ Tool decorator, messages, runnables  │
├──────────────────────────┼──────────────────────────────────────┤
│ langgraph >= 0.3         │ StateGraph, START, END for agent DAG │
│ langgraph-supervisor     │ create_supervisor(), agent routing   │
├──────────────────────────┼──────────────────────────────────────┤
│ faiss-cpu >= 1.8         │ Fast vector similarity search        │
│ sentence-transformers    │ Embedding model (all-MiniLM-L6-v2)   │
│ pypdf >= 4.0             │ Extract text from PDF pages          │
├──────────────────────────┼──────────────────────────────────────┤
│ beautifulsoup4 >= 4.12   │ Parse HTML from Screener.in          │
│ requests >= 2.31         │ HTTP GET request to Screener.in      │
│ python-dotenv >= 1.0     │ Load GROQ_API_KEY from .env file     │
└──────────────────────────┴──────────────────────────────────────┘
```

**Import Map** — which import comes from which package:

```python
# PDF Loading
from langchain_community.document_loaders import PyPDFLoader

# Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings + Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM + Prompt + Chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Agent Graphs
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# Typing
from typing import TypedDict, Optional

# Standard Library
import os, requests, json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
```

---

## 4. Section 1 — Environment Setup

### What happens here
Load secrets, verify the PDF exists, set model names.

### 1.1 — Install Packages

```python
%pip install -q \
    langchain>=0.3 \
    langchain-community>=0.3 \
    langchain-groq>=0.2 \
    langchain-huggingface>=0.1 \
    langgraph>=0.3 \
    langgraph-supervisor>=0.0.9 \
    faiss-cpu>=1.8 \
    pypdf>=4.0 \
    sentence-transformers>=3.0 \
    beautifulsoup4>=4.12 \
    requests>=2.31 \
    python-dotenv>=1.0
```

> **Why `%pip` not `!pip`?** The `%pip` magic ensures packages are installed in the *same Python kernel* that the notebook is running in. `!pip` can install into the wrong environment.

### 1.2 — Environment Variables & Config

```python
import os
from dotenv import load_dotenv

load_dotenv()                          # reads .env file in current directory

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# .env file must contain:  GROQ_API_KEY=gsk_xxxxxxxxxxxxx

PDF_PATH = "/Users/pradipwasre/Desktop/GenAI/LangGraph/Multi-Agent RAG/Bharti Airtel.pdf"

LLM_MODEL   = "llama-3.1-8b-instant"   # Groq hosted, free tier, very fast
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs LOCALLY on CPU
```

**Model choices on Groq (2026):**

| Model | Speed | Best For |
|-------|-------|----------|
| `llama-3.1-8b-instant` | Fastest ⚡ | Demos, free tier |
| `llama-3.3-70b-versatile` | Medium | Better reasoning |
| `mixtral-8x7b-32768` | Medium | Long context |

---

## 5. Section 2 — Base RAG Pipeline

### The 5 sub-steps with full code

---

### 2.1 — Load PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path=PDF_PATH)
raw_docs = loader.load()
```

**What `loader.load()` returns:**
```
raw_docs = [
    Document(
        page_content="Bharti Airtel Limited\nAnnual Report FY2025...",
        metadata={"source": "/path/Bharti Airtel.pdf", "page": 0}
    ),
    Document(
        page_content="Financial Highlights...",
        metadata={"source": "/path/Bharti Airtel.pdf", "page": 1}
    ),
    ...
]
```

> Each `Document` = one PDF page. A 300-page PDF gives you 300 Documents.

---

### 2.2 — Split Documents into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # each chunk ≈ 1000 characters
    chunk_overlap=200,     # last 200 chars of chunk N = first 200 chars of chunk N+1
    length_function=len,
    separators=["\n\n", "\n", " ", ""]   # tries these split points in order
)

chunks = splitter.split_documents(raw_docs)
```

**Why overlap?** Without it, a sentence spanning a chunk boundary would be cut in half and context would be lost.

```
CHUNK 1:  "...Airtel's total revenue for FY2025 grew to ₹1,51,781 Cr"
           ←─────────────────────────────────────── overlap ──────►
CHUNK 2:                "...₹1,51,781 Cr, driven by India Mobile segment..."
```

**`separators` priority:** The splitter tries `\n\n` (paragraph) first, then `\n` (line), then space, then character — this keeps semantic units intact.

---

### 2.3 — Create Embeddings + FAISS Index

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},           # run on CPU, no GPU needed
    encode_kwargs={"normalize_embeddings": True}  # L2-normalize → cosine similarity
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("airtel_faiss_index")  # save to disk to avoid re-embedding
```

**What is an embedding?**
```
Text: "Airtel revenue FY2025"
  ↓  (embedding model)
Vector: [0.23, -0.11, 0.54, 0.09, ..., -0.33]  ← 384 numbers
```
Chunks with similar meaning get vectors that are *close together* in 384-dimensional space.

**FAISS** (Facebook AI Similarity Search) stores all chunk vectors and finds the closest ones to any query vector in milliseconds.

```
Query: "What was Airtel's revenue?"
  ↓  embed
Query vector: [0.21, -0.09, 0.51, ...]

FAISS finds top-5 closest chunk vectors
  → returns those 5 chunks as context
```

---

### 2.4 — Build Retriever and LLM

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Retriever wraps FAISS with a clean .invoke() interface
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}     # return top 5 most similar chunks
)

# Groq LLM — cloud-hosted, fast inference
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,           # low temperature = more factual, less creative
    api_key=GROQ_API_KEY
)

# Prompt Template — {context} and {question} are placeholders filled at runtime
rag_prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant specializing in Bharti Airtel's annual reports.
Use ONLY the context below to answer the question. Be precise and cite relevant figures.

Context:
{context}

Question: {question}

Answer:
""")

# format_docs converts List[Document] → a single formatted string
def format_docs(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

# THE RAG CHAIN — LangChain Expression Language (LCEL) pipe syntax
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

**How the LCEL chain works step by step:**

```
rag_chain.invoke("What was Airtel's FY2025 revenue?")

Step 1: {"context": retriever | format_docs, "question": RunnablePassthrough()}
        → retriever.invoke(query) → [Doc1, Doc2, Doc3, Doc4, Doc5]
        → format_docs([Doc1...Doc5]) → "[Page 12]\n text...\n\n[Page 45]\n text..."
        → RunnablePassthrough() just passes the query string through unchanged
        → Result: {"context": "...", "question": "What was Airtel's FY2025 revenue?"}

Step 2: | rag_prompt
        → Fills {context} and {question} placeholders
        → Result: A ChatPromptValue ready to send to LLM

Step 3: | llm
        → Sends to Groq API
        → Result: AIMessage(content="Airtel's FY2025 revenue was ₹1,51,781 Cr...")

Step 4: | StrOutputParser()
        → Extracts just the .content string from AIMessage
        → Result: "Airtel's FY2025 revenue was ₹1,51,781 Cr..."
```

---

### 2.5 — Test the Base RAG

```python
test_queries = [
    "Summarize the AGM notice from the annual report.",
    "What were Airtel's total revenues for FY2025?",
    "Who are the board members mentioned in the annual report?",
]

for q in test_queries:
    answer = rag_chain.invoke(q)
    print(answer)
```

---

## 6. Section 3 — Network Multi-Agent RAG

### What is a LangGraph StateGraph?

A **StateGraph** is a directed graph where:
- **Nodes** = functions that process data (your agents)
- **Edges** = connections that define execution order
- **State** = a shared dictionary passed between all nodes

```
StateGraph visually:

  START
    │
    ▼
[research_agent_node]   ← Node 1: reads state["query"], writes state["raw_findings"]
    │
    ▼
[analysis_agent_node]   ← Node 2: reads state["raw_findings"], writes state["analysis"]
    │
    ▼
[writer_agent_node]     ← Node 3: reads state["analysis"], writes state["final_report"]
    │
    ▼
   END
```

---

### 3.1 — Define Shared State Schema

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

class NetworkAgentState(TypedDict):
    query:        str            # INPUT: user's original question
    raw_findings: Optional[str]  # filled by Research Agent
    analysis:     Optional[str]  # filled by Analysis Agent
    final_report: Optional[str]  # filled by Writer Agent
```

> `TypedDict` is Python's way of giving a dict a strict schema — it tells other developers (and the type checker) exactly what keys are allowed and what types they should be.

---

### 3.2 — Define Agent Node Functions

Each node is a **pure function**: takes state in, returns updated state out.

```python
# ── RESEARCH AGENT ──────────────────────────────────────────────
research_prompt = ChatPromptTemplate.from_template("""
You are a financial research agent. Retrieve and list factual data from the context.
Focus on numbers, dates, KPIs, and direct statements from the annual report.
Do NOT interpret — only extract facts.

Context:
{context}

Query: {query}

Raw Findings:
""")

def research_agent_node(state: NetworkAgentState) -> NetworkAgentState:
    docs     = retriever.invoke(state["query"])     # FAISS search
    context  = format_docs(docs)                    # format chunks as string
    chain    = research_prompt | llm | StrOutputParser()
    findings = chain.invoke({"context": context, "query": state["query"]})
    return {**state, "raw_findings": findings}      # spread + overwrite one key
#          ↑ This keeps ALL existing state fields and just updates raw_findings


# ── ANALYSIS AGENT ──────────────────────────────────────────────
analysis_prompt = ChatPromptTemplate.from_template("""
You are a financial analysis agent. Given the raw research findings below,
identify key trends, growth rates, ratios, and business insights.
Compare figures where possible and highlight what is significant for investors.

Raw Findings:
{raw_findings}

Original Query: {query}

Analysis:
""")

def analysis_agent_node(state: NetworkAgentState) -> NetworkAgentState:
    chain    = analysis_prompt | llm | StrOutputParser()
    analysis = chain.invoke({
        "raw_findings": state["raw_findings"],   # reads from previous agent
        "query":        state["query"]
    })
    return {**state, "analysis": analysis}


# ── WRITER AGENT ────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_template("""
You are a financial writer creating educational content for MBA students.
Using the analysis below, write a clear, structured summary that:
- Uses simple language (avoid jargon)
- Includes key numbers and what they mean
- Ends with 2-3 key takeaways
- Is suitable for a classroom presentation

Analysis:
{analysis}

Original Query: {query}

Student-Friendly Summary:
""")

def writer_agent_node(state: NetworkAgentState) -> NetworkAgentState:
    chain  = writer_prompt | llm | StrOutputParser()
    report = chain.invoke({
        "analysis": state["analysis"],
        "query":    state["query"]
    })
    return {**state, "final_report": report}
```

**Key design pattern — `{**state, "key": value}`:**
```python
# This is Python dict spreading:
state = {"query": "q", "raw_findings": None, "analysis": None, "final_report": None}

# After research agent:
return {**state, "raw_findings": "Revenue was ₹1.5L Cr..."}
# Equivalent to:
# {"query": "q", "raw_findings": "Revenue was ₹1.5L Cr...", "analysis": None, "final_report": None}
```

---

### 3.3 — Build and Compile the Graph

```python
from langgraph.graph import StateGraph, START, END

network_builder = StateGraph(NetworkAgentState)   # pass the state schema

# Register nodes (name → function)
network_builder.add_node("research_agent", research_agent_node)
network_builder.add_node("analysis_agent", analysis_agent_node)
network_builder.add_node("writer_agent",   writer_agent_node)

# Define edges (execution order)
network_builder.add_edge(START,            "research_agent")  # entry point
network_builder.add_edge("research_agent", "analysis_agent")  # sequential
network_builder.add_edge("analysis_agent", "writer_agent")    # sequential
network_builder.add_edge("writer_agent",   END)               # exit point

# Compile creates an executable app (validates the graph structure)
network_graph = network_builder.compile()
```

---

### 3.4 — Run the Network Pipeline

```python
result = network_graph.invoke({"query": "What is Airtel's EBITDA in FY2025?"})

# result is the final state dict after all 3 agents have run:
print(result["raw_findings"])   # from Research Agent
print(result["analysis"])       # from Analysis Agent
print(result["final_report"])   # from Writer Agent
```

---

## 7. Section 4 — Supervisor Multi-Agent RAG

### What is the Supervisor Pattern?

In the Network pattern, agents always run in fixed order.  
In the Supervisor pattern, an LLM **decides at runtime** which agent to call — and can call agents in any order, multiple times, or skip them entirely.

```
Supervisor's decision loop:
  1. Read the query
  2. Think: "Which specialist handles this?"
  3. Call that specialist (tool call / handoff)
  4. Read specialist's response
  5. Think: "Is the query fully answered?"
     → Yes: produce final answer, FINISH
     → No:  call another specialist
```

---

### 4.1 — Define Tool Functions

Tools are regular Python functions decorated with `@tool`.  
The LLM sees the function's **name** and **docstring** to decide when to use it.

```python
from langchain_core.tools import tool

@tool
def retrieve_financial_data(query: str) -> str:
    """
    Retrieve specific financial data, numbers, metrics, KPIs, and figures
    from Bharti Airtel's FY2025 Annual Report PDF.
    Use this for questions about revenue, EBITDA, capex, subscribers, etc.
    """
    #  ↑ This docstring is what the LLM reads to decide when to call this tool
    docs    = retriever.invoke(query)
    context = format_docs(docs)
    prompt  = f"Extract specific financial numbers and metrics for: {query}\n\nContext:\n{context}\n\nList only the relevant numbers:"
    return llm.invoke(prompt).content


@tool
def analyze_financial_trends(data: str) -> str:
    """
    Analyze financial trends, growth rates, year-over-year changes,
    and business performance patterns from Airtel data.
    Use this after gathering raw financial data to identify trends.
    """
    prompt = f"Analyze for trends, growth rates, and patterns:\n{data}\n\nTrend Analysis:"
    return llm.invoke(prompt).content


@tool
def write_student_summary(content: str) -> str:
    """
    Write a clear, student-friendly summary of financial findings.
    Use this to produce final readable outputs for educational purposes.
    """
    prompt = f"Write a clear MBA student summary:\n{content}\n\nStudent Summary:"
    return llm.invoke(prompt).content
```

---

### 4.2 — Create Specialist ReAct Agents

**ReAct** = **Re**asoning + **Act**ing. The agent loops:
1. **Think** (reasoning step — what should I do?)
2. **Act** (call a tool)
3. **Observe** (get tool result)
4. Back to **Think** — repeat until done

```python
from langgraph.prebuilt import create_react_agent

# Research specialist — ONLY retrieves numbers
research_specialist = create_react_agent(
    model=llm,
    tools=[retrieve_financial_data],          # only 1 tool available
    name="research_specialist",
    prompt=(
        "You are a financial research specialist. "
        "Your job is to retrieve accurate financial figures from Airtel's annual report. "
        "Always use the retrieve_financial_data tool. Be precise and cite page numbers."
    )
)

# Analysis specialist — retrieves THEN analyzes
analysis_specialist = create_react_agent(
    model=llm,
    tools=[retrieve_financial_data,           # step 1: get data
           analyze_financial_trends],          # step 2: analyze it
    name="analysis_specialist",
    prompt=(
        "You are a financial analysis specialist. "
        "First retrieve data, then analyze trends and growth patterns. "
        "Focus on year-over-year changes and business performance insights."
    )
)

# Writer specialist — retrieves THEN summarizes
writer_specialist = create_react_agent(
    model=llm,
    tools=[retrieve_financial_data,           # step 1: get data
           write_student_summary],             # step 2: write summary
    name="writer_specialist",
    prompt=(
        "You are an educational content writer. "
        "Retrieve relevant information and write clear summaries for students. "
        "Always produce structured, easy-to-read content with key takeaways."
    )
)
```

---

### 4.3 — Create the Supervisor

```python
from langgraph_supervisor import create_supervisor

supervisor_prompt = """
You are a supervisor managing a team of Bharti Airtel financial analysts.
Route each query to the most appropriate specialist:

• research_specialist  → queries about specific numbers, figures, KPIs, revenues, profits
• analysis_specialist  → queries about trends, comparisons, growth, year-over-year changes
• writer_specialist    → queries asking for summaries, overviews, explanations, AGM notices

After the specialist responds, compile their output into a final answer.
If the query requires multiple agents, coordinate them sequentially.
"""

supervisor_workflow = create_supervisor(
    agents=[research_specialist, analysis_specialist, writer_specialist],
    model=llm,
    prompt=supervisor_prompt,
)

supervisor_app = supervisor_workflow.compile()
```

**What `create_supervisor` does internally:**
1. Creates a new `StateGraph` with a `messages` list as state
2. Adds each specialist agent as a node
3. Adds a supervisor node (the LLM that routes)
4. Adds conditional edges: supervisor → specialist (based on LLM's routing decision)
5. Adds edges: specialist → supervisor (to report back)
6. Terminates when supervisor returns `FINISH`

---

### 4.4 — Run the Supervisor

```python
from langchain_core.messages import HumanMessage

result = supervisor_app.invoke(
    {"messages": [HumanMessage(content="Compare FY2025 revenue with FY2024.")]},
    config={"recursion_limit": 20}  # safety cap: stops after 20 LLM calls max
)

# Get the final answer (last message in the conversation)
final_messages = [m for m in result["messages"] if hasattr(m, 'content')]
print(final_messages[-1].content)
```

**Why `recursion_limit`?** Without it, if the supervisor LLM never decides to stop (never returns `FINISH`), the loop runs forever. `recursion_limit=20` means after 20 steps, LangGraph raises an error automatically.

---

## 8. Section 5 — Hierarchical Multi-Agent RAG

### Why Hierarchical?

When data has a **natural hierarchy** (years, departments, products), you can:
- Assign one specialist per domain
- Let a parent coordinate them
- Scale without changing the parent's logic

```
Adding a new year? Just add one child agent — parent code unchanged.
```

---

### 5.1 — Year-Specific Tool Factory

```python
def make_year_tool(year: str):
    """
    Factory: creates a different tool for each year.
    Each tool has a UNIQUE name (required by LangGraph).
    """
    @tool(name=f"retrieve_{year}_data")   # e.g., "retrieve_FY2022_data"
    def year_tool(query: str) -> str:
        # Appends the year to the query so FAISS retrieves year-relevant chunks
        year_query = f"{query} {year} fiscal year"
        docs       = retriever.invoke(year_query)
        context    = format_docs(docs)

        prompt = f"""From the annual report context, extract data specifically for {year}.
Focus on: revenue, EBITDA, net profit, subscribers, ARPU, capex.

Context:
{context}

Query: {query}

{year} Data Summary:"""
        return llm.invoke(prompt).content

    year_tool.__doc__ = f"Retrieve Bharti Airtel financial data specifically for {year}."
    return year_tool

# Create 4 tools — each scoped to one year
tool_fy22 = make_year_tool("FY2022")  # tool name: retrieve_FY2022_data
tool_fy23 = make_year_tool("FY2023")  # tool name: retrieve_FY2023_data
tool_fy24 = make_year_tool("FY2024")  # tool name: retrieve_FY2024_data
tool_fy25 = make_year_tool("FY2025")  # tool name: retrieve_FY2025_data
```

**Why a factory function?** If you wrote 4 separate `@tool` functions, you'd repeat yourself. The factory generates them programmatically — add `FY2026` by calling `make_year_tool("FY2026")`.

---

### 5.2 — Create Child Agents

```python
def make_year_agent(year: str, year_tool):
    return create_react_agent(
        model=llm,
        tools=[year_tool],                  # each agent has exactly 1 tool
        name=f"agent_{year.lower()}",       # e.g., "agent_fy2022"
        prompt=(
            f"You are a financial specialist for Bharti Airtel {year}. "
            f"Your ONLY job is to retrieve and report {year} financial data. "
            f"Always use the retrieve_{year}_data tool. "
            f"Be precise, cite numbers, and clearly label all figures as {year} data."
        )
    )

# Create one agent per year
agent_fy22 = make_year_agent("FY2022", tool_fy22)
agent_fy23 = make_year_agent("FY2023", tool_fy23)
agent_fy24 = make_year_agent("FY2024", tool_fy24)
agent_fy25 = make_year_agent("FY2025", tool_fy25)
```

---

### 5.3 — Create Parent Supervisor

```python
parent_supervisor_prompt = """
You are a senior financial analyst supervising a team of year-specific Airtel analysts.
Your team covers: agent_fy2022, agent_fy2023, agent_fy2024, agent_fy2025.

For multi-year queries:
1. Delegate to each relevant year's agent to gather their specific data
2. Wait for all agents to respond
3. Synthesize all responses into a coherent consolidated overview
4. Highlight year-over-year trends and the overall trajectory

Always produce a final consolidated answer after gathering all year data.
"""

hierarchical_workflow = create_supervisor(
    agents=[agent_fy22, agent_fy23, agent_fy24, agent_fy25],
    model=llm,
    prompt=parent_supervisor_prompt,
)

hierarchical_app = hierarchical_workflow.compile()
```

---

### 5.4 — Run Multi-Year Query

```python
result = hierarchical_app.invoke(
    {"messages": [HumanMessage(content=(
        "Summarize Airtel's revenue and EBITDA across FY2022–FY2025. "
        "Show year-over-year growth and the most significant trends."
    ))]},
    config={"recursion_limit": 40}   # higher limit — 4 child agents × ~5 steps each
)

final = [m for m in result["messages"] if hasattr(m, 'content')]
print(final[-1].content)
```

> **Why `recursion_limit=40`?** With 4 child agents, each doing ~5 steps (think → tool call → observe → think → answer), you need ~20–30 steps minimum. `40` gives comfortable headroom.

---

## 9. Section 6 — Hybrid Live + PDF RAG

### 6.1 — Scrape Screener.in

```python
import requests
from bs4 import BeautifulSoup

SCREENER_URL = "https://www.screener.in/company/BHARTIARTL/consolidated/"

def scrape_screener_data(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...",
        # ↑ Mimics a real browser so the server doesn't block the request
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()                  # raises error for 4xx/5xx
        soup = BeautifulSoup(response.text, "html.parser")

        metrics = {}

        # Extract company name
        name_tag = soup.find("h1", {"class": "h2"}) or soup.find("h1")
        metrics["company_name"] = name_tag.text.strip() if name_tag else "Bharti Airtel"

        # Extract key ratios from the summary section at top of page
        # Screener.in HTML: <ul id="top-ratios"><li><span class="name">P/E</span>
        #                                              <span class="number">72.5</span></li>
        ratio_section = soup.find("ul", {"id": "top-ratios"})
        if ratio_section:
            for li in ratio_section.find_all("li"):
                name_span  = li.find("span", {"class": "name"})
                value_span = li.find("span", {"class": "number"})
                if name_span and value_span:
                    key   = name_span.text.strip().replace(" ", "_").lower()
                    value = value_span.text.strip()
                    metrics[key] = value

        metrics["scrape_status"] = "success"

    except requests.exceptions.RequestException as e:
        # If scraping fails (bot detection, network error), use simulated data
        # This is a GRACEFUL FALLBACK — classroom still works without internet
        metrics = {
            "company_name":   "Bharti Airtel Ltd",
            "current_price":  "₹1,850 (simulated)",
            "market_cap":     "₹11,05,000 Cr (simulated)",
            "stock_p/e":      "72.5 (simulated)",
            "book_value":     "₹148 (simulated)",
            "dividend_yield": "0.54% (simulated)",
            "roce":           "13.2% (simulated)",
            "roe":            "22.1% (simulated)",
            "scrape_status":  "simulated_fallback",
        }

    return metrics

live_data = scrape_screener_data(SCREENER_URL)
```

**BeautifulSoup parsing explained:**
```
Screener.in HTML structure:
<ul id="top-ratios">
  <li>
    <span class="name">Market Cap</span>
    <span class="number">11,05,000</span>
    <span class="sub">Cr.</span>
  </li>
  <li>
    <span class="name">Stock P/E</span>
    <span class="number">72.54</span>
  </li>
  ...
</ul>

soup.find("ul", {"id": "top-ratios"})  → finds that <ul> element
.find_all("li")                         → gets each metric row
li.find("span", {"class": "name"})      → gets "Market Cap"
li.find("span", {"class": "number"})    → gets "11,05,000"
```

---

### 6.2 — Hybrid Query Function

```python
def hybrid_query(question: str, live_metrics: dict, retriever, llm) -> str:

    # Step 1: Retrieve relevant chunks from PDF (historical)
    pdf_docs    = retriever.invoke(question)
    pdf_context = format_docs(pdf_docs)

    # Step 2: Format live data into a readable block
    live_context = "\n".join(
        f"  {k}: {v}" for k, v in live_metrics.items()
        if k not in ["source_url", "scrape_status", "meta_description"]
    )

    # Step 3: Combine both sources in one LLM prompt
    hybrid_prompt_template = ChatPromptTemplate.from_template("""
You are a financial analyst combining historical annual report data with live market data.

═══ LIVE MARKET DATA (from Screener.in) ═══
{live_context}

═══ HISTORICAL DATA (from FY2025 Annual Report PDF) ═══
{pdf_context}

═══ QUESTION ═══
{question}

Provide a comprehensive answer that:
1. Quotes the current live metric(s)
2. Links them to relevant historical performance from the annual report
3. Gives an integrated investor perspective

Hybrid Answer:
""")

    chain = hybrid_prompt_template | llm | StrOutputParser()
    return chain.invoke({
        "live_context": live_context,
        "pdf_context":  pdf_context,
        "question":     question
    })
```

**Sample hybrid answer structure:**
```
Q: "What is Airtel's current stock price and how does it compare to FY2025 earnings?"

A: Airtel's current stock price is ₹1,850 (per Screener.in live data),
   giving the company a market cap of ₹11,05,000 Cr.

   Based on the FY2025 Annual Report, Airtel reported a net profit of
   ₹X,XXX Cr for the full year, implying a Price-to-Earnings ratio of
   approximately 72.5x — consistent with the live P/E shown on Screener.

   This premium valuation reflects investor confidence in Airtel's
   continued subscriber growth and ARPU improvement trajectory...
```

---

## 10. Section 7 — Student Exercises

### Exercise 1 — AGM Notice Summary
**Best pipeline: Base RAG or Supervisor (writer_specialist)**

```python
agm_query = "Summarize the FY2025 Annual General Meeting (AGM) notice including key resolutions and dates."
answer    = rag_chain.invoke(agm_query)
```

**Reflection questions:**
- What were the key resolutions voted on at the AGM?
- What does an AGM notice tell us about corporate governance?
- Did the Supervisor's writer_specialist give a better answer? Why?

---

### Exercise 2 — Revenue Growth FY2022–FY2025
**Best pipeline: Hierarchical Multi-Agent**

```python
revenue_query = (
    "Compare revenue growth across FY2022, FY2023, FY2024, and FY2025. "
    "Calculate CAGR if possible and identify the key drivers of growth in each year."
)
result = hierarchical_app.invoke(
    {"messages": [HumanMessage(content=revenue_query)]},
    config={"recursion_limit": 40}
)
final = [m for m in result["messages"] if hasattr(m, 'content')]
print(final[-1].content)
```

**Reflection questions:**
- What is Airtel's approximate revenue CAGR over 4 years?
- In which year was growth highest? What drove it?
- How does Africa vs India segment performance differ?

---

### Exercise 3 — Market Position vs FY2025
**Best pipeline: Hybrid (PDF + live data)**

```python
market_query = (
    "Explain Airtel's current market position based on live stock data. "
    "How does the current valuation (P/E, market cap) reflect FY2025 business performance? "
    "Is the market pricing in future growth expectations?"
)
answer = hybrid_query(market_query, live_data, retriever, llm)
print(answer)
```

**Reflection questions:**
- What does a high P/E ratio (70-80x) suggest about investor expectations?
- Does the current price seem justified given FY2025 earnings?
- What risks could affect this valuation?

---

### Exercise 4 — Open-Ended (Your Design)

Choose any query and any pipeline. Suggested topics:
- Airtel's debt reduction strategy (FY2022–FY2025)
- 5G capex and subscriber growth
- Africa segment profitability vs India
- ARPU (Average Revenue Per User) trends
- Dividend history and payout policy

```python
your_query = "What is Airtel's ARPU trend from FY2022 to FY2025 and what drives it?"

# Available pipelines:
# rag_chain.invoke(your_query)
# network_graph.invoke({"query": your_query})
# supervisor_app.invoke({"messages": [HumanMessage(content=your_query)]})
# hierarchical_app.invoke({"messages": [HumanMessage(content=your_query)]})
# hybrid_query(your_query, live_data, retriever, llm)
```

---

## 11. Pattern Comparison Cheat Sheet

```
┌──────────────────┬────────────────────┬──────────────────┬──────────────────┐
│ Pattern          │ When to Use        │ Pros             │ Cons             │
├──────────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Base RAG         │ Single factual Q&A │ Fast, cheap,     │ No reasoning     │
│                  │ "What was X?"      │ easy to debug    │ layer, no agents │
├──────────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Network          │ Always Research    │ Structured,      │ Rigid — can't    │
│ Multi-Agent      │ → Analyze → Write  │ auditable,       │ skip steps or    │
│                  │ pipeline needed    │ predictable      │ reorder          │
├──────────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Supervisor       │ Mixed query types  │ Flexible routing,│ Supervisor LLM   │
│ Multi-Agent      │ unknown ahead of   │ can combine      │ adds latency +   │
│                  │ time               │ specialists      │ cost             │
├──────────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Hierarchical     │ Multi-year,        │ Scalable, clean  │ Complex setup,   │
│ Multi-Agent      │ multi-source, or   │ separation of    │ high recursion   │
│                  │ domain-separated   │ concerns         │ limit needed     │
├──────────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Hybrid           │ Live + historical  │ Rich real-world  │ Scraping can     │
│ (PDF + Web)      │ data combined      │ answers for      │ break, not       │
│                  │                    │ investors        │ reproducible     │
└──────────────────┴────────────────────┴──────────────────┴──────────────────┘
```

**When to pick which pattern:**

```
Query type                              → Best pipeline
─────────────────────────────────────────────────────────
"What was revenue in FY2025?"           → Base RAG
"Summarize the AGM notice"              → Base RAG or Supervisor
"Compare FY25 revenue to FY24"          → Supervisor (analysis_specialist)
"Explain EBITDA trends"                 → Network (all 3 agents run)
"Summarize all 4 years"                 → Hierarchical
"Current P/E vs FY25 earnings"          → Hybrid
"Write student notes on Airtel FY25"    → Network (full pipeline)
"How many subscribers in each year?"    → Hierarchical
```

---

## 12. Common Errors & Fixes

### Error 1: `GROQ_API_KEY` not found
```
EnvironmentError: ❌ GROQ_API_KEY not found.
```
**Fix:** Create a `.env` file in the same directory as the notebook:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### Error 2: PDF not found
```
⚠️ PDF not found at: /Users/pradipwasre/...
```
**Fix:** Update `PDF_PATH` to your actual file path:
```python
PDF_PATH = "/your/actual/path/Bharti Airtel.pdf"
```

---

### Error 3: Recursion limit exceeded
```
GraphRecursionError: Recursion limit of 20 reached
```
**Fix:** Increase the limit in `.invoke()`:
```python
config={"recursion_limit": 40}   # or 50 for hierarchical
```

---

### Error 4: HuggingFace model download slow
First run downloads ~90MB for `all-MiniLM-L6-v2`. Subsequent runs use cache.  
**Fix:** Run `2.3` cell once, wait for download, then reuse the saved FAISS index:
```python
# To reload a saved index instead of rebuilding:
vectorstore = FAISS.load_local(
    "airtel_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

### Error 5: Screener.in scraping blocked
```
⚠️ Network error: 403 Forbidden
```
**Fix:** The code automatically falls back to simulated data. For real data:
- Open `https://www.screener.in/company/BHARTIARTL/consolidated/` in a browser
- Manually copy the figures into the `metrics` dict

---

### Error 6: Tool name collision in Hierarchical pattern
```
ValueError: Tool name 'retrieve_data' already exists
```
**Fix:** The `make_year_tool` factory uses unique names:  
`f"retrieve_{year}_data"` → `retrieve_FY2022_data`, `retrieve_FY2023_data`, etc.  
Never reuse tool names across agents in the same graph.

---

## 13. Key Concepts Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation — LLM answers from a given context, not just training data |
| **Chunk** | A small piece of text (≈1000 chars) split from a larger document |
| **Chunk overlap** | Shared text between adjacent chunks to preserve context at boundaries |
| **Embedding** | A numerical vector (list of floats) representing the semantic meaning of text |
| **FAISS** | Facebook AI Similarity Search — efficient nearest-neighbor search over vectors |
| **Retriever** | Component that takes a query and returns the most relevant document chunks |
| **LLM** | Large Language Model — the AI (here: Llama 3.1 via Groq) that generates answers |
| **Groq** | Cloud provider offering very fast LLM inference on custom chips |
| **LCEL** | LangChain Expression Language — the `\|` pipe syntax for chaining components |
| **StateGraph** | LangGraph's directed graph where nodes share a typed state dictionary |
| **Node** | A function in a StateGraph that reads and writes to the shared state |
| **Edge** | A connection between nodes defining execution order |
| **ReAct Agent** | An agent that alternates: Think → Tool call → Observe → Think (loop) |
| **Tool** | A Python function with `@tool` decorator that an LLM agent can call |
| **Supervisor** | An LLM that reads queries and routes them to the correct specialist agent |
| **Handoff** | When a supervisor passes control to a specialist agent |
| **Hierarchical** | Multi-level agent structure: parent supervises child supervisors or agents |
| **TypedDict** | Python type annotation for dicts with known key names and value types |
| **recursion_limit** | Safety cap on how many steps a LangGraph agent loop can take |
| **HuggingFaceEmbeddings** | Local embedding model — free, no API key, runs on CPU |
| **ChatPromptTemplate** | A reusable prompt with `{placeholder}` slots filled at runtime |
| **StrOutputParser** | Extracts just the text string from an LLM's response object |
| **RunnablePassthrough** | LCEL component that passes its input unchanged to the next stage |
| **BeautifulSoup** | Python library for parsing HTML/XML — used here to scrape Screener.in |
| **ARPU** | Average Revenue Per User — key telecom metric |
| **EBITDA** | Earnings Before Interest, Tax, Depreciation & Amortisation |
| **AGM** | Annual General Meeting — mandatory yearly shareholder meeting |
| **CAGR** | Compound Annual Growth Rate — the smoothed annual growth over multiple years |
| **P/E Ratio** | Price-to-Earnings — stock price divided by earnings per share |

---

## Complete Data Flow Diagram

```
                         ╔══════════════════════════════════════╗
                         ║  KNOWLEDGE SOURCES                   ║
                         ║                                      ║
                         ║  📄 Bharti Airtel.pdf                ║
                         ║  (FY2025 Annual Report)              ║
                         ║                                      ║
                         ║  🌐 Screener.in                      ║
                         ║  (Live stock data)                   ║
                         ╚══════════════╤═══════════════════════╝
                                        │
                          ┌─────────────▼──────────────┐
                          │      INGESTION PIPELINE      │
                          │                              │
                          │  PyPDFLoader                 │
                          │      ↓                       │
                          │  RecursiveCharacterSplitter  │
                          │      ↓                       │
                          │  HuggingFaceEmbeddings        │
                          │      ↓                       │
                          │  FAISS VectorStore ──save──► │
                          │  airtel_faiss_index/         │
                          └─────────────┬────────────────┘
                                        │ retriever
                          ╔═════════════▼════════════════════╗
                          ║          5 RAG PATTERNS          ║
                          ║                                  ║
                          ║  A: rag_chain (Base RAG)         ║
                          ║  B: network_graph (Network)      ║
                          ║  C: supervisor_app (Supervisor)  ║
                          ║  D: hierarchical_app (Hierarch.) ║
                          ║  E: hybrid_query (PDF + Web)     ║
                          ╚═════════════╤════════════════════╝
                                        │
                                        ▼
                              ChatGroq (Groq API)
                              llama-3.1-8b-instant
                                        │
                                        ▼
                                   📝 Answer
```

---

*Study Guide for: `MultiAgent_RAG_Financials_BhartiAirtel.ipynb`*  
*Built with: LangGraph 0.3+ | LangChain 0.3+ | langgraph-supervisor 0.0.9+ | 2026 APIs*