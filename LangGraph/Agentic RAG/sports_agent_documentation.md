#  Sports Research Agent — Deep Technical Documentation

> **Project:** Sports Research Agent with Adaptive Query Rephrasing  
> **Architecture:** Agentic RAG (Retrieval-Augmented Generation)  
> **Stack:** LangGraph · LangChain · Groq LLaMA 3.3 · BeautifulSoup · Python 3.11  
> **Year:** 2026

---

## Table of Contents

1. [What is This Agent Called?](#1-what-is-this-agent-called)
2. [What Problem Does It Solve?](#2-what-problem-does-it-solve)
3. [High-Level Architecture](#3-high-level-architecture)
4. [The RAG Pattern : Explained Simply](#4-the-rag-pattern--explained-simply)
5. [Graph: Nodes and Edges](#5-graph-nodes-and-edges)
6. [Code Flow : Step by Step](#6-code-flow--step-by-step)
7. [Each Node in Detail](#7-each-node-in-detail)
8. [The Web Scraper : How It Handles Any Website](#8-the-web-scraper--how-it-handles-any-website)
9. [Agent State : The Shared Memory](#9-agent-state--the-shared-memory)
10. [The Rephrasing Loop : Self-Correction](#10-the-rephrasing-loop--self-correction)
11. [Why No Tool-Binding? The BadRequestError Fix](#11-why-no-tool-binding-the-badrequesterror-fix)
12. [Libraries and Their Roles](#12-libraries-and-their-roles)
13. [Data Flow Diagram](#13-data-flow-diagram)
14. [Example Execution Trace](#14-example-execution-trace)
15. [Limitations and Future Improvements](#15-limitations-and-future-improvements)

---

## 1. What is This Agent Called?

**Name:** `Sports Research Agent` (internally referred to as `SRA`)

**Full Classification:**  
`Agentic RAG` — specifically a **Tool-Routing Retrieval Agent with Adaptive Query Rephrasing**

This is not a standard chatbot. It is an **autonomous agent** — meaning it:
- Makes its own decisions about which tool to use
- Evaluates whether it got good data
- Corrects itself by rephrasing the query if needed
- Loops until it has a satisfactory answer
- Only then generates the final response

It is called "Agentic RAG" because it combines:
- **RAG** (Retrieval-Augmented Generation) — fetching real-time web data before answering
- **Agentic behaviour** — the ability to loop, self-evaluate, and rephrase using LangGraph

---

## 2. What Problem Does It Solve?

### Original Problem
The earlier version tried to use **LangChain's native tool-binding** with Groq's API. This caused a `BadRequestError: tool_use_failed` because Groq's LLaMA model generated malformed function call syntax:

```
<function=FootballAnalytics>{"Premier League"}</function>
```

Groq's API expected structured JSON but received this broken format instead — crashing the entire pipeline.

### What This Agent Solves
Beyond just fixing the crash, this agent solves a real-world challenge:

> **"How do I build an AI that fetches live sports data from the web, handles bad/incomplete results gracefully, and always returns a useful answer?"**

It does this by:
- Using plain-text LLM routing instead of fragile tool-binding
- Scraping multiple website sources per query with 3 fallback strategies
- Self-evaluating data quality and rephrasing queries automatically
- Keeping the architecture simple enough to understand and extend

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT LAYER                         │
│          run_agent("Compare EPL vs IPL balance")            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ENGINE                         │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌─────────────┐          │
│   │  Agent   │───▶│  Tool    │───▶│  Evaluator  │          │
│   │  Node    │    │  Node    │    │    Node      │          │
│   └──────────┘    └──────────┘    └──────┬──────┘          │
│        ▲                                 │                  │
│        │                    ┌────────────┴────────────┐     │
│        │                    │                         │     │
│   ┌────┴──────┐        ┌────▼──────┐           ┌─────▼──┐  │
│   │ Rephraser │        │ Rephraser │           │ Answer │  │
│   │  Node     │        │  (loop)   │           │  Node  │  │
│   └───────────┘        └───────────┘           └────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                          │
│                                                             │
│   ┌─────────────────────┐   ┌─────────────────────────┐    │
│   │  FootballAnalytics  │   │   CricketAnalytics       │    │
│   │  Tool               │   │   Tool                   │    │
│   │                     │   │                          │    │
│   │  Sources:           │   │  Sources:                │    │
│   │  • BBC Sport EPL    │   │  • Wikipedia IPL 2025    │    │
│   │  • Sky Sports       │   │  • BBC Sport Cricket     │    │
│   │  • Wikipedia EPL    │   │  • Wikipedia IPL History │    │
│   └──────────┬──────────┘   └────────────┬─────────────┘   │
│              └──────────┬────────────────┘                  │
│                         ▼                                   │
│              scrape_webpage() [3 strategies]                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM LAYER (Groq)                         │
│                  llama-3.3-70b-versatile                    │
│                                                             │
│    Used in:  Agent · Rephraser · Answer Generator           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. The RAG Pattern — Explained Simply

**RAG = Retrieval-Augmented Generation**

Normal LLMs have a knowledge cutoff — they don't know what happened last week in sports. RAG fixes this by adding a retrieval step:

```
WITHOUT RAG:
  Question ──▶ LLM (trained data only) ──▶ Answer
                May be outdated

WITH RAG:
  Question ──▶ Retrieve fresh data ──▶ LLM + fresh data ──▶ Answer
                                         Always current
```

**This agent is "Agentic RAG"** because retrieval is not a passive one-shot fetch. The agent actively:

| Step | What Happens |
|------|-------------|
| **Route** | Decides which source/tool is relevant |
| **Retrieve** | Fetches data from real websites |
| **Evaluate** | Judges if the retrieved data is useful |
| **Retry** | If not useful, rephrases query and retrieves again |
| **Generate** | Synthesizes final answer from collected data |

This cycle is what makes it "agentic" — it has **autonomy, memory (state), tools, and self-correction**.

---

## 5. Graph: Nodes and Edges

### Visual Graph (ASCII)

```
                    ┌──────────────────────────────────────┐
                    │              START                    │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │      AGENT NODE        │
                          │                        │
                          │  • Reads the query     │
                          │  • Asks LLM: which     │
                          │    tool should I use?  │
                          │  • Returns: Football / │
                          │    Cricket / BOTH      │
                          └────────────┬───────────┘
                                       │
                              (always goes to tool)
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │       TOOL NODE        │
                          │                        │
                          │  • Calls Football tool │
                          │    and/or Cricket tool │
                          │  • Each tool scrapes   │
                          │    up to 3 websites    │
                          │  • Returns raw text    │
                          └────────────┬───────────┘
                                       │
                              (always goes to evaluator)
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │    EVALUATOR NODE      │
                          │                        │
                          │  • Checks if data has  │
                          │    200+ chars of text  │
                          │  • Checks for error    │
                          │    indicators          │
                          │  • Sets needs_rephrase │
                          └────────────┬───────────┘
                                       │
                       ┌───────────────┴────────────────┐
                       │  CONDITIONAL EDGE              │
                       │  should_rephrase()             │
                       └───────┬───────────────┬────────┘
                               │               │
                    needs_rephrase=True   needs_rephrase=False
                    (AND attempts < 2)    OR attempts >= 2
                               │               │
                               ▼               ▼
                  ┌────────────────────┐  ┌──────────────────┐
                  │   REPHRASER NODE   │  │   ANSWER NODE    │
                  │                   │  │                  │
                  │ • Asks LLM to     │  │ • Sends query +  │
                  │   rewrite query   │  │   scraped data   │
                  │ • Increments      │  │   to LLM         │
                  │   attempt count   │  │ • Gets final     │
                  │ • Updates query   │  │   synthesized    │
                  │   in state        │  │   answer         │
                  └────────┬──────────┘  └────────┬─────────┘
                           │                      │
                    (loops back to agent)          │
                           │                      ▼
                           └──────────────▶     END
```

### Formal Edge Table

| From Node | To Node | Edge Type | Condition |
|-----------|---------|-----------|-----------|
| `START` | `agent` | Fixed | Always |
| `agent` | `tool` | Fixed | Always |
| `tool` | `evaluator` | Fixed | Always |
| `evaluator` | `rephraser` | Conditional | `needs_rephrase == True AND attempts < 2` |
| `evaluator` | `answer` | Conditional | `needs_rephrase == False OR attempts >= 2` |
| `rephraser` | `agent` | Fixed | Always (creates the loop) |
| `answer` | `END` | Fixed | Always |

### Node Summary Table

| Node | Function Name | LLM Used? | External Call? | Role |
|------|---------------|-----------|----------------|------|
| `agent` | `agent_node()` |   Yes |   No | Routing brain |
| `tool` | `tool_node()` |   No | ✅ Yes (web) | Data fetcher |
| `evaluator` | `evaluator_node()` |   No |   No | Quality checker |
| `rephraser` | `rephraser_node()` |   Yes |   No | Query rewriter |
| `answer` | `answer_node()` |   Yes |   No | Response generator |

---

## 6. Code Flow — Step by Step

Here is the complete execution flow for the query:
> *"Compare the top teams in Premier League football with the leading cricket teams in IPL. Which domain has more competitive balance?"*

```
STEP 1 ─── run_agent(query) is called
           │
           └─ Creates initial AgentState:
              {
                query:               "Compare top teams in EPL...",
                original_query:      "Compare top teams in EPL...",   ← never changes
                tool_choice:         "",
                tool_output:         "",
                final_answer:        "",
                rephrasing_attempts: 0,
                needs_rephrase:      False
              }
              └─ graph.invoke(initial_state) starts the graph

STEP 2 ─── agent_node() runs
           │
           ├─ Sends query to ChatGroq with system prompt:
           │   "Decide which tool to use: FootballAnalytics,
           │    CricketAnalytics, or BOTH"
           │
           └─ LLM replies: "BOTH"
              └─ State updated: tool_choice = "BOTH"

STEP 3 ─── tool_node() runs
           │
           ├─ Sees tool_choice = "BOTH"
           │
           ├─ Calls football_analytics_tool(query)
           │   ├─ Tries BBC Sport EPL table URL
           │   ├─ scrape_webpage() → extracts 2000 chars
           │   └─ Returns EPL standings text
           │
           ├─ Calls cricket_analytics_tool(query)
           │   ├─ Tries Wikipedia IPL 2025 URL
           │   ├─ scrape_webpage() → extracts 2000 chars
           │   └─ Returns IPL data text
           │
           └─ State updated: tool_output = "=== FOOTBALL DATA ===\n...\n=== CRICKET DATA ===\n..."

STEP 4 ─── evaluator_node() runs
           │
           ├─ Checks: rephrasing_attempts (0) < 2 ✓
           ├─ Checks: no error indicators in tool_output ✓
           ├─ Checks: len(tool_output) > 200 ✓
           │
           └─ State updated: needs_rephrase = False

STEP 5 ─── should_rephrase() conditional edge fires
           │
           └─ needs_rephrase = False → routes to "answer"

STEP 6 ─── answer_node() runs
           │
           ├─ Sends to ChatGroq:
           │   • System: "You are a sports analyst. Answer clearly."
           │   • User:   original_query + "\n\n" + tool_output
           │
           └─ LLM generates a 3-5 paragraph comparison answer
              └─ State updated: final_answer = "..."

STEP 7 ─── Graph reaches END
           └─ graph.invoke() returns final state
              └─ run_agent() prints and returns final_answer
```

---

## 7. Each Node in Detail

### Node 1: `agent_node` — The Brain

**Purpose:** Reads the user's query and decides which tool(s) to call.

**How it works:**
```python
# System prompt forces a single-word answer — no room for confusion
system_prompt = """You are a sports research assistant.
Given a user query, decide which tool to use:
- FootballAnalytics: for Premier League, soccer, football questions
- CricketAnalytics: for IPL, cricket questions
- BOTH: if the question compares football AND cricket

Reply with ONLY one of these three words: FootballAnalytics, CricketAnalytics, or BOTH"""
```

**Safety net:** If LLM returns something unexpected (not one of the 3 valid words), it defaults to `BOTH` — so the agent never crashes here.

```python
if tool_choice not in ["FootballAnalytics", "CricketAnalytics", "BOTH"]:
    tool_choice = "BOTH"  # safe default
```

**Input state fields used:** `query`  
**Output state fields updated:** `tool_choice`

---

### Node 2: `tool_node` — The Fetcher

**Purpose:** Calls the appropriate tool function(s) and collects raw web data.

**How it works:**
```python
# Reads tool_choice and calls the matching Python function directly
if choice in ["FootballAnalytics", "BOTH"]:
    results.append(TOOLS["FootballAnalytics"](query))

if choice in ["CricketAnalytics", "BOTH"]:
    results.append(TOOLS["CricketAnalytics"](query))
```

**The TOOLS dictionary** is just a plain Python dict mapping strings to functions:
```python
TOOLS = {
    "FootballAnalytics": football_analytics_tool,
    "CricketAnalytics":  cricket_analytics_tool,
}
```

This is the key architectural decision — **no LangChain tool binding, no JSON function schemas** — just a Python dictionary. Clean, simple, and works with every LLM.

**Input state fields used:** `tool_choice`, `query`  
**Output state fields updated:** `tool_output`

---

### Node 3: `evaluator_node` — The Judge

**Purpose:** Determines whether the retrieved data is actually useful.

**Two checks it runs:**

```python
# Check 1: Is the data full of error messages?
error_indicators = ["Could not retrieve", "Error scraping", "Timeout", "HTTP error"]
is_error = any(ind in tool_output for ind in error_indicators)

# Check 2: Is there enough real content?
is_too_short = len(tool_output.strip()) < 200
```

**The safety valve:**
```python
# Never rephrase more than 2 times — prevents infinite loops
if attempts >= 2:
    return {**state, "needs_rephrase": False}  # force proceed to answer
```

**Input state fields used:** `tool_output`, `rephrasing_attempts`  
**Output state fields updated:** `needs_rephrase`

---

### Node 4: `rephraser_node` — The Self-Corrector

**Purpose:** Rewrites the current query to try getting better search results on the next loop iteration.

**How it works:**
```python
system_prompt = """You are a search query optimizer.
The original query returned poor results. Rephrase it to be more specific and
likely to return useful sports data. Keep the same intent.
Reply with ONLY the rephrased query — no explanation."""
```

**Example transformation:**
```
Before: "Compare top teams in EPL and IPL"
After:  "Premier League 2024-25 standings top 5 teams points table"
        (or) "IPL 2025 team rankings leading clubs competitive"
```

**Input state fields used:** `query`, `rephrasing_attempts`  
**Output state fields updated:** `query` (overwritten), `rephrasing_attempts` (incremented), `needs_rephrase = False`

---

### Node 5: `answer_node` — The Synthesizer

**Purpose:** Uses the LLM to turn raw scraped text into a clean, readable answer.

**Critical design choice:** It always uses `original_query`, not the possibly-rephrased `query`. This ensures the final answer addresses what the user actually asked, even if the rephrased version was more search-optimized.

```python
user_content = f"""Question: {state['original_query']}   ← always original

Sports Data Collected:
{state['tool_output']}

Please answer the question based on this data."""
```

**Input state fields used:** `original_query`, `tool_output`  
**Output state fields updated:** `final_answer`

---

## 8. The Web Scraper — How It Handles Any Website

The `scrape_webpage()` function is one of the most important pieces of this agent. Modern websites are built with JavaScript frameworks, dynamic loading, anti-bot protection, and wildly different HTML structures. Here is exactly how it handles them.

### Why Normal Scraping Fails

```
Problem 1: JavaScript-rendered content
  → requests.get() gets the HTML shell, not the rendered page
  → Fix: we parse all <article>, <main>, <section> tags which 
         usually contain server-side rendered content

Problem 2: Anti-bot blocking
  → Many sites block Python's default User-Agent
  → Fix: we send a real Chrome browser User-Agent header

Problem 3: Different HTML templates
  → BBC uses <article>, Sky Sports uses <div class="content">,
    Wikipedia uses <div class="mw-parser-output">
  → Fix: 3 fallback strategies handle all templates
```

### The 3-Strategy Fallback System

```python
# Strategy 1: Semantic HTML tags (modern websites)
content_tags = soup.find_all(["article", "main", "section"], limit=3)
# Works for: BBC Sport, most news sites, Wikipedia

# Strategy 2: Content div class names (older/custom sites)
for cls in ["content", "article", "post", "story", "body", "text"]:
    div = soup.find("div", class_=re.compile(cls, re.I))
# Works for: Sky Sports, forums, blog-style sports sites

# Strategy 3: Pure paragraph extraction (last resort)
paras = soup.find_all("p")
text = " ".join(p.get_text(strip=True) for p in paras)
# Works for: almost any webpage that has text content
```

### Clutter Removal

Before any strategy runs, all non-content HTML is stripped:

```python
for tag in soup(["script", "style", "nav", "footer",
                 "header", "aside", "form", "noscript",
                 "iframe", "svg", "button"]):
    tag.decompose()
```

This removes ads, navigation menus, cookie banners, social share buttons, and JavaScript — leaving only the meaningful text.

### Multi-Source Fallback Per Tool

Each tool also tries multiple URLs in sequence:

```python
sources = [
    "https://www.bbc.com/sport/football/premier-league/table",  # try first
    "https://www.skysports.com/premier-league-table",           # fallback 1
    "https://en.wikipedia.org/wiki/2024–25_Premier_League",     # fallback 2
]

for url in sources:
    text = scrape_webpage(url)
    if not text.startswith("["):   # "[" means it returned an error string
        collected.append(text)
        if len(collected) >= 2:    # 2 good sources is enough
            break
```

This means even if BBC is down or blocking requests, the agent gets its data from Wikipedia.

---

## 9. Agent State — The Shared Memory

The `AgentState` is a Python `TypedDict` — a structured dictionary that gets passed between every node. Think of it as the agent's working memory.

```python
class AgentState(TypedDict):
    query:               str    # Current query (may be rephrased in loop)
    original_query:      str    # Original user question — NEVER modified
    tool_choice:         str    # "FootballAnalytics" / "CricketAnalytics" / "BOTH"
    tool_output:         str    # Raw text scraped from web
    final_answer:        str    # LLM-generated final response
    rephrasing_attempts: int    # Counter: 0, 1, or 2 max
    needs_rephrase:      bool   # Signal passed from evaluator to router
```

### State Lifecycle

```
INITIAL STATE                  AFTER AGENT          AFTER TOOL
─────────────────────          ────────────────      ──────────────────────────
query:         "Compare..."    (unchanged)           (unchanged)
original_query:"Compare..."    (unchanged)           (unchanged)
tool_choice:   ""          ──▶ "BOTH"          ──▶  "BOTH"
tool_output:   ""              ""              ──▶  "=== FOOTBALL ==="
final_answer:  ""              ""                   ""
attempts:      0               0                    0
needs_rephrase:False           False                False

AFTER EVALUATOR (good)         AFTER ANSWER
────────────────────           ─────────────────────
(unchanged)                    (unchanged)
(unchanged)                    (unchanged)
"BOTH"                         "BOTH"
"=== FOOTBALL ==="             "=== FOOTBALL ==="
""                        ──▶  "Based on the data, EPL..."
0                              0
False ◀ set here               False
```

---

## 10. The Rephrasing Loop — Self-Correction

This is the feature that makes this agent "agentic" rather than a simple pipeline.

### When Does Rephrasing Trigger?

```
evaluator_node checks:

  IF (tool_output contains error strings
      OR tool_output is shorter than 200 characters)
  AND rephrasing_attempts < 2:
      → needs_rephrase = True
      → route to rephraser_node
```

### The Loop Path

```
agent ──▶ tool ──▶ evaluator ──▶ rephraser ──▶ agent ──▶ tool ──▶ evaluator ──▶ answer
  [attempt 0]              [attempt 1]                    [attempt 1]          [done]
```

### Maximum Loop Guard

Without a guard, a bad query could loop forever. The `rephrasing_attempts` counter prevents this:

```python
if attempts >= 2:
    # Force move to answer even with bad data
    # Better to give a partial answer than infinite loops
    return {**state, "needs_rephrase": False}
```

### Rephrasing Example

```
Loop 0:
  Query:    "Compare EPL vs IPL competitive balance"
  Scrape:   [HTTP error 403 for www.bbc.com]      ← blocked
  Evaluate: Error detected → rephrase

Loop 1:
  Query:    "Premier League 2024-25 top teams standings Wikipedia"
  Scrape:   "Arsenal: 89 points, Liverpool: 87 points..."
  Evaluate: 2000+ chars, no errors → proceed to answer

Answer:    "Based on the 2024-25 Premier League data..."
```

---

## 11. Why No Tool-Binding? The BadRequestError Fix

This is important to understand because it explains a fundamental design decision.

### The Old Broken Way (LangChain Native Tool Binding)

```python
# What the old code tried to do
tools = [football_tool, cricket_tool]
llm_with_tools = llm.bind_tools(tools)   # ← This is the problem

# LangChain sends a JSON schema to the LLM describing each tool
# It then expects the LLM to respond with structured JSON like:
# {"tool": "FootballAnalytics", "args": {"query": "..."}}

# But Groq's LLaMA 3.3 sometimes generates this instead:
# <function=FootballAnalytics>{"Premier League"}</function>
# ↑ This is malformed XML, not JSON → BadRequestError 400
```

### The Error Message

```
BadRequestError: Error code: 400 - {
  'error': {
    'message': "Failed to call a function. Please adjust your prompt.",
    'code': 'tool_use_failed',
    'failed_generation': '<function=FootballAnalytics>{"Premier League"}</function>'
  }
}
```

The model tried to call the tool but used XML-like syntax instead of JSON. Groq's API couldn't parse it.

### The New Working Way (Plain Text Routing)

```python
# What this agent does instead
system_prompt = "Reply with ONLY: FootballAnalytics, CricketAnalytics, or BOTH"
response = llm.invoke(messages)
tool_choice = response.content.strip()  # e.g., "BOTH"

# Python then calls the function directly — no JSON parsing needed
TOOLS[tool_choice](query)
```

### Comparison

| Aspect | Old (Tool Binding) | New (Plain Text) |
|--------|-------------------|------------------|
| LLM Output | `{"tool": "X", "args": {...}}` | `"BOTH"` |
| Parse Step | JSON parsing (can fail) | `.strip()` (never fails) |
| Groq Compatible |   Often fails |   Always works |
| Code Complexity | High | Low |
| Extensible | Medium | Very easy |

---

## 12. Libraries and Their Roles

| Library | Version | Purpose in This Project |
|---------|---------|------------------------|
| `langgraph` | ≥0.2 | Builds the directed graph of nodes and edges. Manages state passing, conditional routing, and the rephrase loop |
| `langchain` | ≥0.3 | Base framework — provides `HumanMessage`, `SystemMessage`, `AIMessage` objects |
| `langchain-groq` | ≥0.2 | LangChain adapter for Groq's API — gives us `ChatGroq` class |
| `langchain-community` | ≥0.3 | Community tools and integrations (imported for compatibility) |
| `groq` | (auto) | Underlying Groq API client — used internally by `langchain-groq` |
| `requests` | ≥2.31 | Makes HTTP GET requests to sports websites |
| `beautifulsoup4` | ≥4.12 | Parses HTML and extracts readable text using the 3-strategy system |
| `lxml` | ≥4.9 | Fast HTML/XML parser used by BeautifulSoup (faster than Python's built-in `html.parser`) |
| `typing` | (stdlib) | Provides `TypedDict` for the `AgentState` type definition |
| `re` | (stdlib) | Regular expressions for whitespace cleanup and class name matching |
| `os` | (stdlib) | Sets the `GROQ_API_KEY` environment variable |

---

## 13. Data Flow Diagram

```
USER QUERY (string)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  AgentState (TypedDict)                                     │
│  ┌──────────────────┬──────────────────────────────────┐   │
│  │ query            │ "Compare EPL vs IPL..."          │   │
│  │ original_query   │ "Compare EPL vs IPL..."          │   │
│  │ tool_choice      │ ""  →  "BOTH"                    │   │
│  │ tool_output      │ ""  →  "=== FOOTBALL ==\n..."    │   │
│  │ final_answer     │ ""  →  "Based on data..."        │   │
│  │ attempts         │ 0                                │   │
│  │ needs_rephrase   │ False                            │   │
│  └──────────────────┴──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       │  Flows through nodes ──────────────────────────────┐
       │                                                     │
       ▼                                                     │
┌─────────────┐   LLM Call #1: "Which tool?"                 │
│ agent_node  │ ──────────────────────────▶ Groq API         │
│             │ ◀────────────────────────── "BOTH"           │
└──────┬──────┘                                              │
       │                                                     │
       ▼                                                     │
┌─────────────┐   HTTP GET x4 (2 football + 2 cricket)      │
│  tool_node  │ ──────────────────────────▶ Web Servers      │
│             │ ◀────────────────────────── HTML pages       │
│             │   BeautifulSoup extracts clean text          │
└──────┬──────┘                                              │
       │                                                     │
       ▼                                                     │
┌──────────────┐  (no LLM, no web — pure Python logic)      │
│ evaluator    │                                             │
│   _node      │  len(text) > 200? ✓  No errors? ✓          │
└──────┬───────┘  → needs_rephrase = False                   │
       │                                                     │
       ▼ (conditional edge: answer path)                     │
┌─────────────┐   LLM Call #2: "Synthesize answer"          │
│ answer_node │ ──────────────────────────▶ Groq API         │
│             │ ◀────────────────────────── Full analysis    │
└──────┬──────┘                                              │
       │                                                     │
       ▼                                                     │
    FINAL ANSWER (string)  ◀─────────────────────────────────┘
```

---

## 14. Example Execution Trace

Below is a realistic execution trace showing all print statements you would see in the notebook:

```
============================================================
  QUERY: Compare the top teams in Premier League football
          with the leading cricket teams in IPL. Which domain
          has more competitive balance?
============================================================

  [Agent Node] Analyzing query: 'Compare the top teams...'
  → Tool selected: BOTH

  [Tool Node] Executing: BOTH
    [Football Tool] Searching for: Compare the top teams...
  → Collected 4127 characters of data

  [Evaluator Node] Checking data quality...
  → Data quality looks good! Proceeding to answer.

  [Answer Node] Generating final answer...

============================================================
  FINAL ANSWER:
============================================================
Based on the data retrieved from BBC Sport and Wikipedia,
the 2024-25 Premier League featured intense competition at
the top. Arsenal finished with 89 points, followed closely
by Liverpool with 87, and Manchester City with 85...

In contrast, the 2025 IPL saw Mumbai Indians dominate...

In terms of competitive balance, the Premier League shows
a tighter points gap among the top 5 teams (only 8 points
from 1st to 5th), while the IPL...
============================================================
(Rephrasing attempts used: 0)
```

**Rephrase scenario (what it looks like when data fails):**

```
  [Evaluator Node] Checking data quality...
  → Data quality poor (attempt 1). Will rephrase.

   [Rephraser Node] Rephrasing query...
  → New query: 'Premier League 2024-25 table standings top teams'

  [Agent Node] Analyzing query: 'Premier League 2024-25 table...'
  → Tool selected: FootballAnalytics

  [Tool Node] Executing: FootballAnalytics
    [Football Tool] Searching for: Premier League 2024-25...
  → Collected 3891 characters of data

  [Evaluator Node] Checking data quality...
  → Data quality looks good! Proceeding to answer.
```

---

## 15. Limitations and Future Improvements

### Current Limitations

| Limitation | Description |
|-----------|-------------|
| **JavaScript-heavy sites** | `requests` cannot execute JavaScript. Sites like ESPN that render tables with React/Vue will return empty content |
| **2 domain scope** | Only supports Football (EPL) and Cricket (IPL). Does not handle NBA, F1, Tennis, etc. |
| **No memory** | Each `run_agent()` call is independent — no conversation history |
| **No caching** | Same URL scraped every time even if called twice in a row |
| **Rate limiting** | Calling too many URLs quickly may trigger anti-bot blocking |

### Possible Improvements

**Short term:**
- Add `Selenium` or `Playwright` for JavaScript-rendered pages
- Add URL response caching with `functools.lru_cache`
- Add more sports domains (NBA, F1, Tennis, NHL)

**Medium term:**
- Add a `memory_node` that stores results between queries
- Add source citation in the final answer ("According to BBC Sport...")
- Add streaming output so the answer appears token-by-token

**Long term:**
- Replace BeautifulSoup with a dedicated search API (SerpAPI, Tavily) for more reliable retrieval
- Add a vector database to store and reuse previously scraped data (true RAG with embeddings)
- Deploy as a FastAPI web service so any front-end can call it

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              SPORTS RESEARCH AGENT — QUICK REFERENCE        │
├────────────────┬────────────────────────────────────────────┤
│ Architecture   │ Agentic RAG (Tool-Routing + Self-Correction)│
│ LLM            │ llama-3.3-70b-versatile via Groq API        │
│ Graph Engine   │ LangGraph (StateGraph)                      │
│ Retrieval      │ BeautifulSoup web scraping (3 strategies)   │
│ Max Retries    │ 2 rephrase loops                            │
│ LLM Calls/Run  │ 2–4 (agent + optional rephrase + answer)    │
│ Web Calls/Run  │ 2–6 (2-3 sources per tool)                  │
│ Entry Point    │ run_agent(query: str) → str                 │
├────────────────┼────────────────────────────────────────────┤
│ Node Order     │ agent → tool → evaluator → answer → END    │
│ Loop Path      │ evaluator → rephraser → agent (max 2x)     │
│ State Key      │ AgentState (7 fields)                       │
│ Tool Registry  │ TOOLS dict {name: function}                 │
└────────────────┴────────────────────────────────────────────┘
```

---

*Documentation written for the Sports Research Agent (2026). Built with LangGraph + Groq + BeautifulSoup.*
