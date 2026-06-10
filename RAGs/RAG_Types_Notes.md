# 📚 RAG Types — Complete Student Notes
### "What Changes = What Makes Each RAG Unique"

---

## 🔑 GOLDEN RULE
> **Don't chase the most powerful RAG — chase the RIGHT RAG for your problem.**
> Start simple → Add layers only when complexity demands it.

---

---

# PAGE 1 — RAG Evolution: What Changes Each Time

---

## 1️⃣ Hybrid RAG
**What Changes:** RETRIEVAL STRATEGY
**Core Idea:** Two retrievers fused into one

| Layer | Detail |
|---|---|
| Sparse Search | BM25 — keyword exact match |
| Dense Search | FAISS / Chroma — semantic vector |
| Fusion Engine | EnsembleRetriever (RRF algorithm) |

**Special Module:** `EnsembleRetriever` from `langchain.retrievers`
**Key Config:** `weights=[0.5, 0.5]` — balance keyword + semantic
**What's Unique:** Retrieval Rank Fusion (RRF) re-ranks merged results intelligently

✅ Best For: Single-domain, precision-critical retrieval (player names + context together)

---

## 2️⃣ Agentic RAG
**What Changes:** REASONING LAYER added on top of retrieval
**Core Idea:** Agent THINKS before and after retrieval

| Layer | Detail |
|---|---|
| Pattern | ReAct = Reason + Act |
| Engine | LangGraph agent node |
| Loop | Think → Retrieve → Observe → Repeat |

**Special Module:** `create_react_agent` from `langgraph.prebuilt`
**What's Unique:** ReAct pattern — agent decides WHEN and WHAT to retrieve
**Note:** Autonomous behavior CAN be added manually here

✅ Best For: Queries that need multi-step reasoning before answering

---

## 3️⃣ Autonomous RAG
**What Changes:** SELF-CORRECTION becomes DEFAULT (not manual)
**Core Idea:** RAG that grades itself and retries automatically

| Layer | Detail |
|---|---|
| Self-Grade | `grade_documents` node |
| Self-Rewrite | `rewrite_query` node |
| Loop Control | Conditional edges in StateGraph |

**Special Module:** `StateGraph` with conditional edges — LangGraph itself is the engine
**What's Unique:** `FINISH` condition + loopback edges built-in by design
**Key Insight:** Agentic RAG + ReAct + self-correction loop = Autonomous RAG essentially

✅ Best For: High-accuracy use cases needing zero manual intervention

---

---

# PAGE 2 — Advanced RAG Architectures

---

## 4️⃣ Supervisor RAG
**What Changes:** ONE LLM BRAIN controls all agents sequentially
**Core Idea:** A supervisor decides who works next based on previous output

| Layer | Detail |
|---|---|
| Router | LLM-based supervisor node |
| Direction | `Command(goto="agent_name")` |
| Termination | `FINISH` keyword stops the chain |
| Type Hint | `Literal["agent1", "agent2", "FINISH"]` |

**Special Modules:** `Command(goto=)` + `FINISH` + `Literal[]` type hint
**What's Unique:** Supervisor THINKS before delegating — not hardcoded rules
**vs Multi-Agent:** Sequential + controlled vs Parallel + independent

✅ Best For: Workflows needing controlled, step-by-step intelligent delegation

---

## 5️⃣ Multi-Agent RAG
**What Changes:** MULTIPLE BRAINS work in PARALLEL
**Core Idea:** Specialized agents run simultaneously, each with own tools + memory

| Layer | Detail |
|---|---|
| Agent Creation | `create_react_agent` per domain |
| Parallel Trigger | `send()` API |
| State Sharing | `MessagesState` |
| Coordination | Supervisor node routes between agents |

**Special Module:** `send()` API from LangGraph — the REAL star (parallel execution)
**What's Unique:** True parallelism — cricket agent + football agent run at same time
**vs Supervisor RAG:** No single sequential brain — agents are independent specialists

✅ Best For: Multi-domain analytics (cricket + football + basketball + tennis simultaneously)

---

## 6️⃣ Hierarchical RAG ⭐ MOST EXTENSIBLE
**What Changes:** ENTIRE ARCHITECTURE becomes layered and structured
**Core Idea:** Multiple supervisors embedded at every layer of the hierarchy

```
TOP LAYER      →  Domain Supervisor (cricket? football? tennis?)
MID LAYER      →  Strategy Supervisor (hybrid? semantic? keyword?)
BOTTOM LAYER   →  Chunk Retriever (actual document fetching)
```

| Layer | What Happens |
|---|---|
| Top Supervisor | Routes by domain |
| Mid Supervisor | Selects retrieval strategy |
| Bottom Layer | Executes hybrid/semantic retrieval |

**Special Power:** Can embed Supervisor RAG + Hybrid RAG + ReAct at EVERY layer
**What's Unique:** Structured intelligence at every level — not just flat routing
**Key Insight:** Hierarchical RAG is the CEILING, not the starting point

✅ Best For: Enterprise-scale, multi-domain, multi-strategy complex systems

---

---

# 🧠 QUICK COMPARISON TABLE

| RAG Type | What Changes | Special Module | Best For |
|---|---|---|---|
| Hybrid | Retrieval strategy | `EnsembleRetriever` | Precision retrieval |
| Agentic | Reasoning added | `create_react_agent` | Multi-step queries |
| Autonomous | Self-correction default | `StateGraph` + conditional edges | Zero-intervention accuracy |
| Supervisor | Sequential LLM routing | `Command(goto=)` + `FINISH` | Controlled delegation |
| Multi-Agent | Parallel specialized agents | `send()` API | Multi-domain parallel tasks |
| Hierarchical | Layered architecture | All of the above combined | Enterprise-scale systems |

---

## ⚡ ReAct Availability
> **ReAct (Reason + Act) is available in ALL RAG types** — it is a reasoning PATTERN, not an architecture. Any RAG can use it.

---

*Notes by: Pradip | Agentic AI Course | LangGraph + LangChain 2026*
