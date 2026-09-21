# RAG, RAG FINE-TUNING, AI AGENTS & PROJECT COST MANAGEMENT

INTERVIEW PREPARATION GUIDE
Level: Intermediate to Advanced
Focus: AI/LLM Engineering, RAG Systems, Fine-Tuning, Agents, and Cost Management

## TABLE OF CONTENTS

1. Executive Overview
2. Core LLM Concepts
3. RAG Fundamentals
4. Advanced RAG
5. RAG Fine-Tuning
6. RAG vs Fine-Tuning vs Prompt Engineering
7. AI Agents
8. Agent + RAG Architecture
9. RAG Fine-Tuning Interview Questions
10. Agents Interview Questions
11. RAG Interview Questions
12. System Design Questions
13. Project Cost Management
14. Cost Estimation Framework
15. Production Optimization
16. Practical Project Blueprint
17. Scenario-Based Interview Questions
18. Quick Revision Cheat Sheet
19. Sample 60-Second Interview Answer
20. Final Interview Checklist

# 1. EXECUTIVE OVERVIEW

Retrieval-Augmented Generation (RAG) is an architecture in which an LLM retrieves relevant information from an external knowledge source and uses that information to generate an answer.

A typical RAG flow is:

User Query
   |
   v
Query Processing
   |
   v
Retriever -> Vector DB / Search Engine
   |
   v
Relevant Documents / Chunks
   |
   v
Reranker (optional)
   |
   v
Prompt Construction
   |
   v
LLM
   |
   v
Grounded Answer + Citations

RAG is especially useful when:

- Knowledge changes frequently.
- Data is private or enterprise-specific.
- Answers must be grounded in source documents.
- Re-training the model for every knowledge update is impractical.
- Source citations or traceability are important.

Fine-tuning is useful when the main problem is model behavior rather than knowledge retrieval. Examples include:

- Domain-specific response style.
- Structured output behavior.
- Classification.
- Tool-calling behavior.
- Consistent instruction following.
- Specialized task performance.

A strong production system often combines:
RAG + prompt engineering + evaluation + optional fine-tuning + observability.

# 2. CORE LLM CONCEPTS

## 2.1 Tokens
LLMs process text as tokens rather than complete words. Token count affects:

- Input cost.
- Output cost.
- Context-window usage.
- Latency.

Interview point:
Reducing unnecessary context can directly reduce inference cost.

## 2.2 Embeddings
An embedding converts text into a numerical vector representing semantic meaning.

Example:
"How do I reset my password?"
and
"I forgot my password. What is the recovery process?"

These sentences can have similar embeddings even though their wording differs.

Embeddings are commonly used for:

- Semantic search.
- RAG retrieval.
- Similarity matching.
- Clustering.
- Recommendation systems.

## 2.3 Vector Similarity
Common similarity metrics:

- Cosine similarity.
- Dot product.
- Euclidean distance.

Cosine similarity measures the angle between vectors and is frequently used for text embeddings.

## 2.4 Context Window
The context window is the amount of information the model can process in one request.

Large context does NOT automatically mean better RAG.

Too much retrieved content can:

- Increase cost.
- Increase latency.
- Introduce irrelevant information.
- Make the model less focused.

# 3. RAG FUNDAMENTALS

## 3.1 Basic RAG Pipeline

1. Collect documents.
2. Parse documents.
3. Clean and normalize text.
4. Split text into chunks.
5. Generate embeddings.
6. Store embeddings and metadata.
7. Retrieve relevant chunks.
8. Optionally rerank results.
9. Build an LLM prompt.
10. Generate an answer.
11. Return citations/source references.
12. Evaluate and monitor the system.

## 3.2 Document Chunking

Chunking is one of the most important RAG design decisions.

Common approaches:

- Fixed-size chunks.
- Token-based chunks.
- Sentence-based chunks.
- Paragraph-based chunks.
- Semantic chunking.
- Structure-aware chunking.

Chunking trade-off:

Small chunks:

- Precise retrieval.
- May lose context.

Large chunks:

- More context.
- More irrelevant information.
- Higher token cost.

A common production strategy is to start with a moderate chunk size and evaluate it using the actual document collection and query distribution.

## 3.3 Chunk Overlap
Overlap preserves information across chunk boundaries.

Example:
Chunk 1: tokens 1-500
Chunk 2: tokens 450-950

Overlap can help preserve continuity, but excessive overlap:

- Duplicates content.
- Increases storage.
- Increases retrieval noise.
- Increases cost.

## 3.4 Metadata
Useful metadata includes:

- Document ID.
- Source URL.
- Page number.
- Section.
- Department.
- Product.
- Customer/tenant.
- Timestamp.
- Access-control information.

Metadata filtering can significantly improve retrieval quality.

## 3.5 Retrieval Strategies
A production system may use:

A. Dense retrieval
Uses embeddings and semantic similarity.

B. Sparse retrieval
Uses keyword-based methods such as BM25.

C. Hybrid retrieval
Combines dense + sparse search.

D. Reranking
Retrieves a candidate set and uses a stronger relevance model to reorder results.

A common architecture is:

Hybrid Retrieval
      |
Top 20-50 candidates
      |
Reranker
      |
Top 3-8 chunks
      |
LLM

# 4. ADVANCED RAG

## 4.1 Query Rewriting
The original user query can be transformed into a better search query.

Example:
User:
"What happened to the project budget?"

Rewritten:
"Project budget variance, approved budget, actual spend, forecast, and cost overrun."

Benefits:

- Better retrieval.
- Better handling of vague questions.

## 4.2 Query Expansion
Generate multiple search queries for one user question.

Useful when:

- Terminology varies.
- Documents use different language.
- The user query is short.

## 4.3 Multi-Query RAG
Instead of one retrieval query:

Query
 -> Q1
 -> Q2
 -> Q3
 -> retrieve
 -> merge
 -> rerank

## 4.4 Parent-Child Retrieval
Retrieve a small child chunk but return the larger parent section to the LLM.

This can provide:

- Precise retrieval.
- Better context.

## 4.5 Hierarchical Retrieval
Search from:
document -> section -> subsection -> chunk

Useful for large enterprise documentation.

## 4.6 Contextual Compression
Retrieve documents and then remove irrelevant parts before sending them to the LLM.

This reduces:

- Prompt size.
- Cost.
- Noise.

## 4.7 Graph RAG
Graph-based retrieval represents entities and relationships explicitly.

Useful for questions involving:

- Relationships.
- Dependencies.
- Multi-hop reasoning.
- Enterprise knowledge graphs.

Graph RAG is not automatically better than standard vector RAG. It should be selected based on the query type and data structure.

# 5. RAG FINE-TUNING

IMPORTANT:
"Fine-tuning RAG" can mean different things.

RAG itself is usually an architecture, not a single model. Fine-tuning may target different components.

## 5.1 What Can Be Fine-Tuned?

A. Embedding model
Goal:
Improve semantic representation for a specific domain.

B. Retrieval model
Goal:
Improve which documents/chunks are retrieved.

C. Reranker
Goal:
Improve ordering of retrieved candidates.

D. Generator LLM
Goal:
Improve response behavior, style, format, domain task performance, or instruction following.

E. Query rewriting model
Goal:
Generate better retrieval queries.

## 5.2 Retrieval Fine-Tuning
Training examples can look like:

Query:
"What is the approved cloud infrastructure budget?"

Positive passage:
"FY26 cloud infrastructure budget approved at ..."

Hard negative:
"FY26 software licensing budget ..."

Training objective:
Make the positive document closer to the query than irrelevant but semantically similar documents.

This is especially useful when generic embeddings perform poorly on specialized terminology.

## 5.3 Contrastive Learning
A common retrieval fine-tuning setup uses:

- Query.
- Positive document.
- Negative document(s).

The model learns:
similarity(query, positive) > similarity(query, negative)

Hard negatives are especially valuable because easy negatives do not challenge the retriever.

## 5.4 Generator Fine-Tuning
A generator can be fine-tuned on examples such as:

Context:
[retrieved enterprise documents]

Question:
[customer question]

Ideal Answer:
[grounded, structured answer]

The goal should generally be improved behavior, not memorization of frequently changing enterprise facts.

## 5.5 Fine-Tuning vs RAG

Use RAG when:

- Knowledge changes frequently.
- Documents are private.
- Citations are required.
- Knowledge needs to be updated without retraining.

Use fine-tuning when:

- Behavior needs to change.
- Output format must be consistent.
- A specialized task needs better performance.
- Domain-specific language/style is important.

Use both when:

- RAG supplies current facts.
- Fine-tuning improves how the model uses those facts.

# 6. RAG VS FINE-TUNING VS PROMPT ENGINEERING

Prompt Engineering:

- Fastest to implement.
- Low training cost.
- Good for behavior/instruction improvements.
- Limited by model capabilities.

RAG:

- Externalizes knowledge.
- Good for private/current data.
- Supports citations.
- Requires retrieval infrastructure.

Fine-Tuning:

- Changes model behavior/weights.
- Requires curated training data.
- Has training and evaluation cost.
- Can improve consistency for specific tasks.

A useful decision rule:

Problem = Missing/Changing Knowledge?
-> RAG

Problem = Model Behavior/Style/Task Skill?
-> Fine-Tuning or Prompt Engineering

Problem = Both?
-> RAG + Fine-Tuning

# 7. AI AGENTS

An AI agent is a system that can reason about a task, decide which actions to take, use tools, observe results, and continue until a goal is reached.

Typical agent loop:

Goal
 |
 v
Plan / Reason
 |
 v
Choose Tool
 |
 v
Tool Execution
 |
 v
Observe Result
 |
 v
Evaluate Next Step
 |
 +----> Continue
 |
 v
Final Answer

## 7.1 Agent Components

- LLM.
- System instructions.
- Tools.
- Memory/state.
- Planning/reasoning logic.
- Retrieval.
- Guardrails.
- Observability.
- Evaluation.

## 7.2 Tool Examples

- Search.
- Database query.
- Calculator.
- Python/code execution.
- CRM API.
- Ticketing system.
- Email.
- Internal enterprise API.

## 7.3 Agent vs RAG
RAG answers:
"Which information should I retrieve to answer this question?"

An agent handles:
"What sequence of actions should I take to complete this task?"

RAG can be one tool inside an agent.

Example:

User:
"Analyze our Q3 project costs, find projects with overruns, and create a summary."

Agent may:

1. Query project database.
2. Retrieve project documentation.
3. Calculate cost variance.
4. Identify anomalies.
5. Generate summary.
6. Produce recommendations.

# 8. AGENT + RAG ARCHITECTURE

User
 |
 v
Agent Orchestrator
 |
 +------> Retrieval Tool ------> Vector DB
 |
 +------> SQL Tool ------------> Project DB
 |
 +------> Cost Calculator
 |
 +------> Document Search
 |
 v
Observation / State
 |
 v
LLM Decision
 |
 v
Final Answer / Action

Important design principle:
Do not let an agent freely execute every available tool.

Use:

- Tool permissions.
- Input validation.
- Timeouts.
- Rate limits.
- Human approval for sensitive actions.
- Audit logs.

# 9. RAG FINE-TUNING INTERVIEW QUESTIONS

## Q1. What is RAG?
Answer:
RAG is an architecture that retrieves relevant external knowledge and provides it to an LLM as context before generation. It improves grounding and allows knowledge to be updated without retraining the generator.

## Q2. Why not simply fine-tune the model on company documents?
Answer:
Fine-tuning is not an ideal mechanism for frequently changing factual knowledge. It can also make source attribution difficult. RAG keeps knowledge external and retrievable, while fine-tuning is better suited for behavior and task specialization.

## Q3. What does it mean to fine-tune a RAG system?
Answer:
It can mean fine-tuning the embedding model, retriever, reranker, query-rewriting model, or generator depending on where the quality problem exists.

## Q4. How do you create a dataset for retrieval fine-tuning?
Answer:
Create query-positive document pairs and add hard negatives. Queries should represent real production questions. Labels can come from human annotations, existing search behavior, synthetic data followed by validation, or domain experts.

## Q5. What are hard negatives?
Answer:
Hard negatives are documents that appear relevant but are actually incorrect. They teach the retriever to distinguish subtle semantic differences.

## Q6. How do you evaluate retrieval?
Key metrics:

- Recall@K.
- Precision@K.
- MRR.
- NDCG.
- Hit Rate.

Recall@K:
Did the correct document appear within the top K retrieved results?

## Q7. How do you evaluate the final RAG answer?
Measure:

- Answer correctness.
- Faithfulness/groundedness.
- Context relevance.
- Citation accuracy.
- Completeness.
- Latency.
- Cost.

## Q8. What if retrieval is good but the answer is wrong?
Investigate:

- Prompt construction.
- Context ordering.
- Context length.
- Generator behavior.
- Contradictory documents.
- Model limitations.

## Q9. What if retrieval is bad?
Investigate:

- Chunking.
- Embedding model.
- Query formulation.
- Metadata filters.
- Search strategy.
- Reranker.
- Document quality.

## Q10. When would you fine-tune embeddings?
When generic embeddings fail to capture domain-specific terminology, relationships, or relevance patterns, and sufficient labeled training data is available.

# 10. AGENTS INTERVIEW QUESTIONS

## Q1. What is an AI agent?
An AI agent is a system that uses an LLM and tools to decide and execute a sequence of actions toward a goal.

## Q2. Agent vs chatbot?
A chatbot primarily generates conversational responses. An agent can perform actions through tools and maintain task state.

## Q3. What is tool calling?
Tool calling allows the model to request a structured function/API invocation.

## Q4. What are agent failure modes?

- Infinite loops.
- Wrong tool selection.
- Hallucinated tool arguments.
- Tool failures.
- Prompt injection.
- Excessive cost.
- Unbounded execution.
- Incorrect assumptions.

## Q5. How do you control agent cost?

- Limit steps.
- Limit tool calls.
- Use smaller models for simple tasks.
- Cache results.
- Summarize state.
- Avoid repeated retrieval.
- Use deterministic workflows where possible.

## Q6. When should you NOT use an agent?
Do not use an agent when a deterministic workflow solves the problem more reliably, cheaply, and transparently.

# 11. RAG INTERVIEW QUESTIONS

## Q1. Why is chunking important?
Because retrieval operates on chunks. Poor chunk boundaries can separate important context or create overly broad results.

## Q2. What is hybrid search?
Combining keyword/sparse retrieval with semantic/dense retrieval.

## Q3. Why use a reranker?
The initial retriever may produce many approximate candidates. A reranker can use a stronger relevance model to reorder them.

## Q4. How many chunks should be retrieved?
There is no universal number. Start with a small candidate set, evaluate Recall@K and answer quality, then optimize. More chunks can increase noise and cost.

## Q5. What is RAG hallucination?
An answer that contains unsupported or incorrect information despite using retrieved context.

## Q6. How can hallucinations be reduced?

- Better retrieval.
- Strong grounding instructions.
- Citation requirements.
- Refusal when evidence is insufficient.
- Context filtering.
- Evaluation.
- Fine-tuning when appropriate.

## Q7. What is a vector database?
A database or search system optimized for storing and querying vectors, often with metadata filtering and similarity search.

# 12. SYSTEM DESIGN QUESTIONS

Scenario:
"Design a company-wide knowledge assistant."

Suggested architecture:

Data Sources
  |
  v
Ingestion Pipeline
  |
  +--> PDF/HTML/Office Parsing
  |
  +--> Cleaning
  |
  +--> Chunking
  |
  +--> Metadata
  |
  +--> Embeddings
  |
  v
Vector / Hybrid Search Index
  |
  v
Retriever
  |
  v
Reranker
  |
  v
Context Builder
  |
  v
LLM
  |
  v
Answer + Citations

Production requirements:

- Authentication.
- Authorization.
- Tenant isolation.
- Document-level ACLs.
- Encryption.
- Monitoring.
- Logging.
- Evaluation.
- PII/security controls.
- Rate limiting.
- Cost controls.

Interview follow-up:
"What happens when a document is updated?"

Good answer:
Use document IDs and versioning. Re-process the changed document, replace or version its chunks/embeddings, and invalidate affected caches.

# 13. PROJECT COST MANAGEMENT

AI project cost has several layers:

1. Development cost.
2. Data preparation cost.
3. Embedding cost.
4. Vector database/search cost.
5. LLM inference cost.
6. Fine-tuning training cost.
7. Agent/tool execution cost.
8. Infrastructure cost.
9. Monitoring/evaluation cost.
10. Human review/annotation cost.

## 13.1 Main LLM Cost Drivers
LLM cost approximately depends on:

Cost =
(input tokens / 1,000,000 * input price)
+
(output tokens / 1,000,000 * output price)

Actual pricing depends on the selected model/provider and can change over time.

## 13.2 RAG Cost Drivers
RAG adds:

- Document ingestion.
- Embedding generation.
- Vector storage.
- Search queries.
- Reranking.
- Larger prompts due to retrieved context.

## 13.3 Agent Cost
Agent cost can multiply because one user request may trigger multiple model calls.

Approximate:

Total request cost =
sum(all LLM calls)

- embeddings
- reranking
- tool/API costs
- infrastructure allocation

Example:

One agent request:
Call 1: Planning
Call 2: Retrieval reasoning
Call 3: Tool selection
Call 4: Final response

Four model calls may cost much more than a single direct generation request.

# 14. COST ESTIMATION FRAMEWORK

Before production, estimate:

A. Traffic

- Requests/day.
- Peak requests/minute.
- Average requests/user.

B. Token usage

- Average input tokens/request.
- Average output tokens/request.
- Retrieved context tokens/request.

C. Model usage

- Model for routing.
- Model for retrieval/query rewriting.
- Model for generation.
- Model for evaluation.

D. Infrastructure

- Vector DB.
- Compute.
- Object storage.
- Databases.
- Monitoring.

E. Training

- Fine-tuning compute.
- Dataset preparation.
- Annotation.
- Evaluation.

Example estimation:

Daily requests = 10,000
Average input = 4,000 tokens
Average output = 800 tokens

Daily input tokens:
10,000 * 4,000 = 40,000,000 tokens

Daily output tokens:
10,000 * 800 = 8,000,000 tokens

Monthly approximate tokens using 30 days:
Input = 1.2B tokens
Output = 240M tokens

Then multiply these volumes by the current model's input/output pricing.

Important:
Never give a project estimate using old model pricing. Confirm current provider pricing before presenting a commercial quote.

# 15. PRODUCTION OPTIMIZATION

## 15.1 Reduce Token Usage

- Retrieve fewer but better chunks.
- Compress context.
- Remove duplicated content.
- Summarize long documents.
- Use structured prompts.
- Limit output length.

## 15.2 Use Model Routing
Simple task -> small/cheap model
Complex task -> stronger model

Example:

Classifier
   |
   +-- Simple FAQ -> Small LLM
   |
   +-- Complex analysis -> Large LLM

## 15.3 Cache
Cache:

- Embeddings.
- Retrieval results where appropriate.
- Repeated tool calls.
- Stable system instructions.
- Frequently requested answers where safe.

## 15.4 Batch Processing
Batch embeddings and offline processing where possible.

## 15.5 Control Agent Loops
Set:

- Maximum steps.
- Maximum tokens.
- Maximum tool calls.
- Timeout.
- Budget per request.

## 15.6 Cost Observability
Track:

- Cost/request.
- Tokens/request.
- Retrieval latency.
- LLM latency.
- Tool latency.
- Error rate.
- Agent steps.
- Cache hit rate.

# 16. PRACTICAL PROJECT BLUEPRINT

PROJECT:
Enterprise Project Cost Management Assistant

Business Goal:
Allow project managers to ask questions about budgets, forecasts, invoices, project documents, and cost overruns.

Data:

- Project budgets.
- Actual costs.
- Forecasts.
- Invoices.
- Contracts.
- Project status reports.
- Meeting notes.

Architecture:

User
 |
 v
Web/API
 |
 v
Agent Orchestrator
 |
 +--> SQL Tool -----------------> Project Cost DB
 |
 +--> RAG Tool -----------------> Vector/Hybrid Search
 |
 +--> Calculator ----------------> Cost Variance
 |
 +--> Document Tool -------------> Source Documents
 |
 v
LLM
 |
 v
Answer + Evidence + Calculations

Example question:
"Which projects are more than 10% over budget, and what are the likely reasons?"

Possible agent execution:

1. Query current budget and actual spend.
2. Calculate variance percentage.
3. Filter projects > 10% over budget.
4. Retrieve project reports for those projects.
5. Search for causes such as scope change, vendor increase, delay, or resource growth.
6. Summarize evidence.
7. Return a table with citations.

Important:
Numerical calculations should preferably be performed using deterministic code/SQL rather than relying on the LLM's arithmetic.

# 17. SCENARIO-BASED INTERVIEW QUESTIONS

Scenario 1:
"The RAG system has poor answers. Where do you start?"

Strong answer:
Start with evaluation and separate retrieval quality from generation quality. Check whether the correct evidence is retrieved. If not, investigate chunking, embeddings, query transformation, hybrid search, metadata filters, and reranking. If the evidence is correct but the answer is wrong, investigate prompt construction, context ordering, model behavior, and grounding.

Scenario 2:
"Your RAG system is accurate but too expensive."

Answer:
Measure token usage first. Reduce retrieved context, use contextual compression, reduce top-K, cache embeddings/retrieval where safe, route simple requests to cheaper models, reduce unnecessary agent steps, and monitor cost per successful answer.

Scenario 3:
"Should we fine-tune the LLM on all company documents?"

Answer:
Usually no. Current company knowledge is better stored in a retrievable system. Fine-tuning should target behavior, style, task specialization, or structured output rather than frequently changing factual knowledge.

Scenario 4:
"The agent is making too many tool calls."

Answer:
Add explicit execution limits, tool-selection constraints, routing, caching, state summarization, and deterministic workflows for known task paths. Measure tool-call frequency and cost per completed task.

Scenario 5:
"How would you prove your RAG improvement works?"

Answer:
Create a fixed evaluation set with representative queries. Measure retrieval metrics and end-to-end metrics before and after the change. Compare answer correctness, groundedness, citation quality, latency, and cost. Use statistical or repeated evaluation where appropriate rather than relying on a few examples.

# 18. QUICK REVISION CHEAT SHEET

## RAG
RAG = Retrieve + Context + Generate

Best for:
Current/private/large external knowledge.

## Retriever
Finds candidate evidence.

## Reranker
Reorders candidates by relevance.

## Embedding
Maps text to vectors.

## Chunking
Splits documents into retrievable units.

## Hybrid Search
Dense semantic + sparse keyword retrieval.

## Fine-Tuning
Changes model parameters to improve behavior/task performance.

## Retrieval Fine-Tuning
Improves which evidence is retrieved.

## Hard Negative
Looks relevant but is actually incorrect.

## Agent
LLM + Tools + State + Decision Loop.

## Tool Calling
Structured request to execute an external function/API.

## Agent Guardrails
Permissions + validation + limits + monitoring.

## RAG Evaluation
Recall@K, Precision@K, MRR, NDCG, faithfulness, answer correctness.

## Cost Optimization
Fewer tokens + fewer calls + cheaper routing + caching + deterministic tools.

# 19. SAMPLE 60-SECOND INTERVIEW ANSWER

Question:
"Explain how you would build a production RAG system and optimize it."

Answer:

"I would first build a document ingestion pipeline with parsing, cleaning, structure-aware chunking, metadata, and embeddings. For retrieval, I would start with hybrid search and add a reranker if evaluation shows a relevance gap. I would keep the retrieved context small and grounded, then pass it to the LLM with source references.

For evaluation, I would maintain a representative test set and separately measure retrieval quality and generation quality using metrics such as Recall@K, answer correctness, groundedness, and citation accuracy.

If the problem is domain-specific retrieval, I would consider fine-tuning the embedding or reranking model using query-positive pairs and hard negatives. I would fine-tune the generator only when the problem is model behavior, such as structured output or task-specific instruction following.

For production, I would monitor latency, token usage, cost per request, retrieval quality, errors, and user feedback. I would reduce cost through context compression, model routing, caching, and controlled agent/tool execution."

# 20. FINAL INTERVIEW CHECKLIST

Before an interview, be able to explain:

[ ] What is RAG?
[ ] Why use RAG instead of fine-tuning for changing knowledge?
[ ] What is an embedding?
[ ] How does vector search work?
[ ] What is chunking and why does it matter?
[ ] Dense vs sparse vs hybrid retrieval.
[ ] What is a reranker?
[ ] What are hard negatives?
[ ] How do you fine-tune a retriever?
[ ] How do you fine-tune a generator?
[ ] How do you evaluate retrieval?
[ ] How do you evaluate RAG answers?
[ ] What is an AI agent?
[ ] Agent vs RAG.
[ ] How does tool calling work?
[ ] Agent failure modes.
[ ] Agent guardrails.
[ ] How do you control agent cost?
[ ] How do you estimate LLM cost?
[ ] How do you reduce token usage?
[ ] How do you design a production RAG architecture?
[ ] How do you handle document updates?
[ ] How do you enforce access control?
[ ] How do you monitor a production system?
[ ] When should you avoid using an agent?
[ ] When should you combine RAG and fine-tuning?

## KEY TAKEAWAY

A strong AI engineer does not start by asking:
"Which model should I fine-tune?"

Instead, start with:

1. What business problem are we solving?
2. Is the problem knowledge, behavior, reasoning, or workflow?
3. What evidence is required?
4. What is the simplest architecture that can solve it?
5. How will we evaluate quality?
6. What is the latency and cost budget?
7. What security and access controls are required?
8. How will the system behave when retrieval or tools fail?

The best production architecture is usually the simplest system that reliably meets quality, security, latency, and cost requirements.

## End Of Document
