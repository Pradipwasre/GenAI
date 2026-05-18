# Multimodal RAG — A Complete Technical Guide

---

## Table of Contents

1. What is RAG
2. What is Vanilla RAG and How It Works
3. Limitations of Vanilla RAG
4. What is Multimodal RAG
5. How Multimodal RAG Works — Step by Step
6. Architecture Diagram (Text)
7. Why CLIP is the Core of Multimodal RAG
8. Embedding Models — Free and Paid
9. Vector Databases Compatible with Multimodal RAG
10. Vanilla RAG vs Multimodal RAG — Full Comparison
11. When to Use Multimodal RAG
12. Client Scenario — Adani Green Energy Annual Report
13. Sample Queries and Expected Answers
14. Limitations and Considerations

---

## 1. What is RAG

Retrieval-Augmented Generation (RAG) is a design pattern that combines two systems — a retrieval system and a generative language model — to produce answers that are grounded in a specific knowledge base rather than relying solely on the model's training data.

The core idea is simple: instead of asking an LLM to answer from memory, you first retrieve the most relevant pieces of information from your own documents, then pass those pieces as context to the LLM so it can generate an accurate, sourced answer.

RAG solves three fundamental problems with plain LLMs:

- Hallucination — LLMs can confidently generate incorrect facts. RAG anchors answers to real documents.
- Knowledge cutoff — LLMs have a training date after which they know nothing. RAG gives them access to current documents.
- Private data — LLMs are not trained on your company's internal documents. RAG bridges that gap without fine-tuning.

---

## 2. What is Vanilla RAG and How It Works

Vanilla RAG refers to the standard, text-only retrieval-augmented generation pipeline. It processes only textual content from documents.

The pipeline has five stages:

**Stage 1 — Document Ingestion**
PDF, Word, or text files are loaded. The raw text is extracted page by page.

**Stage 2 — Chunking**
The extracted text is split into smaller overlapping chunks, typically 300 to 1000 characters, using a splitter like RecursiveCharacterTextSplitter. Chunking ensures that no single piece is too large for the embedding model's token limit.

**Stage 3 — Text Embedding**
Each chunk is converted into a dense vector (a list of floating point numbers) using a text embedding model such as OpenAI text-embedding-3-small or sentence-transformers. Chunks with similar meaning produce vectors that are close together in vector space.

**Stage 4 — Vector Storage**
All chunk vectors are stored in a vector database such as FAISS, Chroma, Pinecone, or Weaviate, paired with the original text and metadata (page number, source file, etc.).

**Stage 5 — Retrieval and Generation**
When a user submits a query, the query is embedded using the same model. The top-k most similar vectors are retrieved from the store. The original text of those chunks is assembled into a context block and sent to the LLM with the query. The LLM generates an answer using only the provided context.

This works well when all the meaningful information in a document is in its text. For most text-heavy documents — legal contracts, research papers, policy documents — vanilla RAG is sufficient.

---

## 3. Limitations of Vanilla RAG

Vanilla RAG completely ignores any content that is not text. This creates significant blind spots in real-world documents.

**Charts and graphs** — A revenue trend chart, a market share pie graph, or a risk heatmap contains information that is not written anywhere in the document as text. Vanilla RAG cannot retrieve or interpret this.

**Tables rendered as images** — Many PDFs, especially annual reports and government publications, embed tables as scanned images rather than structured text. Vanilla RAG sees nothing in them.

**Infographics and diagrams** — System architecture diagrams, process flow charts, and product roadmap visuals carry meaning that has no textual equivalent in the document.

**Scanned or photographed PDFs** — If the PDF is a scan of a physical document, there is no extractable text at all. Vanilla RAG returns empty results.

**Mixed documents** — Real business documents — annual reports, investor presentations, technical specifications — are a mix of prose, tables, charts, and images. Vanilla RAG handles only one part of the document.

The consequence is that a user asking "What does the revenue chart show?" or "Describe the organisational structure diagram" receives either no answer or a hallucinated one, because the system never saw the visual content.

---

## 4. What is Multimodal RAG

Multimodal RAG extends the vanilla pipeline to handle multiple data modalities — text, images, tables, charts, diagrams — within the same unified retrieval and generation system.

The key insight is that the embedding model must be able to produce vectors for both text and images in the same vector space. This is what CLIP (Contrastive Language-Image Pretraining) achieves. Because text and image embeddings live in the same space, a text query like "revenue growth chart" can retrieve an image of a bar chart that visually depicts revenue growth.

The LLM at the end of the pipeline must also be multimodal — capable of receiving both text and images as input. Models such as GPT-4 Vision, GPT-4.1, Claude 3, and Llava fulfill this role.

The result is a system where:
- A text query can retrieve both relevant text chunks and relevant images
- An image query (or a description of a visual concept) can retrieve charts and diagrams
- The LLM receives the full retrieved context — text and images together — and generates an answer that synthesises both

---

## 5. How Multimodal RAG Works — Step by Step

**Step 1 — PDF Parsing**
The PDF is opened with a library such as PyMuPDF (fitz). For each page, two things are extracted separately:
- The raw text of the page
- All embedded images (charts, diagrams, photos, scanned tables) as raw byte streams

**Step 2 — Text Chunking**
The extracted text from each page is split into overlapping chunks using a text splitter. Each chunk is stored as a Document object with metadata: page number, document type (text), and source file.

**Step 3 — Image Processing**
Each extracted image is:
- Converted from raw bytes to a PIL Image object for CLIP processing
- Saved as a base64-encoded PNG string for later use with GPT-4 Vision (which requires base64 images in its API payload)
- Assigned a unique identifier such as page_2_img_0

**Step 4 — Unified CLIP Embedding**
Both text chunks and images are embedded using CLIP into the same 512-dimensional vector space.
- Text chunks: passed through CLIP's text encoder
- Images: passed through CLIP's image encoder
- All embeddings are L2-normalised (unit vectors) so cosine similarity equals dot product

This is the critical difference from vanilla RAG. A single embedding model produces comparable vectors for both modalities.

**Step 5 — FAISS Index Construction**
All vectors — from both text chunks and images — are inserted into a single FAISS index. The index stores the vector alongside the Document object (text content or image placeholder) and metadata.

**Step 6 — Query Embedding and Retrieval**
When a user submits a query:
- The query string is embedded using CLIP's text encoder
- The FAISS index is searched for the top-k nearest neighbours
- Results may include a mix of text chunks and image placeholders, ranked by cosine similarity

**Step 7 — Multimodal Message Construction**
The retrieved documents are separated into text chunks and image chunks. A structured message is assembled:
- The user query
- All retrieved text chunks as plain text context
- All retrieved images as base64-encoded image_url blocks

This message is formatted according to the OpenAI Vision API schema (or the equivalent for the chosen LLM).

**Step 8 — LLM Generation**
The multimodal message is sent to a vision-capable LLM such as GPT-4.1 or Llava. The LLM reads both the text context and interprets the chart images, then generates a grounded answer that draws from both sources.

---

## 6. Architecture — Text Representation

```
PDF Document
      |
      +------ PyMuPDF ------+
      |                     |
  Text Pages            Embedded Images
      |                     |
  Chunking (500c)       PIL Conversion
      |                     |
  CLIP Text Encoder     CLIP Image Encoder
      |                     |
  512-dim vector        512-dim vector
      |                     |
      +-------- FAISS Index (Unified) --------+
                            |
                      User Query
                            |
                  CLIP Text Encoder
                            |
                    512-dim query vector
                            |
              FAISS similarity_search_by_vector
                            |
              Top-k Results (text + image mix)
                            |
              Multimodal Message Builder
              (text context + base64 images)
                            |
              GPT-4V / GPT-4.1 / Llava / Claude
                            |
                    Grounded Answer
```

---

## 7. Why CLIP is the Core of Multimodal RAG

CLIP was developed by OpenAI and trained on 400 million image-text pairs from the internet. The training objective was contrastive: the model learned to pull the embedding of an image and its caption close together in vector space, while pushing apart unrelated image-text pairs.

The result is a shared semantic space where:
- The embedding of a bar chart showing revenue growth is geometrically close to the embedding of the phrase "revenue growth bar chart"
- The embedding of a pie chart showing shareholding is close to the phrase "shareholding pattern breakdown"
- An image of a financial table is close to text describing financial ratios

This cross-modal alignment is what makes unified retrieval possible. Without a model like CLIP, you would need two separate retrieval systems — one for text and one for images — and a separate ranking layer to merge results.

CLIP produces 512-dimensional embeddings with the base model (clip-vit-base-patch32) and 768-dimensional embeddings with the large model (clip-vit-large-patch14). The base model is sufficient for most RAG applications and runs efficiently on CPU.

---

## 8. Embedding Models — Free and Paid

### Free and Open Source Models

**openai/clip-vit-base-patch32**
Provider: Hugging Face (OpenAI weights, open source)
Dimensions: 512
Modalities: Text and Image (unified)
Notes: The standard choice for multimodal RAG. Runs on CPU. Max text length 77 tokens. Available via the transformers library.

**openai/clip-vit-large-patch14**
Provider: Hugging Face
Dimensions: 768
Modalities: Text and Image (unified)
Notes: Higher quality than base but requires more memory. Suitable when retrieval precision is critical.

**laion/CLIP-ViT-H-14-laion2B-s32B-b79K**
Provider: Hugging Face (LAION)
Dimensions: 1024
Modalities: Text and Image (unified)
Notes: Trained on a much larger dataset than OpenAI CLIP. Stronger cross-modal alignment. GPU recommended.

**sentence-transformers/all-MiniLM-L6-v2**
Provider: Hugging Face
Dimensions: 384
Modalities: Text only
Notes: Fast and lightweight. Not multimodal. Use only if you have a separate image embedding model.

**BAAI/bge-m3**
Provider: Hugging Face
Dimensions: 1024
Modalities: Text only (multilingual)
Notes: Excellent for multilingual documents. No image support.

**nomic-ai/nomic-embed-text-v1**
Provider: Hugging Face / Ollama
Dimensions: 768
Modalities: Text only
Notes: Good open-source text embedder. Use with a separate image encoder for multimodal setups.

### Paid and API-Based Models

**OpenAI text-embedding-3-small**
Provider: OpenAI API
Dimensions: 1536
Modalities: Text only
Cost: USD 0.02 per million tokens
Notes: High quality text embeddings. Does not support images. Pair with CLIP for image embedding in a hybrid setup.

**OpenAI text-embedding-3-large**
Provider: OpenAI API
Dimensions: 3072
Modalities: Text only
Cost: USD 0.13 per million tokens
Notes: Best-in-class text embedding quality. Same caveat on images.

**Google Vertex AI multimodalembedding@001**
Provider: Google Cloud Vertex AI
Dimensions: 1408
Modalities: Text and Image (unified)
Cost: Pay per request (varies by region)
Notes: Direct competitor to CLIP with API convenience. Produces unified embeddings for text, images, and video frames. Strong choice for production systems on GCP.

**Amazon Titan Multimodal Embeddings**
Provider: AWS Bedrock
Dimensions: 1024
Modalities: Text and Image (unified)
Cost: USD 0.00006 per 1000 input tokens, USD 0.00006 per image
Notes: Designed for AWS-native RAG architectures. Integrates with Amazon Kendra and OpenSearch.

**Cohere embed-v3**
Provider: Cohere API
Dimensions: 1024
Modalities: Text only
Cost: USD 0.10 per million tokens
Notes: Strong retrieval-focused text embedder. No native image support.

### Summary Table

| Model | Modality | Dimensions | Cost | Best For |
|---|---|---|---|---|
| CLIP ViT-B/32 | Text + Image | 512 | Free | Multimodal RAG, local |
| CLIP ViT-L/14 | Text + Image | 768 | Free | Higher quality local |
| LAION CLIP-H | Text + Image | 1024 | Free | Best open source quality |
| OpenAI text-3-small | Text | 1536 | Paid | Text-only RAG via API |
| Vertex multimodal | Text + Image | 1408 | Paid | GCP production systems |
| Amazon Titan MM | Text + Image | 1024 | Paid | AWS production systems |

---

## 9. Vector Databases Compatible with Multimodal RAG

**FAISS**
Type: In-memory / local
Cost: Free (Meta, open source)
Notes: No persistence by default. Fastest for local development and prototyping. Supports precomputed embeddings via from_embeddings. Used in this project.

**Chroma**
Type: Local with optional cloud
Cost: Free (open source core)
Notes: Supports metadata filtering, persistent storage, and LangChain integration. Good for mid-scale projects.

**Pinecone**
Type: Cloud
Cost: Free tier (1 index, 100K vectors), paid from USD 70/month
Notes: Fully managed, production-grade. Excellent for team deployments where uptime matters.

**Weaviate**
Type: Local or Cloud
Cost: Free self-hosted, paid cloud
Notes: Native multimodal support. Can store image vectors and retrieve images directly. Strong choice for true multimodal production RAG.

**Qdrant**
Type: Local or Cloud
Cost: Free self-hosted, paid cloud
Notes: Payload filtering, named vectors (you can store separate text and image vector spaces per document). Well-suited for multimodal where you want separate retrieval channels.

**Milvus**
Type: Local or Cloud
Cost: Free open source
Notes: Enterprise-grade scale. Handles billions of vectors. Supports multiple vector fields per document — ideal for large multimodal corpora.

---

## 10. Vanilla RAG vs Multimodal RAG — Full Comparison

| Dimension | Vanilla RAG | Multimodal RAG |
|---|---|---|
| Input modalities | Text only | Text, Images, Charts, Diagrams, Tables |
| Embedding model | Text encoder (e.g. OpenAI, BGE) | Unified encoder (CLIP, Vertex AI) |
| What gets embedded | Text chunks | Text chunks + image tensors |
| Query type | Text string | Text string (retrieves text and/or images) |
| LLM requirement | Any LLM | Vision-capable LLM (GPT-4V, Claude 3, Llava) |
| Chart interpretation | Not possible | Possible — images sent to vision LLM |
| Scanned PDF handling | Fails silently | Works with OCR or visual embedding |
| Annual report coverage | Partial (text sections only) | Complete (text + all visual content) |
| Setup complexity | Low | Medium |
| Inference cost | Lower | Higher (larger payloads to vision LLM) |
| Local/offline use | Yes | Yes (CLIP + Llava fully local) |
| Best document types | Legal docs, articles, logs | Annual reports, research papers, slide decks |

---

## 11. When to Use Multimodal RAG

Use Multimodal RAG when one or more of the following is true:

- The source documents contain charts, graphs, or data visualisations that carry information not repeated in text
- The documents are investor reports, financial filings, or annual reports where charts are the primary carrier of trend information
- The source is a slide deck (PowerPoint, PDF export) where most content is visual
- The documents include technical diagrams, architecture drawings, or process flow charts
- Users are expected to ask questions about visual elements ("What does the Q3 revenue chart show?")
- The PDF may be partially or fully scanned, with no reliable text layer

Vanilla RAG is sufficient when:

- The document is purely text-based (legal contracts, research articles, transcripts)
- All meaningful information is expressed in prose or structured text
- No charts, images, or diagrams exist in the document
- Cost and latency must be minimised

---

## 12. Client Scenario — Adani Green Energy Annual Report

### Background

The client is a financial research firm that produces investment analysis for institutional and retail investors. Their analysts spend significant time reading annual reports and financial data PDFs from listed companies on NSE and BSE.

The specific use case: the firm receives the Adani Green Energy annual report in PDF format, which contains dense financial text alongside multiple charts — revenue growth bars, operating margin trends, cash flow waterfalls, peer comparison tables, and shareholding pie charts.

Their analysts need to query this document conversationally — asking both factual questions about the financials and interpretive questions about the charts — without reading every page manually.

### What Was Built

A Multimodal RAG system was built on top of the Adani Green Energy annual performance report PDF. The PDF contains:

- 8 pages of structured financial analysis text covering FY2016 to FY2026
- 7 embedded matplotlib chart images: revenue/profit bar chart, OPM line chart, debt/equity bar chart, shareholding pie chart, quarterly results chart, cash flow waterfall, and peer market cap comparison
- Data tables for quarterly results and peer comparison

The CLIP model was used to embed all content into a unified 512-dimensional vector space. A FAISS index was built over the combined text and image embeddings. GPT-4.1 Vision was configured as the generation model, receiving both retrieved text and retrieved chart images as context.

### System Capabilities

Text queries retrieve relevant paragraphs and send them to the LLM for synthesis. Image queries retrieve the relevant chart images and send them to GPT-4 Vision, which interprets the visual and describes what the chart shows. Mixed queries retrieve both text and images and the LLM synthesises an answer from both sources.

---

## 13. Sample Queries and Expected Answers

The following queries demonstrate how the system handles different types of retrieval from the Adani Green Energy PDF.

---

**Query 1 (Text Retrieval)**
"What is the revenue growth of Adani Green Energy from FY20 to FY26?"

Expected Retrieval: Text chunks from Section 2 (Annual Financial Performance)

Expected Answer: Adani Green Energy's revenue grew from Rs.2,549 crores in FY20 to Rs.12,928 crores in FY26, representing approximately a 5x expansion in six years. The 10-year compounded sales growth rate stands at 85%, making it one of the fastest-growing listed energy companies in India. The 3-year CAGR is 18% and the TTM growth rate is 15%, indicating a moderation from peak growth rates as the base becomes larger.

---

**Query 2 (Image Retrieval — Chart Interpretation)**
"What does the operating profit margin chart show?"

Expected Retrieval: The OPM line chart image (Figure 2 in the PDF)

Expected Answer: The operating profit margin chart shows a line plot spanning FY20 to FY26. The OPM started at 57% in FY20, rose to 72% in FY21, then dipped to 68% in FY22 and further to 64% in FY23 during a period of rapid plant construction and integration. From FY24 onwards, margins recovered strongly to 80% in FY24, 79% in FY25, and reached 83% in FY26. The upward trajectory reflects growing scale benefits and the increasing proportion of fully operational low-cost solar assets.

---

**Query 3 (Text Retrieval — Risk Analysis)**
"What are the key risks of investing in Adani Green Energy?"

Expected Retrieval: Text chunks from Section 8 (Key Risks and Investment Outlook)

Expected Answer: The primary risks identified are: high leverage with a debt-to-equity ratio of 5.19x and interest coverage of approximately 1.66x, leaving the company vulnerable to revenue shortfalls or rising interest rates; deeply negative free cash flow of approximately minus Rs.15,774 crores in FY26 making the company dependent on continuous debt market access; a stretched valuation at 122x P/E and 11x book value that is sensitive to any slowdown in growth targets; conglomerate headline risk from the broader Adani Group; possible capitalisation of interest costs that could flatter reported profitability; and volatile effective tax rates that create EPS unpredictability.

---

**Query 4 (Image Retrieval — Shareholding Pie Chart)**
"Describe the shareholding pattern of the company."

Expected Retrieval: The shareholding pie chart image (Figure 6 in the PDF)

Expected Answer: The shareholding pie chart as of March 2026 shows four categories. The promoter group (Adani family) holds the largest slice at 62.44%, shown in dark green. Public and retail shareholders account for 21.92%. Foreign institutional investors (FIIs) hold 11.10%, shown in blue. Domestic institutional investors (DIIs) hold 4.56%, shown in gold. The promoter slice has an exploded separation from the rest, emphasising the dominant controlling stake. Zero shares are pledged by promoters, which the chart caption notes as a governance positive.

---

**Query 5 (Mixed Retrieval — Quarterly Analysis)**
"Why was Q3 FY26 a weak quarter for Adani Green Energy?"

Expected Retrieval: Text chunks from Section 3 + the quarterly chart image (Figure 3)

Expected Answer: Q3 FY26 (October to December 2025) was an anomalous quarter. While revenue was Rs.2,618 crores and operating profit was Rs.2,241 crores at an 86% operating margin, net profit collapsed to just Rs.5 crores. The quarterly chart visually shows this as a near-zero bar for net profit while the sales bar remains at normal levels. The text analysis attributes this to three compounding factors: interest costs of Rs.1,698 crores (the highest single quarter on record), an adverse deferred tax adjustment that created a negative 27% effective tax rate, and the seasonal softness typical of monsoon and early winter quarters in renewable generation. The income statement for that quarter shows profit before tax of negative Rs.135 crores, which was then partially recovered through tax adjustments to produce the marginal Rs.5 crore net profit.

---

**Query 6 (Image Retrieval — Peer Comparison Chart)**
"How does Adani Green Energy compare to NTPC and other peers by market capitalisation?"

Expected Retrieval: The peer comparison horizontal bar chart (Figure 7) + peer table text

Expected Answer: The horizontal bar chart ranks seven power sector companies by market capitalisation. NTPC leads at Rs.3,76,812 crores, shown in blue. Adani Green Energy is second at Rs.2,21,471 crores, highlighted in dark green to distinguish it from peers. JSW Energy and NTPC Green Energy are clustered together in the Rs.89,000 to Rs.92,000 crore range, well below the top two. NHPC follows at Rs.76,985 crores, then NLC India at Rs.47,998 crores, and SJVN at Rs.28,581 crores. Despite being second by market cap, Adani Green trades at the second-highest P/E in the group at 122x, compared to NTPC's 15.6x, reflecting the growth premium the market assigns to pure-play renewable energy companies versus diversified utilities.

---

**Query 7 (Text Retrieval — Balance Sheet)**
"What is the total asset base of Adani Green Energy and how is it structured?"

Expected Retrieval: Text chunks from Section 4 (Balance Sheet and Debt Analysis)

Expected Answer: As of FY26, Adani Green Energy's total assets stand at Rs.1,44,097 crores, up from Rs.18,424 crores in FY20 — a 7.8x expansion in six years. The asset base is heavily weighted toward physical infrastructure: fixed assets (operational power plants) account for Rs.1,02,153 crores, and construction work-in-progress (CWIP, representing plants under active development) contributes Rs.19,016 crores. Together these two items represent over 83% of total assets. Investments stand at Rs.2,983 crores and other assets at Rs.19,945 crores. On the liabilities side, total borrowings of Rs.1,03,545 crores fund the majority of the asset base, consistent with the project finance model used by large-scale renewable infrastructure developers.

---

## 14. Limitations and Considerations

**CLIP token limit**: CLIP's text encoder accepts a maximum of 77 tokens. Longer text chunks must be truncated. This means very long paragraphs may lose information at the tail end. Mitigation: keep chunk size below 300 characters when using CLIP for text embedding, or use a separate high-quality text embedder for text chunks and CLIP only for image embedding.

**Semantic gap for specialised charts**: CLIP was trained on general internet image-text pairs. Highly specialised financial charts or engineering diagrams may not align as well with domain-specific text queries. Fine-tuned CLIP variants or domain-specific contrastive training can improve this.

**Vision LLM cost**: Each query that retrieves images sends base64-encoded PNG data to the vision LLM. A single large chart image can be 200 to 500 KB, increasing API payload size and cost. Use image compression (reduce DPI before base64 encoding) to manage costs.

**Free cash flow dependency**: The entire pipeline assumes that images are embedded in the PDF as native image objects. Scanned PDFs where text and images are flattened into a single raster layer require an OCR step before any meaningful extraction is possible.

**Retrieval quality**: CLIP is not a specialised retrieval model. For production systems, consider using a dedicated text retrieval model for text chunks and CLIP only for the image retrieval path, with a cross-modal reranker to merge results.

**Local vs cloud**: The full pipeline (CLIP + Llava) can run entirely offline on a machine with a GPU. For production deployments requiring high availability and scale, move to Vertex AI multimodal embeddings and GPT-4V via API.

---

*Document prepared for internal technical reference. Data sourced from Screener.in. For educational purposes only.*
