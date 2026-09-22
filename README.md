**<u>Enterprise Business Loans Knowledge Base Engine (RAG & Voice Bot Pipeline)</u>**

An end-to-end Python pipeline designed to ingest, clean, sanitize, deduplicate, and vector-index unstructured business content (web pages, PDFs, application forms, underwriting guidelines, and policy documents).

This system builds a structured, searchable, traceable, and PII-redacted Knowledge Base optimized for downstream usage by Voice Agents, RAG Systems, Copilots, and LLMs.

**Table of Contents**
1.Overview & Objective

2.Input Types Supported

3.Data Collection & Cleaning Architecture

  -Website Extraction & Document Parsing

  -Boilerplate & Noise Removal

  -Failure Handling & Error Flagging

  -Near-Deduplication Engine (MinHash LSH)

  -Standardization & Normalization

  -PII Identification & Protection

4.Knowledge Base Architecture

  -Document Schema & Sample Records

  -Chunking & Metadata Strategy

  -Taxonomy & Source Tracking

  -Indexing, Embedding, Retrieval & Citations

5.Repository Pipeline Files

6.Quickstart Guide

**Overview & Objective**
The objective of this repository is to transform heterogeneous, noisy, and unstructured loan documents into a high-precision knowledge store.

By unifying data extraction, hybrid PII scrubbing, MinHash near-deduplication, and local vector indexing (ChromaDB), this engine guarantees high retrieval precision and compliance for automated voice systems (e.g., Question 1 Voice Bot) and generative AI assistants.


**<u>Data Collection & Cleaning Architecture</u>**

**Website Extraction & Document Parsing**
  -Resilient Web Scraping: Uses requests.Session with urllib3 retry logic (handling 429, 500, 502, 503, 504 HTTP errors) and customized User-Agent headers.

  -HTML Parsing: Leverages unstructured.partition.html to break down web pages into semantic DOM blocks.

  -Robust PDF Extraction: Utilizes a fail-safe hierarchy. pdfplumber extracts structured page text and converts tables into Markdown. If pdfplumber yields low character counts or encounters errors, the pipeline automatically triggers a pypdf.PdfReader fallback.

       +-------------------------+
       |   Inbound PDF Document  |
       +-------------------------+
                    |
                    v
       +-------------------------+
       | Try pdfplumber Engine   | ----> Extract Markdown Tables & Text
       +-------------------------+
                    | (If empty / fails)
                    v
       +-------------------------+
       | Try PyPDF Fallback      | ----> Extract Raw Text
       +-------------------------+

**Boilerplate & Noise Removal**
  -The clean_boilerplate_from_text module applies compiled regular expressions to strip out non-content noise line-by-line:

  -Navigation menus, footers, headers, "skip to main content" links.

  -Repeated PDF page numbers (Page X of Y).

  -Cookie consent banners, copyright notices, terms of service headers.

  -Structural whitespace normalization (collapsing extra newlines and tabs).

**Failure Handling & Error Flagging**
  -Instead of crashing or dropping unreadable sources, the pipeline captures operational issues gracefully:

  -HTTP Errors / Corrupted Files: Logged with status EXTRACTION_FAILED alongside explicit error codes (e.g., CORRUPTED_PDF_BINARY, HTTP_404).

  -Content Validation Audit: Records with low character counts (< 40 characters) or broken formatting (> 30% non-ASCII noise) are assigned status FLAGGED_FOR_REVIEW.

**Near-Deduplication Engine (MinHash LSH)**
  -To eliminate duplicate content across web pages and PDF guides, the system implements MinHash Local Sensitivity Hashing (LSH) via datasketch:

  -Converts text into 3-word shingles (n-grams).

  -Computes a 128-permutation MinHash signature per chunk.

  -Indexes signatures into an LSH structure with a Jaccard similarity threshold of 0.85. Near-duplicates are identified and dropped before vector database insertion.

**Standardization & Normalization**
  -Timestamps: Every record includes ISO 8601 UTC timestamps (created_at).

  -Source Tracking: Unified source strings distinguish web links from PDF page numbers (url#page=X).

  -Versions: Standardized schema versioning (version: "1.0").

**PII Identification & Protection**
  -The PIIRedactor engine combines Regex patterns and spaCy Named Entity Recognition (NER) (en_core_web_sm) to enforce zero-leakage compliance:

  -Regex Redaction: Standardizes and redacts Social Security Numbers ([REDACTED SSN]), Credit Card Numbers, Email Addresses, Phone Numbers, and Employer Identification Numbers ([REDACTED EIN]).

spaCy NER Redaction: Detects PERSON entities ([REDACTED NAME]) and addresses containing numerical digits (GPE, LOC, FAC tagged as [REDACTED ADDRESS]).

Knowledge Base Architecture
Document Schema & Sample Records
Schema Definition
JSON
{
  "record_id": "string (Unique identifier, e.g., kb_web_1a2b3c4d)",
  "parent_doc_id": "string (Groups multi-page chunks, e.g., doc_9f8e7d)",
  "title": "string (Title of document or page number)",
  "content": "string (Cleaned, PII-sanitized markdown/text block)",
  "category": "string (Taxonomy classification path)",
  "source": "string (Source URL or page anchor)",
  "source_type": "string ('website' | 'pdf')",
  "version": "string (Schema versioning)",
  "pii_flag": "boolean (True if PII was detected and redacted)",
  "redacted_pii_types": "array of strings (List of redacted entity labels)",
  "status": "string ('PROCESSED' | 'FLAGGED_FOR_REVIEW' | 'EXTRACTION_FAILED')",
  "extraction_errors": "array of strings (Validation or runtime errors)",
  "created_at": "string (ISO 8601 UTC Timestamp)"
}
Sample Record (PDF Page with Redact & Markdown Table)
JSON
{
  "record_id": "kb_pdf_a1b2c3d4",
  "parent_doc_id": "doc_e5f6g7",
  "title": "Page 1 - Policy & Underwriting Data",
  "content": "SBA 7(a) Eligibility Guidelines.\nContact representative [REDACTED NAME] at [REDACTED EMAIL].\n\n--- Table 1 (Page 1) ---\n| Loan Type | Max Amount | Preferred Credit Score |\n| 7(a) Standard | $5,000,000 | 680+ |",
  "category": "/policy/eligibility",
  "source": "https://seedcorp.com/images/SBA_7a_Lender_TrainingPowerPoint.pdf#page=1",
  "source_type": "pdf",
  "version": "1.0",
  "pii_flag": true,
  "redacted_pii_types": ["[REDACTED NAME]", "[REDACTED EMAIL]"],
  "status": "PROCESSED",
  "extraction_errors": [],
  "created_at": "2026-09-23T02:15:00.000000+00:00"
}
Chunking & Metadata Strategy
Chunk Boundaries: Page-level and section-level chunking. Documents are split logically by HTML elements or single PDF pages to preserve context.

Flattening for Embedding: Prior to vector indexing, records are passed through flatten_json_object() to serialize all key-value pairs into structured prose blocks, preserving contextual metadata inside the vector payload.

Metadata Attachment: Every vector entry retains source_id mapping to maintain end-to-end traceability.

Taxonomy & Source Tracking
Content is indexed under standardized hierarchical taxonomy paths:

/policy/eligibility

/underwriting/guidelines

/faq/qualification

/policy/guaranty_rules

/forms/borrower_application

/checklist/prequalification

Indexing, Embedding, Retrieval & Citations
Embedding Model: SentenceTransformerEmbeddingFunction using all-MiniLM-L6-v2 for fast, lightweight inference.

Vector Database: ChromaDB persistent local storage (./chroma_db).

Retrieval & Citation: Distance metrics (Cosine/Euclidean) score query relevance. Retrieved metadata supplies the downstream Voice Agent or LLM with exact source_id references for verified answer attribution.

Repository Pipeline Files
ingest_and_clean.py: Fetches web and PDF sources, applies PII scrubbing, runs MinHash deduplication, and writes business_loans_knowledge_base.json.

remove_boilerplate.py: Reads the JSON output, removes headers, footers, navigation noise, and exports business_loans_knowledge_base_cleaned.json.

index_chroma.py: Reads the clean JSON dataset, flattens records, generates embeddings, and populates the local ChromaDB database (./chroma_db).

Quickstart Guide
1. Prerequisites & Installation
Ensure you have Python 3.9+ installed, then install the required dependencies:

Bash
pip install requests pdfplumber pypdf pandas unstructured datasketch spacy chromadb sentence-transformers
python -m spacy download en_core_web_sm
2. Execution Sequence
Step 1: Run Web Extraction, PII Scrubbing & Deduplication
Bash
python ingest_and_clean.py
Step 2: Remove Boilerplate & Structural Noise
Bash
python remove_boilerplate.py
Step 3: Embed & Index Knowledge Base into ChromaDB
Bash
python index_chroma.py
