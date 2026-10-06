# NHAI Policy RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application designed to answer questions from NHAI policy circulars using document retrieval and a locally running Large Language Model.

The system processes NHAI policy PDF documents, extracts and cleans their text, converts the content into vector embeddings, stores them in a FAISS vector database, retrieves relevant policy sections for a user's question, and generates a grounded response using a local LLM through Ollama.

The application provides both a command-line interface and an interactive Streamlit chatbot.

---

## Project Objective

NHAI projects and implementation activities are governed by policy circulars and related documents.

Finding specific information manually from large policy documents can be time-consuming, especially when documents contain hundreds of pages.

This project demonstrates how Retrieval-Augmented Generation can be used to create a policy knowledge assistant that:

- Searches relevant policy information
- Answers natural-language questions
- Handles scanned PDFs using OCR
- Provides source document and page references
- Supports conversational follow-up questions
- Performs calculations when a policy rule provides the required information
- Refuses to answer when relevant information is not available
- Runs the LLM locally using Ollama

---

## System Architecture

```text
                NHAI Policy PDFs
                       |
                       v
              Document Extraction
              PyMuPDF + OCR Fallback
                       |
                       v
                 Text Cleaning
                       |
                       v
                    Chunking
              800 Characters
              150 Character Overlap
                       |
                       v
                  Embeddings
          all-MiniLM-L6-v2 (384-D)
                       |
                       v
                FAISS Vector DB
                  IndexFlatIP
                       |
                       v
                 User Question
                       |
                       v
          Conversational Query Rewrite
                       |
                       v
              Semantic Retrieval
                  Top-K Chunks
                       |
                       v
               Context Builder
                       |
                       v
              Qwen 2.5 7B / Ollama
                       |
                       v
             Grounded Policy Answer
                       |
              +--------+--------+
              |                 |
              v                 v
           Answer            Sources
                                |
                                v
                         Streamlit UI
```

---

## RAG Workflow

### 1. Document Loading

NHAI policy circulars are loaded from the raw document directory.

For text-based PDFs, text is extracted directly using PyMuPDF.

For scanned documents where normal text extraction fails, the system uses OCR.

### 2. OCR Fallback

Scanned PDF pages are converted into images and processed using:

- PyMuPDF
- Pillow
- Pytesseract
- Tesseract OCR

The extraction pipeline therefore supports both digitally generated and scanned policy documents.

### 3. Text Cleaning

Extracted text is cleaned before further processing to reduce unnecessary formatting and extraction noise.

### 4. Chunking

Large policy documents are divided into smaller overlapping chunks.

Current configuration:

```text
Chunk Size: 800 characters
Chunk Overlap: 150 characters
```

The current knowledge base contains approximately:

```text
1,904 chunks
```

### 5. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Embeddings are normalized before being stored in the vector database.

### 6. Vector Database

The project uses FAISS for semantic vector search.

Index type:

```text
IndexFlatIP
```

The vector database allows the application to retrieve policy sections semantically related to the user's question.

### 7. Query Processing

Before retrieval, conversational questions can be rewritten into standalone questions.

For example:

```text
User:
What is the ambulance deployment rule?

User:
What about for 120 km?
```

The second question can be contextualized using the previous conversation before retrieval.

The system also contains a query-normalization fallback for poorly formatted or mistyped questions.

### 8. Retrieval

The processed question is embedded using the same embedding model.

FAISS then retrieves the highest-ranking policy chunks.

The application currently retrieves up to:

```text
TOP_K = 8
```

chunks for generating context.

### 9. Context Construction

Retrieved chunks are combined into structured context containing metadata such as:

```text
Source
Page
Chunk
Text
```

This context is supplied to the language model.

### 10. Local LLM Generation

The project uses Ollama to run the language model locally.

Current model:

```text
qwen2.5:7b
```

Ollama endpoint:

```text
http://localhost:11434/api/generate
```

The LLM is instructed to answer only from retrieved policy context.

If sufficient information is not available, the system responds:

```text
The information is not available in the provided NHAI policy circulars.
```

---

## Key Features

- Retrieval-Augmented Generation
- NHAI policy-specific knowledge base
- PDF text extraction
- OCR fallback for scanned PDFs
- Text cleaning
- Overlapping document chunking
- Sentence Transformer embeddings
- FAISS semantic vector search
- Similarity-based retrieval filtering
- Local LLM inference with Ollama
- Qwen 2.5 7B integration
- Grounded policy answers
- Source document references
- Page references
- Conversational follow-up questions
- Query correction fallback
- Policy-based calculations
- Hallucination reduction through strict prompting
- Streamlit chatbot interface
- Command-line interface
- Developer retrieval diagnostics

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| PDF Processing | PyMuPDF |
| OCR | Tesseract OCR / Pytesseract |
| Image Processing | Pillow |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Vector Search | Inner Product |
| LLM Runtime | Ollama |
| LLM | Qwen 2.5 7B |
| Web Interface | Streamlit |
| HTTP Communication | Requests |

---

## Project Structure

```text
NHAI_RAG/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   └── .gitkeep
│   │
│   └── vector_store/
│       └── .gitkeep
│
├── main/
│   ├── __init__.py
│   ├── app.py
│   └── streamlit_app.py
│
├── src/
│   ├── chunker.py
│   ├── document_loader.py
│   ├── embedding_generator.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── text_cleaner.py
│   └── vector_store.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Module Description

### `document_loader.py`

Loads PDF documents and extracts text.

Uses normal PDF text extraction when possible and OCR as a fallback for scanned pages.

### `text_cleaner.py`

Cleans extracted policy text before chunking.

### `chunker.py`

Splits large policy documents into overlapping text chunks.

### `embedding_generator.py`

Generates vector embeddings using the Sentence Transformer model.

### `vector_store.py`

Creates and stores the FAISS vector index.

### `retriever.py`

Embeds user queries, searches FAISS and returns relevant policy chunks with metadata and similarity scores.

### `rag_pipeline.py`

Contains the main RAG generation pipeline, including:

- Context construction
- Question normalization
- Conversational question contextualization
- Ollama communication
- Grounded answer generation

### `app.py`

Provides the command-line interface for the RAG assistant.

### `streamlit_app.py`

Provides the interactive web-based chatbot interface.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd NHAI_RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Install Tesseract OCR

Tesseract OCR is required for scanned policy documents.

Install Tesseract separately and make sure the executable is available to the application.

You can verify the installation using:

```bash
tesseract --version
```

---

## Install Ollama

Install Ollama separately on the system.

After installation, download the required model:

```bash
ollama pull qwen2.5:7b
```

Verify that the model is available:

```bash
ollama list
```

---

## Add Policy Documents

Place the policy PDF documents inside:

```text
data/raw/
```

Raw policy PDFs are intentionally excluded from this repository.

The processed data and generated vector database are also excluded because they can be regenerated from the source documents.

---

## Running the Application

### Streamlit Web Application

From the project root:

```bash
python -m streamlit run main/streamlit_app.py
```

Streamlit will normally make the application available locally at:

```text
http://localhost:8501
```

### Command-Line Application

Run:

```bash
python main/app.py
```

---

## Example Questions

The assistant can answer questions such as:

```text
What is the required frequency of ambulance deployment on a Project Highway?
```

```text
What infrastructure is required for ambulance deployment?
```

```text
Who is responsible for ambulance deployment and payment?
```

The system can also use an explicitly retrieved policy rule for calculations.

For example:

```text
If an NHAI highway stretch is 120 km long, how many ambulances should be provided?
```

If the retrieved policy specifies one ambulance for every 50 km or part thereof, the assistant can apply that rule and calculate the required number.

---

## Conversational Questions

The chatbot maintains conversation context.

Example:

```text
User:
What is the required frequency of ambulance deployment?

Assistant:
One ambulance is required for every 50 km or part thereof...

User:
What about for 120 km?

Assistant:
For 120 km, 3 ambulances would be required...
```

The follow-up question is converted into a standalone retrieval query before vector search.

---

## Source Transparency

The application displays the highest-ranked unique policy sources used during retrieval.

Example:

```text
11.107.pdf — Page 2
11.108.pdf — Page 2
11.106.pdf — Page 270
```

The Streamlit interface limits the normal user-facing source list while retaining the complete retrieval information inside the Developer Details section.

---

## Developer Diagnostics

The Streamlit application includes an expandable Developer Details section showing:

- Original user question
- Contextualized standalone question
- Final search query
- Number of retrieved chunks
- Source document
- Page number
- Chunk ID
- Similarity score

This makes it easier to evaluate retrieval quality and debug RAG behavior.

---

## Hallucination Control

The generation prompt instructs the model to:

1. Use only the supplied policy context.
2. Avoid outside or general knowledge.
3. Ignore irrelevant OCR noise.
4. Perform calculations only when the required policy rule exists in the retrieved context.
5. Refuse unsupported questions.
6. Avoid presenting retrieval metadata as part of the generated answer.

These constraints improve grounding but do not guarantee that every generated answer is correct.

---

## Data Privacy and Repository Policy

The following files are excluded from version control:

```text
.venv/
data/raw/*.pdf
data/processed/*
data/vector_store/*
```

This keeps:

- Local environments out of Git
- Source policy documents separate from the code repository
- Generated vector artifacts out of version control

---

## Current Limitations

- The knowledge base is limited to indexed policy documents.
- OCR quality depends on the quality of scanned PDFs.
- Semantic retrieval may occasionally return partially relevant chunks.
- The local LLM requires sufficient system resources.
- The system does not automatically guarantee legal or administrative interpretation of a policy.
- Answers should be verified against original policy documents before official use.

---

## Future Improvements

Possible future improvements include:

- Structured citation generation
- Reranking retrieved chunks
- Hybrid keyword + vector search
- Automated document ingestion
- Policy version tracking
- Improved OCR preprocessing
- Evaluation dataset and RAG accuracy metrics
- Automated retrieval evaluation
- Better source visualization
- Authentication and role-based access
- Deployment on an internal organizational server

---

## Disclaimer

This project is a technical demonstration of Retrieval-Augmented Generation applied to policy documents.

Generated responses should not be treated as an official interpretation of NHAI policy.

For administrative, contractual, legal or operational decisions, users should verify information against the original official policy circular.

---

## Author

**Chaitanya Virkar**

Information Technology | Data Analytics | AI / RAG Development