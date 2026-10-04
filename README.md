# Document-based RAG Application

A simple Streamlit demonstration of Retrieval-Augmented Generation (RAG). Upload one or more PDF, DOCX, or UTF-8 TXT documents, then ask questions grounded strictly in their contents.

## Features

- Extracts text from PDF, DOCX, and TXT uploads
- Splits text into overlap-preserving chunks while retaining filename and page/chunk provenance
- Creates real OpenAI embeddings with `text-embedding-3-small` and persists a local FAISS index in `vectorstore/`
- Retrieves the most semantically relevant chunks and uses OpenAI generation for the final grounded answer
- Uses a strict grounding prompt: missing information returns exactly `I could not find this information in the uploaded documents.`
- Shows each retrieved filename, page/chunk, similarity score, and source text
- Supports multiple documents and handles common upload/API/extraction errors

## Setup

1. Use Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your own API key. The application reads this value from the environment; it never stores a key in source code. You may optionally change the OpenAI model names in `.env`.

## Run

```powershell
streamlit run app.py
```

Upload `data/sample_company_policy.txt` to try the application. Click **Process Documents**, then ask a question.

## Example test questions

1. How many paid annual leave days do full-time employees receive?
2. How far ahead should a leave request be submitted?
3. How many days per week may eligible employees work remotely?
4. What is the daily meal reimbursement limit during approved business travel?
5. What security measure is required on company accounts?
6. What is Acme's policy on spacecraft maintenance? *(not in the documents; the required fallback answer should be returned)*

## Tests

Run the offline unit tests (they do not call OpenAI or require an API key):

```powershell
pytest -q
```

For normal application use, ingestion calls OpenAI embeddings and answering calls the OpenAI Responses API. The FAISS index is rebuilt on every document-ingestion run, so vectors from different embedding models are never mixed.
