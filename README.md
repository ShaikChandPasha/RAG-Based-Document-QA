# RAG-Based Document Q&A Chatbot

A Streamlit application that extracts text from uploaded PDFs, splits it into chunks, embeds the chunks with Sentence Transformers, stores vectors in FAISS, retrieves relevant passages, and generates a context-grounded answer with FLAN-T5.

## Features
- Upload multiple text-based PDF files.
- Chunking with overlap to preserve context.
- Semantic retrieval using FAISS.
- Answers grounded in retrieved passages.
- Displays retrieved passages so users can inspect the evidence.
- Refuses to invent an answer when the context does not contain it.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding and language models. A machine with several GB of free RAM is recommended. Scanned/image-only PDFs require OCR; this version extracts embedded PDF text.

## Architecture
PDF upload -> text extraction -> recursive chunking -> sentence-transformer embeddings -> FAISS similarity search -> FLAN-T5 answer generation -> answer + retrieved passages.

## Evaluation ideas
Create a test set of questions with expected answers. Measure retrieval Recall@k and answer faithfulness manually or with an evaluation framework. Do not report a performance score until it has been measured on a documented test set.

## Limitations
- Local model quality and speed depend on hardware.
- PDF text extraction may fail on scans or complex layouts.
- Retrieved context can still be incomplete; always inspect source passages.
