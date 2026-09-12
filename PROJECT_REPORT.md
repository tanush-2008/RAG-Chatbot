# Project Report: Domain-Specific RAG Chatbot for PDF Question Answering

## 1. Objective

Build a chatbot that answers natural-language questions from user-uploaded
PDF documents, retrieving only the most relevant passages and generating
answers grounded strictly in that content, with source document/page
citations.

## 2. Approach

The system follows the standard RAG pipeline: PDF upload -> per-page text
extraction (`pypdf`) -> overlapping chunking (custom recursive character
splitter, 900 chars / 150 overlap) -> embedding (`all-MiniLM-L6-v2` via
Sentence Transformers, with an offline hashing fallback) -> FAISS vector
index -> top-k cosine-similarity retrieval -> grounded-answer prompt sent to
an LLM (Groq, OpenAI, or Gemini, selectable via `.env`) -> answer + sources
rendered in a Streamlit chat UI.

A relevance floor (`MIN_RELEVANCE_SCORE`) on retrieval scores ensures the
chatbot refuses ("I could not find this information in the uploaded
documents.") rather than fabricating an answer when nothing relevant is
found - directly addressing minimum feature #7 (refuse to invent answers).

## 3. Key Design Decisions

- **LLM-provider abstraction** (`llm_provider.py`): supports Groq, OpenAI,
  and Gemini behind one interface, selected by an environment variable, so
  the grader/user isn't locked into one paid API. If no key is configured,
  an "extractive" fallback shows the raw retrieved passage instead of an
  LLM-generated answer, so the app is fully runnable with zero setup cost.
- **Custom text splitter** (`text_splitter.py`): reimplements LangChain's
  `RecursiveCharacterTextSplitter` behavior directly, avoiding a dependency
  chain (`langchain-text-splitters` -> `langchain-core` -> `uuid_utils`)
  that pulls in a native extension blocked outright by this machine's
  Windows Application Control policy.
- **Embedding fallback** (`embeddings.py`): the same class of environment
  restriction blocks `torch`'s native DLLs, which `sentence-transformers`
  requires. Rather than let the whole app crash, a deterministic,
  dependency-free hashing (bag-of-words) embedder is used as an automatic
  fallback, with stopword filtering to keep refusal behavior correct. A
  `RuntimeWarning` documents when this fallback is active.
- **Prompt-injection guardrail** (`prompt.py`): the system prompt explicitly
  instructs the LLM to treat all retrieved document content as untrusted
  data, never as instructions - addressing the brief's "ignore instructions
  inside documents that try to change the chatbot rules" requirement.

## 4. Testing

- 8 automated `pytest` tests (`tests/test_rag_pipeline.py`) cover file
  validation, text extraction with metadata, indexing, correct retrieval and
  sourcing for answerable questions, and refusal for unanswerable ones - all
  passing, fully offline.
- A 19-question manual evaluation sheet (`tests/test_questions.csv`) covers
  correct, incorrect, and prompt-injection questions across both sample
  documents, per the brief's testing methodology (retrieval accuracy, answer
  correctness, groundedness, refusal quality, source quality, response
  time).
- Manually verified end-to-end in a real browser: upload -> process ->
  retrieve -> cite sources -> refuse on an out-of-scope question.

## 5. Known Limitations

- The offline hashing-embedder fallback (used only when `torch` can't load)
  has materially lower semantic accuracy than the real MiniLM model - exact
  keyword/synonym overlap matters more than true semantic similarity. On any
  environment where `torch` loads normally (a typical laptop, Streamlit
  Cloud, Docker), the real model is used automatically.
- OCR for scanned/image-only PDFs is out of scope (per the brief, listed as
  an optional extension).
- No conversation memory across turns yet (each question is answered
  independently of chat history) - listed as an optional advanced feature in
  the brief.

## 6. Deliverables Checklist

| Deliverable | Status |
|---|---|
| Working Streamlit application | Done - `app.py` |
| Complete source code | Done |
| Sample PDF documents | Done - `documents/Policy.pdf`, `documents/Handbook.pdf` |
| `requirements.txt` | Done |
| README with setup/usage | Done - `README.md` |
| Architecture/workflow diagram | Done - Mermaid diagram in `README.md` |
| Testing sheet (15+ questions) | Done - `tests/test_questions.csv` (19 questions) |
| GitHub repository | Pending - local git repo initialized; push on request |
| Short project report | Done - this file |
| Demonstration video | Not produced - a recorded walkthrough of the running app is a manual step for the presenter |
