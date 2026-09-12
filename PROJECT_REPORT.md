# Project Report: Domain-Specific RAG Chatbot for PDF Question Answering

## 1. Objective

Build a chatbot that answers natural-language questions from user-uploaded
PDF documents, retrieving only the most relevant passages and generating
answers grounded strictly in that content, with source document/page
citations - implemented to cover not just the brief's minimum features but
its full optional advanced feature set as well.

## 2. Approach

The system follows the standard RAG pipeline: PDF upload -> per-page text
extraction (`pypdf`, with an OCR fallback for scanned pages) -> overlapping
chunking (custom recursive character splitter, 900 chars / 150 overlap) ->
embedding (`all-MiniLM-L6-v2` via Sentence Transformers, with an offline
hashing fallback) -> FAISS vector index -> top-k cosine-similarity retrieval,
optionally scoped to a document subset -> grounded-answer prompt (with prior
chat turns for follow-up questions) sent to an LLM (Groq, OpenAI, or Gemini,
selectable via `.env`) -> answer + sources rendered in a Streamlit chat UI
(or via the FastAPI backend for other clients), with 👍/👎 feedback logging.

A two-tier, backend-calibrated relevance threshold on retrieval scores
ensures the chatbot refuses ("I could not find this information in the
uploaded documents.") rather than fabricating an answer when nothing
relevant is found - directly addressing minimum feature #7.

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
  `RuntimeWarning` documents when this fallback is active. Mid-project, the
  environment started allowing torch again, which surfaced a real
  calibration bug (see below).
- **Two-tier relevance thresholds** (`rag_pipeline.py`): switching to the
  real embedding model exposed that a single fixed similarity floor, tuned
  loosely for the hashing fallback, let weakly topical-but-wrong content
  through with real semantic embeddings (e.g. "Who is the CEO?" scored 0.23
  against generic handbook boilerplate). Replaced with a `(gate, support)`
  pair per backend, calibrated empirically against ~15 probe questions: the
  top match must clear `gate` for a question to be in-scope at all
  (refusing immediately, without an LLM call, if not); `support` is a lower
  floor for keeping secondary context once the gate passes.
- **Prompt-injection guardrail** (`prompt.py`): the system prompt explicitly
  instructs the LLM to treat all retrieved document content as untrusted
  data, never as instructions, and separately clarifies that retrieval has
  already filtered for relevance so terse/informal questions ("attendance?")
  should be answered from on-topic context rather than refused - a real
  inconsistency this fixed during testing (the LLM would refuse the same
  well-supported question at random, roughly half the time, until that
  clarification was added).
- **Per-user document isolation** (`auth.py`): discovered during testing
  that all Streamlit sessions originally shared one global vector store on
  disk, so documents uploaded in one browser session were visible to (and
  overwritable by) any other. The opt-in login gate now gives each
  authenticated user their own vector store subdirectory; the API backend
  (`api.py`) and its own subdirectory follow the same isolation principle.
- **Question-batch splitting** (`rag_pipeline.split_into_questions`): a
  message containing several questions (pasted as one block, e.g. from a
  test sheet) previously only retrieved context for the combined text,
  starving most of the individual questions. Now detected and answered
  independently, with sources deduplicated and merged across the batch.

## 4. Testing

- 40 automated `pytest` tests across `test_rag_pipeline.py`, `test_auth.py`,
  `test_feedback.py`, `test_api.py`, and `test_ocr.py` - covering file
  validation, text extraction with metadata, indexing, retrieval and
  sourcing, refusal behavior, question-batch splitting, source
  deduplication, login/password hashing and per-user isolation, feedback
  logging, the FastAPI endpoints (including its API-key gate), and the OCR
  fallback's graceful degradation (verified both with Tesseract genuinely
  absent, and via a simulated-available case that exercises the real
  plumbing) - all passing, fully offline.
- A 19-question evaluation sheet (`tests/test_questions.csv`) covers
  correct, incorrect, and prompt-injection questions across both sample
  documents. `tests/evaluate.py` runs it automatically against the live
  pipeline and grades each answer against its expected source/refusal - a
  checked-in run (`tests/evaluation_results.csv`) scores 19/19 (100%),
  averaging 0.78s per response.
- Manually verified end-to-end in a real browser multiple times: upload ->
  process -> retrieve -> cite sources -> refuse on an out-of-scope question;
  separately, the login flow (wrong password rejected, correct login,
  per-user upload isolation confirmed via unchanged file timestamps on the
  shared store), feedback buttons (logged record inspected on disk), and the
  OCR checkbox's degradation warning.

## 5. Known Limitations

- The offline hashing-embedder fallback (used only when `torch` can't load)
  has materially lower semantic accuracy than the real MiniLM model - exact
  keyword/synonym overlap matters more than true semantic similarity. On any
  environment where `torch` loads normally (a typical laptop, Streamlit
  Cloud, Docker), the real model is used automatically.
- OCR requires the Tesseract binary installed separately (not pip-
  installable); without it, the feature degrades to the pre-OCR behavior.
  Actual text-recognition accuracy hasn't been verified end-to-end on this
  machine since Tesseract isn't installed here - only the graceful-
  degradation and plumbing paths were testable.
- The Docker setup (`Dockerfile`, `docker-compose.yml`) was written
  carefully but not build-tested, since Docker isn't installed on this
  machine.
- The login system is intentionally minimal (PBKDF2 + a JSON file, no
  sessions/cookies beyond Streamlit's own state) - adequate for a
  single-instance course project, not a production identity provider.

## 6. Deliverables Checklist

| Deliverable | Status |
|---|---|
| Working Streamlit application | Done - `app.py` |
| Complete source code | Done |
| Sample PDF documents | Done - `documents/Policy.pdf`, `documents/Handbook.pdf` |
| `requirements.txt` | Done (+ `requirements-dev.txt` for tests) |
| README with setup/usage | Done - `README.md` |
| Architecture/workflow diagram | Done - Mermaid diagram in `README.md` |
| Testing sheet (15+ questions) | Done - `tests/test_questions.csv` (19 questions), 100% pass rate |
| GitHub repository | Done - pushed |
| Short project report | Done - this file |
| Demonstration video | Not produced - a recorded walkthrough of the running app is a manual step for the presenter |

## 7. Optional Advanced Features (brief section 11) - all implemented

| Feature | Status |
|---|---|
| Multiple document collections and filters | Done - sidebar "Search scope" multiselect |
| Conversation memory for follow-up questions | Done - history-aware query condensing |
| OCR support for scanned PDFs | Done - opt-in, graceful degradation without Tesseract |
| FastAPI backend for mobile/web clients | Done - `api.py` |
| User login and document access control | Done - `auth.py`, per-user isolation |
| Feedback buttons for useful/incorrect answers | Done - `feedback.py` |
| Docker deployment | Done - not build-tested (see Limitations) |
| Evaluation using a prepared QA dataset | Done - `tests/evaluate.py`, 19/19 (100%) |
