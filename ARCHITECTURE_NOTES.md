# Final Baseline Architecture

## Core principle
- **Only the Memory State Generator is assumed to be a trainable module.**
- Router / Search Prep / Retriever / Answer LLM remain non-trainable baselines.

## Parallel execution policy
When a PDF arrives with a chat turn:
1. `attach_pdf_to_session()` saves the file and marks the session as `RUNNING`.
2. `ensure_pdf_ingest_started()` launches the heavy PDF ingest task in background.
3. In parallel, the pipeline runs:
   - user message append
   - multiturn resolve
   - intent extraction
   - topic tracking
   - router decision
4. Only if the router chooses a document-dependent path (`RETRIEVE_DOC` or `SEARCH_PREP_THEN_RETRIEVE`), the pipeline calls `wait_for_pdf_ready()` and continues after ingest is ready.

## Layers
1. **Context & Document Preparation Layer**
   - PDF parser / OCR fallback placeholder
   - page split / chunk build
   - recent turns / previous memory / attachment metadata
2. **Document Indexing & Cache Layer**
   - parser cache
   - chunk cache
   - summary cache
   - embedding cache
   - FAISS or numpy fallback vector index
3. **Memory State Generator**
   - current_topic
   - active_document
   - resolved_refs
   - open_questions
   - memory_summary
4. **Router**
   - `DIRECT_ANSWER`
   - `RETRIEVE_DOC`
   - `SEARCH_PREP_THEN_RETRIEVE`
   - `ASK_CLARIFICATION`
5. **Search Prep**
   - lightweight query normalization / reference resolution
6. **Retriever**
   - vector search / hybrid TODO
7. **Answer LLM**
   - prompt = query + memory + recent turns + retrieved context
   - request-level KV cache is noted as runtime optimization TODO

## Important TODOs
- Replace heuristic router with LLM judge or classifier trained from auto-labeled logs.
- Replace heuristic memory summary updater with the actual trained memory model.
- Implement OCR fallback and finer-grained page-level parallel workers.
- Add persistent cache and persistent FAISS index.
