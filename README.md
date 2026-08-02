# Rag-Agent

### Node Explanations & State Effects

#### 1. `Decompose User Query` (`decompose_user_query`)
* **What it does:** Takes the complex user prompt from the initial state, passes it to the **Gemini 2.5 Flash** model, and breaks it down into a list smaller sub-queries.
* **State Effect:** Populates the `sub_queries` list and sets up variables in the `AgentState`.

#### 2. `Iterate Through Sub-queries` (`advance_subquery`)
* **What it does:** Iterates through the sub querry list until all sub querries are processed.
* **State Effect:** Updates the `current_sub_query` state variable so all nodes know which question is currently being processed.

#### 3. `Check Cache` (`check_cache`)
* **What it does:** Performs a semantic similarity search against your cache store to see if the user's exact sub-query (or a semantically equivalent one) has already been answered before.
* **State Effect:** Appends a cache similarity score or match status flag to the state to let the conditional router decide between a cache hit or miss.

#### 4. `Extract from Cache and Append` (`append_cache_result`)
* **What it does:** **(Cache Hit Path)** Bypasses vector database retrieval entirely. It grabs the answer directly from the cache.
* **State Effect:** Appends the cached response dictionary containing the sub-query, source tag, and cached context to the `sub_query_results` list in the state, then loops back to the advance node.

#### 5. `Route to Document Source` (`route_subquery_source`)
* **What it does:** **(Cache Miss Path)** Uses an LLM router to inspect the sub-query and determine which specific PDF document or source file contains the relevant information.
* **State Effect:** Updates the `target_source` variable in the state to specify the file name for targeted metadata filtering.

#### 6. `Retrieve from Vector DB` (`retrieve_from_chroma`)
* **What it does:** Embeds the sub-query using an embedding model and queries your ChromaDB master collection using the `target_source` as a metadata pre-filter (or falling back to a global search if no source is matched).
* **State Effect:** Populates the `retrieved_chunks` list in the state with the top matching text snippets from the database.

#### 7. `Assess Retrieved Chunks` (`grade_retrieved_documents`)
* **What it does:** Grades the quality and relevance of the retrieved text chunks to determine if they contain sufficient information to answer the sub-query.
* **State Effect:** Sets a grading evaluation flag in the state, routing the workflow either toward saving results (if sufficient) or triggering a web search fallback (if insufficient).

#### 8. `Web Search` (`fallback_web_search`)
* **What it does:** **(Fallback Path)** Executes an external search when local vector retrieval chunks are graded as insufficient or irrelevant.
* **State Effect:** Overwrites or appends the newly fetched web search text snippets into the `retrieved_chunks` state variable so they can be processed like local chunks.

#### 9. `Save Results` (`save_pdf_result`)
* **What it does:** Packages the successfully retrieved and graded text context along with its source metadata.
* **State Effect:** Appends a structured dictionary of the finding (`sub_query`, `source`, and combined `context`) to the `sub_query_results` state array.

#### 10. `Save to Cache` (`cache_store`)
* **What it does:** Takes the newly verified findings from the RAG pipeline or web search and commits them into your semantic cache database.
* **State Effect:** Persists the new data externally for future runs, then loops control back to the `advance` node to process the next sub-query.

#### 11. `Generate Response` (`generate_final_answer`)
* **What it does:** Triggered once all sub-queries have been fully iterated through (`finish` state). It aggregates all collected items from `sub_query_results`.
* **State Effect:** Passes the accumulated context to Gemini 2.5 Flash to synthesize the final, comprehensive response that answers the user's original prompt, marking the end of execution.

  
![RAG Pipeline Architecture](assets/State_graph.svg)