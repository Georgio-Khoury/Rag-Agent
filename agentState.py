from typing import TypedDict, List

class AgentState(TypedDict):
    query: str                # The user's input question
    retrieved_chunks: List[str] # Text chunks pulled from ChromaDB
    web_search_results: List[str] # Results from DuckDuckGo if PDF was insufficient
    is_pdf_sufficient: bool    # Evaluated by Gemini ("yes" or "no")
    final_response: str       # The final answer presented to the user

    