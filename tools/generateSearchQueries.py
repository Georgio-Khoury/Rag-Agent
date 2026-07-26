import os
from google import genai
from typing import List

def generate_multiple_search_queries(user_query: str) -> List[str]:
    """Asks Gemini to break a multi-part user question down into a clean list of individual search queries."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    prompt = f"""You are an advanced search coordinator. Analyze the user's input. If it asks for multiple distinct things, break it down into a list of separate, punchy search engine queries (maximum 3).
    Format your output strictly as a comma-separated list. Do not include bullet points, numbers, or quotes.
    
    User Query: {user_query}
    Search Queries:"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Split by comma and strip whitespace to get a clean Python list
    queries = [q.strip() for q in response.text.split(",") if q.strip()]
    return queries