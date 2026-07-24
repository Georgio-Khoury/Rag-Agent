# main.py
import os
from dotenv import load_dotenv
from graph import create_agent_graph

def run_tests():
    # Load environment variables (.env)
    load_dotenv()
    
    # Verify API key is present
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY is missing from environment variables or .env file.")
        return

    # Compile the LangGraph agent
    app = create_agent_graph()

    # Define test cases
    test_queries = [
        #"What is the d_k parameter mentioned in the document?",
        #"Explain the self-attention mechanism and compare it to modern breakthroughs in quantum computing."
        #"Give me a list of the basic human rights and also what is causing the global warming?"
        #"What is the main compound that contributes to the thickening of the ozone layer and what's teh link with human rights?"
        "Does climate change have any effect on the water sea levels and on the united kinddom??"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n==========================================")
        print(f"🧪 TEST CASE {i}: {query}")
        print(f"==========================================")
        
        initial_state = {
            "query": query,
            "collection_name": "knowledge_base",
            "sub_queries": [],
            "current_index": -1,
            "current_sub_query": "",
            "target_source": "",
            "retrieved_chunks": [],
            "is_pdf_sufficient": False,
            "sub_query_results": [],
            "final_response": "",
            "is_cache_hit": False,
            "cached_context": ""
        }
        
        try:
            # Run the graph
            final_state = app.invoke(initial_state)
            
            print(f"\n🏁 --- FINAL RESPONSE FOR TEST {i} ---")
            print(final_state.get("final_response", "No response generated."))
            print(f"-----------------------------------------\n")
            
        except Exception as e:
            print(f"❌ Error running test case {i}: {e}")

if __name__ == "__main__":
    run_tests()