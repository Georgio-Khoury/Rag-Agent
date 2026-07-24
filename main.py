# main.py
from dotenv import load_dotenv
from graph import create_agent_graph

if __name__ == "__main__":
    # Ensure environment variables are loaded
    load_dotenv()
    
    # 1. Construct and compile the agent application once at boot
    agent_app = create_agent_graph()
    print("🚀 Dynamic Agent System Ready for Production Testing!\n" + "="*60)
    
    # ==================================================================
    # TEST CASE 1: Pure PDF Query (Should return TRUE & skip Web Search)
    # ==================================================================
    # query_1 = {
    #     "query": "What are the components of the Multi-Head Attention mechanism?",
    #     "collection_name": "attention_is_all_you_need"
    # }
    # print(f"\n▶️ RUNNING TEST 1: Pure PDF Information Retrieval...")
    # output_1 = agent_app.invoke(query_1)
    # print("\n🏁 FINAL RESPONSE 1:")
    # print(output_1["final_response"])
    # print("="*60)
    
    # ==================================================================
    # TEST CASE 2: Multi-Topic PDF Query (Should return TRUE & skip Web Search)
    # This asks for two completely distinct concepts that both live in the PDF.
    # ==================================================================
    #query_2 = {
    #     "query": "Explain what the Scaled Dot-Product Attention formula is AND tell me what optimizer was used to train the model.",
    #     "collection_name": "attention_is_all_you_need"
    # }
    # print(f"\n▶️ RUNNING TEST 2: Multi-Topic Query (All internal to PDF)...")
    # output_2 = agent_app.invoke(query_2)
    # print("\n🏁 FINAL RESPONSE 2:")
    # print(output_2["final_response"])
    # print("="*60)

    query_test = {
        "query": "Explain to me what happened in the Lebanese Civil war, and also explain to me about the components of multi threaded attention mechanism and which optimizer did we use to train the model?",
        "collection_name":"attention_is_all_you_need"
    }
    output_test = agent_app.invoke(query_test)
    print(output_test["final_response"])
    
    # ==================================================================
    # TEST CASE 3: Mixed-Context Query (Should return FALSE & trigger multi-query Web Search)
    # One topic is inside the PDF, the other must be sourced live from the web.
    # ==================================================================
    # query_3 = {
    #     "query": "What is the specific dimensionality parameter d_k used in the paper, and who is the current CEO of Google?",
    #     "collection_name": "attention_is_all_you_need"
    # }
    # print(f"\n▶️ RUNNING TEST 3: Mixed-Context Compound Query (PDF + Web)...")
    # output_3 = agent_app.invoke(query_3)
    # print("\n🏁 FINAL RESPONSE 3:")
    # print(output_3["final_response"])
    # print("="*60)