from langgraph.graph import StateGraph, START, END

# Initialize the Graph with our State schema
workflow = StateGraph(AgentState)

# 1. Register our Nodes
workflow.add_node("retrieve_from_chroma", retrieve_from_chroma)
workflow.add_node("grade_retrieved_documents", grade_retrieved_documents)
workflow.add_node("fallback_web_search", fallback_web_search)
workflow.add_node("generate_final_answer", generate_final_answer)

# 2. Wire the Connections (Edges)
workflow.add_edge(START, "retrieve_from_chroma")
workflow.add_edge("retrieve_from_chroma", "grade_retrieved_documents")

# 3. Add the Conditional Decision Point
workflow.add_conditional_edges(
    "grade_retrieved_documents",
    route_after_grading,
    {
        "generate_final_answer": "generate_final_answer",
        "fallback_web_search": "fallback_web_search"
    }
)

# 4. Finish paths
workflow.add_edge("fallback_web_search", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

# 5. Compile the graph!
app = workflow.compile()