from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import operator

# --- Step 1: Define the Graph State ---
# This defines the schema of the state that will be passed between nodes.
# 'text' will store the string we're processing.
# 'messages' is a common state key for conversational graphs (though we won't fully use it today).
# Annotated[str, operator.add] means if multiple nodes update 'text', their outputs are concatenated.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    - text: string, the main text being processed.
    - messages: list of messages, for conversational context (optional for this simple demo).
    """
    text: Annotated[str, operator.add] # Use operator.add to concatenate string updates
    messages: Annotated[List[str], add_messages] # For list, use add_messages from langgraph.graph.message

# --- Step 2: Define Nodes (as Python functions) ---
# Each node receives the current state and returns an update to the state.

def start_node(state: GraphState) -> GraphState:
    """
    The initial node that processes the incoming input text.
    It simply ensures the text is part of the state.
    """
    print(f"--- Node: start_node ---")
    current_text = state.get("text", "")
    print(f"Current text in state: '{current_text}'")
    # In a real app, this might do initial processing or validation
    return {"text": current_text + " (processed by start_node)"}

def process_node(state: GraphState) -> GraphState:
    """
    A node that further processes the text by appending a string.
    """
    print(f"--- Node: process_node ---")
    current_text = state.get("text", "")
    print(f"Current text in state: '{current_text}'")
    # In a real app, this could be an LLM call, a tool use, etc.
    return {"text": current_text + " (appended by process_node)"}

# --- Step 3: Build the LangGraph ---
print("--- Building the LangGraph ---")
workflow = StateGraph(GraphState)

# Add nodes to the workflow
workflow.add_node("start_node", start_node)
workflow.add_node("process_node", process_node)

# Set the entry point of the graph
workflow.set_entry_point("start_node")

# Add a simple edge from start_node to process_node
workflow.add_edge("start_node", "process_node")

# Set the finish point of the graph
workflow.set_finish_point("process_node")

# Compile the graph into a runnable
app = workflow.compile()
print("Graph compiled successfully.\n")

# --- Step 4: Invoke the Graph ---
print("--- Invoking the Graph ---")

initial_input = "Hello LangGraph"
print(f"Initial input to graph: '{initial_input}'")

# When invoking, pass the initial state for the 'text' key.
# LangGraph will automatically initialize the state with this input.
final_state = app.invoke({"text": initial_input})

print("\n--- Final Graph State ---")
print(f"Final Text: {final_state['text']}")
print("-" * 50)

# You can also trace the execution (requires graphviz and specific env setup)
# from IPython.display import Image, display
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
#     print("Graph visualization generated (if graphviz is installed and path configured).")
# except Exception as e:
#     print(f"Could not generate graph visualization: {e}")
#     print("Ensure `pip install pygraphviz graphviz` and Graphviz binaries are in PATH.")
