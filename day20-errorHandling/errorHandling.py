import os
import random
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower() # e.g., 'llama2', 'mistral'

# --- LLM Initialization ---
def initialize_llm(provider: str, model_name: str = None, temp: float = 0.7):
    """Initializes and returns the ChatLargeLanguageModel based on provider."""
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp)
    elif provider == "ollama":
        try:
            llm_instance = ChatOllama(model=model_name or OLLAMA_MODEL_CHAT, temperature=temp)
            llm_instance.invoke("Hello!", config={"stream": False}) # Test connection
            return llm_instance
        except Exception as e:
            print(f"Error connecting to Ollama LLM or model '{model_name or OLLAMA_MODEL_CHAT}' not found: {e}")
            print("Please ensure Ollama is running and the specified model is pulled (e.g., 'ollama pull llama2').")
            exit()
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'openai' or 'ollama'.")

# Initialize the chosen LLM
llm = initialize_llm(LLM_PROVIDER)
print(f"Using LLM: {LLM_PROVIDER} ({llm.model_name if hasattr(llm, 'model_name') else OLLAMA_MODEL_CHAT})\n")


# --- 1. Agent State Definition with Error Fields ---
class AgentState(TypedDict):
    """
    Represents the shared memory for our resilient agent.
    Includes fields to track tool output, error messages, and workflow status.
    """
    messages: Annotated[List[BaseMessage], add_messages] # Conversation history
    tool_output: str # To store the successful output from the simulated tool
    error_message: str # To store details if a tool/process fails
    status: str # Tracks the state of the tool call: "tool_success", "tool_failed", "final_response"


# --- 2. Define Agent Nodes ---

# Simulated External Tool (designed to fail under certain conditions)
def simulated_search_tool(query: str) -> str:
    """
    Simulates an external search API call.
    - Fails ~30% of the time randomly.
    - Fails specifically if the query contains "fail me".
    """
    print(f"\n--- Simulated Tool: Processing query: '{query}' ---")
    
    # Simulate a specific failure condition based on input
    if "fail me" in query.lower():
        raise ConnectionError("Simulated: API connection lost for this specific query.")
        
    # Simulate random transient failure (e.g., network timeout)
    if random.random() < 0.3: # 30% chance of failure
        raise TimeoutError("Simulated: API request timed out after 5 seconds.")

    # Simulate successful response for common queries
    if "python" in query.lower():
        return "Python is a versatile high-level programming language, widely used in web development, data science, AI, and automation."
    elif "capital of france" in query.lower():
        return "The capital of France is Paris, famous for its cultural landmarks like the Eiffel Tower and the Louvre Museum."
    elif "mount everest" in query.lower():
        return "Mount Everest, Earth's highest mountain, is located in the Himalayan range and attracts climbers globally."
    else:
        return f"Simulated search result for '{query}': Found some general information, but specific details are scarce."


# Node 1: Call Tool & Catch Errors
def call_tool_node(state: AgentState) -> AgentState:
    """
    Attempts to call the simulated external tool.
    Handles exceptions and updates the state with success/failure status.
    """
    print("\n--- Node: call_tool_node (Attempting Tool Call) ---")
    messages = state['messages']
    last_user_message = messages[-1].content # Get the latest user's question

    try:
        # Attempt to invoke the simulated tool with the user's question
        tool_result = simulated_search_tool(last_user_message)
        print("  Tool call successful. Preparing for response generation.")
        return {
            "tool_output": tool_result,
            "status": "tool_success",
            "error_message": "", # Clear any previous error message
            "messages": [AIMessage(content=f"Tool result received.")] # Log that tool result was obtained
        }
    except (ConnectionError, TimeoutError, Exception) as e:
        # Catch specific exceptions or a general Exception if unexpected errors occur
        error_msg = f"Tool failed: {type(e).__name__} - {e}"
        print(f"  Tool call FAILED: {error_msg}. Routing to error handler.")
        return {
            "tool_output": "", # Clear tool output on failure
            "status": "tool_failed",
            "error_message": error_msg,
            "messages": [AIMessage(content=f"Error during tool use.")] # Log the error occurrence
        }

# Node 2: Generate Final Response (Success Path)
def generate_response_node(state: AgentState) -> AgentState:
    """
    Generates a final, user-facing response using the successful tool output.
    """
    print("\n--- Node: generate_response_node (Success Path) ---")
    tool_output = state['tool_output']
    user_question = state['messages'][-1].content # Original question is always the last HumanMessage

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and articulate assistant. Based on the following tool's search result for the question: '{question}', provide a concise and clear answer to the user."),
        ("human", f"Question: {user_question}\n\nSearch Result:\n{tool_output}")
    ])
    
    response = llm.invoke(prompt)
    final_answer = response.content.strip()
    print(f"  Final Answer generated: {final_answer[:100]}...")
    return {
        "messages": [AIMessage(content=final_answer)],
        "status": "final_response" # Mark as completed
    }

# Node 3: Error Handler / Fallback Strategy (Failure Path)
def error_handler_node(state: AgentState) -> AgentState:
    """
    Handles errors by generating a user-friendly fallback message.
    Could also trigger alternative strategies (e.g., try another tool, human handoff).
    """
    print("\n--- Node: error_handler_node (Fallback Path) ---")
    error_message = state['error_message']
    user_question = state['messages'][-1].content # Original question

    fallback_response = (
        f"I'm sorry, I couldn't get the information for '{user_question}' right now due to a technical issue. "
        f"Details: '{error_message}'. "
        "Please try rephrasing your question or ask about something else."
    )
    
    print(f"  Fallback response generated: {fallback_response[:150]}...")
    return {
        "messages": [AIMessage(content=fallback_response)],
        "status": "error_handled" # Mark as error handled
    }


# --- 3. Build the LangGraph Workflow with Resilience Logic ---
print("--- Building the Resilient Agent Workflow ---")

workflow = StateGraph(AgentState)

# Add our three nodes to the graph
workflow.add_node("call_tool_node", call_tool_node)
workflow.add_node("generate_response_node", generate_response_node)
workflow.add_node("error_handler_node", error_handler_node)

# Set the entry point of the workflow: always start by trying the tool
workflow.set_entry_point("call_tool_node")

# Define conditional edges from the 'call_tool_node'
# This is where the core resilience routing happens based on the 'status'
workflow.add_conditional_edges(
    "call_tool_node",
    # The router function inspects the state and returns the name of the next node
    lambda state: state['status'], 
    {
        "tool_success": "generate_response_node", # If tool worked, go to generate response
        "tool_failed": "error_handler_node"     # If tool failed, go to the error handler
    }
)

# Define the end points for the successful and error handling paths
workflow.add_edge("generate_response_node", END) # After generating response, end
workflow.add_edge("error_handler_node", END)     # After handling error, end

# Compile the graph into a runnable application
resilient_app = workflow.compile()
print("Resilient Agent workflow compiled successfully.\n")


# --- 4. Demonstrate Resilience in Action ---
print("--- Demonstrating Resilient Workflow Scenarios ---")

# Scenario A: Successful Tool Call
print("\n=== Scenario A: Successful Tool Call ===")
user_input_success = "What is Python used for?"
print(f"USER: {user_input_success}")
final_state_success = resilient_app.invoke(
    {"messages": [HumanMessage(content=user_input_success)]}
)
print(f"\nFINAL STATUS: {final_state_success['status']}")
print(f"AI RESPONSE: {final_state_success['messages'][-1].content}")
print("=" * 60)

# Scenario B: Simulated Tool Failure (specific query to guarantee failure)
print("\n=== Scenario B: Simulated Tool Failure (Specific Query) ===")
user_input_fail_query = "Please fail me now, tell me about failure modes in APIs."
print(f"USER: {user_input_fail_query}")
final_state_fail_query = resilient_app.invoke(
    {"messages": [HumanMessage(content=user_input_fail_query)]}
)
print(f"\nFINAL STATUS: {final_state_fail_query['status']}")
print(f"AI RESPONSE: {final_state_fail_query['messages'][-1].content}")
print("=" * 60)

# Scenario C: Simulated Tool Failure (random chance)
print("\n=== Scenario C: Simulated Tool Failure (Random Chance - run multiple times) ===")
user_input_fail_random = "Tell me about Mount Everest."
print(f"USER: {user_input_fail_random}")
# This might succeed or fail based on the random chance (30%).
# Run the script multiple times to observe both success and failure paths.
final_state_fail_random = resilient_app.invoke(
    {"messages": [HumanMessage(content=user_input_fail_random)]}
)
print(f"\nFINAL STATUS: {final_state_fail_random['status']}")
print(f"AI RESPONSE: {final_state_fail_random['messages'][-1].content}")
print("=" * 60)

print("\n--- Resilience Demonstration Complete ---")
print("Observe how the workflow gracefully handles tool failures by routing to the error_handler_node,")
print("providing a user-friendly message instead of crashing.")
