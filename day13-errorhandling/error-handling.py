import os
import random
from typing import TypedDict, Annotated, List, Union
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import operator # For Annotated[bool, operator.or_]

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Step 1: Define Agent State with Error Flag ---
class AgentState(TypedDict):
    """
    Represents the state of our agent's graph.

    - messages: A list of messages forming the conversation history.
    - error_occurred: A flag to indicate if an error happened in a node.
                      operator.or_ combines boolean updates (True if any update is True).
    """
    messages: Annotated[List[BaseMessage], add_messages]
    error_occurred: Annotated[bool, operator.or_] # New error flag

# --- Step 2: Define Custom Tools (with a potential for error) ---
@tool
def get_current_datetime() -> str:
    """Returns the current date and time in a human-readable format."""
    print("\n--- Tool Action: Executing get_current_datetime ---")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def risky_calculator(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.
    This tool has a 30% chance of failing to demonstrate error handling.
    Example: '2 + 2', '(5 * 3) / 2'
    """
    print(f"\n--- Tool Action: Executing risky_calculator on '{expression}' ---")
    if random.random() < 0.3: # 30% chance of failure
        raise ValueError("Simulated calculation error!")
    try:
        return str(eval(expression))
    except Exception as e:
        # Catch explicit evaluation errors as well
        raise ValueError(f"Calculation expression error: {e}")

tools = [get_current_datetime, risky_calculator]
print(f"Available tools: {[tool.name for tool in tools]}\n")

# --- Step 3: Initialize LLM ---
def initialize_llm(provider, model_name=None, temp=0.7):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp).bind_tools(tools)
    elif provider == "ollama":
        try:
            llm = ChatOllama(model=model_name or OLLAMA_MODEL_CHAT, temperature=temp)
            llm.invoke("Hello!") # Test connection
            return llm
        except Exception as e:
            print(f"Error connecting to Ollama LLM or model '{model_name or OLLAMA_MODEL_CHAT}' not found: {e}")
            print("Please ensure Ollama is running and the specified model is pulled.")
            exit()
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'openai' or 'ollama'.")

llm = initialize_llm(LLM_PROVIDER)
print(f"Using LLM: {LLM_PROVIDER} ({llm.model_name if hasattr(llm, 'model_name') else OLLAMA_MODEL_CHAT})\n")

# --- Step 4: Define Graph Nodes ---

def call_llm_node(state: AgentState) -> AgentState:
    """Node to call the LLM and get its response."""
    print("--- Node: call_llm_node ---")
    messages = state['messages']
    
    # Reset error flag at the start of LLM call (new thought cycle)
    # This is important if we want to potentially recover or retry in a loop
    return_state = {"error_occurred": False} 

    if LLM_PROVIDER == "ollama":
        tool_names = ", ".join([t.name for t in tools])
        tool_descriptions = "\n".join([f"Tool Name: {t.name}\nTool Description: {t.description}\nTool Schema: {t.args_schema.schema() if t.args_schema else 'No schema'}" for t in tools])
        system_message = (
            "You are a helpful AI assistant. You have access to the following tools: "
            f"{tool_names}.\n\n"
            f"Here are their descriptions and schemas:\n{tool_descriptions}\n\n"
            "If you need to use a tool, respond with a JSON object like: "
            "```json\n{{\"tool_name\": \"<tool_name>\", \"tool_input\": {{...}}}}\n```. "
            "Think step by step. If a tool is useful, call it. Otherwise, provide a direct answer. "
            "Your response should be either a tool call JSON or a direct final answer."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            *messages
        ])
        response = llm.invoke(prompt)
    else:
        response = llm.invoke(messages)
    
    return_state["messages"] = [response]
    return return_state


def call_tool_node(state: AgentState) -> AgentState:
    """Node to execute a tool, with error handling."""
    print("--- Node: call_tool_node ---")
    messages = state['messages']
    last_message = messages[-1]
    tool_outputs = []
    error_flag = False

    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.args
            print(f"Attempting to execute tool: {tool_name} with input: {tool_input}")
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                try:
                    output = selected_tool.invoke(tool_input)
                    tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call.id))
                except Exception as e:
                    print(f"!!! Error executing {tool_name}: {e} !!!")
                    tool_outputs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call.id))
                    error_flag = True # Set error flag on tool failure
            else:
                print(f"!!! Error: Tool '{tool_name}' not found. !!!")
                tool_outputs.append(ToolMessage(content=f"Tool '{tool_name}' not found.", tool_call_id=tool_call.id))
                error_flag = True # Set error flag
    elif LLM_PROVIDER == "ollama" and isinstance(last_message.content, str):
        import json
        try:
            json_str = last_message.content[last_message.content.find('{'):last_message.content.rfind('}')+1]
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("tool_name")
            tool_input = tool_call_data.get("tool_input", {})
            print(f"Attempting to execute Ollama-parsed tool: {tool_name} with input: {tool_input}")
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                output = selected_tool.invoke(tool_input)
                tool_outputs.append(AIMessage(content=f"Tool output for {tool_name}: {output}"))
            else:
                print(f"!!! Error: Ollama-parsed Tool '{tool_name}' not found. !!!")
                tool_outputs.append(AIMessage(content=f"Tool '{tool_name}' not found for Ollama.", tool_call_id=None)) # tool_call_id None for AI message
                error_flag = True
        except (json.JSONDecodeError, StopIteration, ValueError) as e:
            print(f"!!! Error parsing Ollama tool call or executing: {e} !!!")
            tool_outputs.append(AIMessage(content=f"Error parsing or executing Ollama tool: {e}"))
            error_flag = True
    else:
        print("No tool calls detected or parsed for execution.")
        error_flag = True # Treat as error if call_tool was reached but no tool call found

    return {"messages": tool_outputs, "error_occurred": error_flag}

def error_handler_node(state: AgentState) -> AgentState:
    """
    Node to handle errors. It logs the error and provides a fallback message.
    """
    print("\n--- Node: error_handler_node (Error Detected!) ---")
    error_message = "An unexpected error occurred during processing. Please try again or rephrase your request."
    # Optionally, you could log the full state or specific error details here
    final_messages = state['messages'] + [AIMessage(content=error_message)]
    print(f"Error handled. Final message: {error_message}")
    return {"messages": final_messages, "error_occurred": False} # Reset error flag

# --- Step 5: Define the Routing/Decider Function ---
def route_decision(state: AgentState) -> str:
    """
    Decides the next step based on error flag or LLM's last message.
    """
    print("--- Decider: route_decision ---")
    if state.get("error_occurred", False):
        print("Decision: Error detected, routing to error_handler_node.")
        return "handle_error"

    last_message = state['messages'][-1]
    if last_message.tool_calls:
        print("Decision: LLM wants to call a tool.")
        return "tool_call"
    
    if LLM_PROVIDER == "ollama" and isinstance(last_message.content, str):
        import json
        try:
            json_str = last_message.content[last_message.content.find('{'):last_message.content.rfind('}')+1]
            json.loads(json_str)
            print("Decision: Ollama LLM seems to want to call a tool (parsed JSON).")
            return "tool_call"
        except json.JSONDecodeError:
            print("Decision: Ollama LLM content looks like text (no tool call JSON).")
            return "end"

    print("Decision: LLM has a final answer or no tool needed.")
    return "end"

# --- Step 6: Build the LangGraph with Error Handling ---
print("--- Building the LangGraph with Error Handling & Debugging ---")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("llm", call_llm_node)
workflow.add_node("tool", call_tool_node)
workflow.add_node("error_handler", error_handler_node) # New error handler node

# Set entry point
workflow.set_entry_point("llm")

# Add conditional edge from 'llm' node (can route to tool or end)
workflow.add_conditional_edges(
    "llm",
    route_decision,
    {
        "tool_call": "tool",
        "end": END,
        "handle_error": "error_handler" # If an error somehow occurs before tool
    }
)

# Add conditional edge from 'tool' node (after tool execution)
# Crucially, after a tool, we check for errors first, then loop back to LLM.
workflow.add_conditional_edges(
    "tool",
    route_decision,
    {
        "tool_call": "tool", # This path shouldn't typically be taken directly after a tool
        "end": "llm", # If no error and no more tool calls, loop back to LLM to finalize
        "handle_error": "error_handler"
    }
)

# After handling an error, we want to terminate the graph for this query,
# or you could try a retry mechanism (more advanced)
workflow.add_edge("error_handler", END)

# Compile the graph
# return_intermediate_steps=True will give us the full trace in the output
app = workflow.compile(
    # debug=True # LangGraph also has a debug option for more verbose internal logging
)
print("Agent graph with error handling compiled successfully.\n")

# --- Step 7: Invoke the Agent ---
print("--- Invoking the Agent (Verbose output below) ---")

agent_questions = [
    "What is the current date and time?", # Expected to succeed
    "Calculate 10 / 0.", # Expected to fail (div by zero in calculator)
    "Calculate 5 * 5.", # Expected to sometimes fail (simulated error)
    "Tell me a joke.", # Expected to succeed (no tool needed)
]

for i, question in enumerate(agent_questions):
    print(f"\n===== Agent Turn {i+1} =====")
    print(f"User Question: {question}")
    initial_input = {"messages": [HumanMessage(content=question)], "error_occurred": False} # Initialize error flag to False

    try:
        # Use return_intermediate_steps=True to inspect the full trace programmatically
        final_result = app.invoke(initial_input, config={"return_intermediate_steps": True})
        
        # Access the final state and intermediate steps
        final_messages = final_result['messages']
        intermediate_steps = final_result.get('intermediate_steps', [])

        print(f"\nAgent Final Answer: {final_messages[-1].content}")
        # print("\n--- Intermediate Steps (for debugging) ---")
        # for step in intermediate_steps:
        #     print(step) # Print the raw step objects if you want
        # print("------------------------------------------")

    except Exception as e:
        print(f"Global Agent Executor encountered an unexpected error: {e}")
    print("\n" + "="*80 + "\n")

# You can still use the graph visualization:
# from IPython.display import Image, display
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
#     print("Graph visualization generated (if graphviz is installed and path configured).")
# except Exception as e:
#     print(f"Could not generate graph visualization: {e}")
#     print("Ensure `pip install pygraphviz graphviz` and Graphviz binaries are in PATH.")
