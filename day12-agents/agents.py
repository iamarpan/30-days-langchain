import os
from typing import TypedDict, Annotated, List, Union
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import operator # For Annotated[str, operator.add] if needed for string concatenation

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Step 1: Define Agent State ---
class AgentState(TypedDict):
    """
    Represents the state of our agent's graph.

    - messages: A list of messages forming the conversation history.
                New messages are appended using add_messages.
    """
    messages: Annotated[List[BaseMessage], add_messages]

# --- Step 2: Define Custom Tools ---
@tool
def get_current_datetime() -> str:
    """Returns the current date and time in a human-readable format."""
    print("\n--- Tool Action: Executing get_current_datetime ---")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def simple_calculator(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.
    Example: '2 + 2', '(5 * 3) / 2'
    """
    print(f"\n--- Tool Action: Executing simple_calculator on '{expression}' ---")
    try:
        # Using eval() can be dangerous in production, use a safer math parser for real apps.
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"

tools = [get_current_datetime, simple_calculator]
print(f"Available tools: {[tool.name for tool in tools]}\n")

# --- Step 3: Initialize LLM ---
def initialize_llm(provider, model_name=None, temp=0.7):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        # Bind tools to OpenAI LLM for function calling
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
    """
    Node to call the LLM and get its response.
    The LLM will decide whether to call a tool or give a final answer.
    """
    print("--- Node: call_llm_node ---")
    messages = state['messages']

    if LLM_PROVIDER == "ollama":
        # For Ollama, we need to explicitly inject tool definitions into the prompt
        tool_names = ", ".join([t.name for t in tools])
        tool_descriptions = "\n".join([f"Tool Name: {t.name}\nTool Description: {t.description}\nTool Schema: {t.args_schema.schema() if t.args_schema else 'No schema'}" for t in tools])
        # A more detailed prompt for Ollama to encourage JSON tool calls
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
            *messages # Pass all previous messages
        ])
        response = llm.invoke(prompt)
    else: # OpenAI handles tools automatically when bound
        response = llm.invoke(messages)

    return {"messages": [response]}


def call_tool_node(state: AgentState) -> AgentState:
    """
    Node to execute a tool if the LLM has decided to call one.
    It takes the last AI message (which should contain tool calls) and executes them.
    """
    print("--- Node: call_tool_node ---")
    messages = state['messages']
    last_message = messages[-1]
    tool_outputs = []

    # Handle OpenAI structured tool calls
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.args
            print(f"Executing tool: {tool_name} with input: {tool_input}")
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                try:
                    output = selected_tool.invoke(tool_input)
                    tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call.id))
                except Exception as e:
                    tool_outputs.append(ToolMessage(content=f"Error executing {tool_name}: {e}", tool_call_id=tool_call.id))
            else:
                tool_outputs.append(ToolMessage(content=f"Tool '{tool_name}' not found.", tool_call_id=tool_call.id))
    # Basic parsing for Ollama if it tried to output JSON tool call
    elif LLM_PROVIDER == "ollama" and isinstance(last_message.content, str):
        import json
        try:
            # Attempt to find JSON block in the string
            json_str = last_message.content[last_message.content.find('{'):last_message.content.rfind('}')+1]
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("tool_name")
            tool_input = tool_call_data.get("tool_input", {})
            print(f"Executing Ollama-parsed tool: {tool_name} with input: {tool_input}")
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                output = selected_tool.invoke(tool_input)
                # For Ollama, represent tool output as an AIMessage or a HumanMessage with context
                tool_outputs.append(AIMessage(content=f"Tool output for {tool_name}: {output}"))
            else:
                tool_outputs.append(AIMessage(content=f"Tool '{tool_name}' not found or invalid format: {last_message.content}"))
        except (json.JSONDecodeError, StopIteration, ValueError) as e:
            print(f"Ollama tool parsing failed or no valid tool call JSON found: {e}")
            tool_outputs.append(AIMessage(content=f"LLM did not provide a valid tool call or final answer. Its response was: {last_message.content}"))
    else:
        print("No tool calls detected or parsed for execution.")
        # If no tool calls, it means the LLM likely intended a direct answer already.
        # This node would ideally only be reached if a tool was intended.
        # For robustness, we might add a "no_tool_found" path or error handling.
        pass

    return {"messages": tool_outputs}

# --- Step 5: Define the Routing/Decider Function ---
def route_decision(state: AgentState) -> str:
    """
    Decides the next step based on the last message from the LLM.
    Returns 'tool_call' if a tool needs to be called, otherwise 'end'.
    """
    print("--- Decider: route_decision ---")
    last_message = state['messages'][-1]

    # Check for OpenAI's structured tool calls
    if last_message.tool_calls:
        print("Decision: LLM wants to call a tool (OpenAI structured call).")
        return "tool_call"
    
    # Check for Ollama's string output with potential JSON tool call
    if LLM_PROVIDER == "ollama" and isinstance(last_message.content, str):
        # A simple check for the presence of a tool_name pattern.
        # For production, use more robust JSON parsing.
        if "tool_name" in last_message.content and "tool_input" in last_message.content:
             try:
                 # Attempt to parse as JSON to confirm it's a tool call
                 json_str = last_message.content[last_message.content.find('{'):last_message.content.rfind('}')+1]
                 json.loads(json_str) # Just try to load, don't need the result here
                 print("Decision: Ollama LLM seems to want to call a tool (based on string content and JSON parse).")
                 return "tool_call"
             except json.JSONDecodeError:
                 print("Decision: Ollama LLM content looks like text, not a tool call JSON.")
                 return "end" # It's a final answer or not a tool call
        print("Decision: Ollama LLM content looks like text, not a tool call JSON.")
        return "end"
    
    # Default case: if no tool calls detected for any provider
    print("Decision: LLM has a final answer or no tool needed.")
    return "end"

# --- Step 6: Build the LangGraph ---
print("--- Building the Autonomous Agent with LangGraph ---")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("llm", call_llm_node)
workflow.add_node("tool", call_tool_node)

# Set entry point
workflow.set_entry_point("llm")

# Add conditional edge from 'llm' node
# The 'route_decision' function will determine the next step
workflow.add_conditional_edges(
    "llm", # Source node
    route_decision, # The function that decides the next step
    {
        "tool_call": "tool", # If 'tool_call', go to 'tool' node
        "end": END           # If 'end', terminate the graph
    }
)

# Add a normal edge from 'tool' node back to 'llm' node
# This creates the agentic loop: execute tool, then re-evaluate with LLM
workflow.add_edge("tool", "llm")

# Compile the graph
app = workflow.compile()
print("Autonomous agent graph compiled successfully.\n")

# --- Step 7: Invoke the Agent ---
print("--- Invoking the Autonomous Agent (Verbose output below) ---")

# Example questions for the agent
agent_questions = [
    "What is the current date and time?",
    "Calculate (15 * 3) + 7.",
    "Tell me a fun fact about giraffes.", # Should not use a tool
    "What is the current date and time, then what is 100 divided by 4?" # Multi-step
]

for i, question in enumerate(agent_questions):
    print(f"\n===== Agent Turn {i+1} =====")
    print(f"User Question: {question}")
    # Initial input to the graph
    initial_input = {"messages": [HumanMessage(content=question)]}

    try:
        # Invoke the graph with the initial input
        # The 'verbose' output will show the step-by-step reasoning and tool use
        final_state = app.invoke(initial_input)
        print(f"\nAgent Final Answer: {final_state['messages'][-1].content}")
    except Exception as e:
        print(f"Agent encountered an error: {e}")
    print("\n" + "="*80 + "\n")

# Optional: You can visualize the graph (requires graphviz)
# from IPython.display import Image, display
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
#     print("Graph visualization generated (if graphviz is installed and path configured).")
# except Exception as e:
#     print(f"Could not generate graph visualization: {e}")
#     print("Ensure `pip install pygraphviz graphviz` and Graphviz binaries are in PATH.")
