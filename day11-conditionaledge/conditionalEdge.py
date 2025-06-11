# Save this as day11-langgraph-conditional-looping.py
import os
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.agents import AgentExecutor, create_react_agent # Not strictly needed for graph building, but useful context


# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Step 1: Define Graph State ---
# The state will primarily contain a list of messages.
# Annotated[List[BaseMessage], add_messages] ensures new messages are appended.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Step 2: Define Custom Tools (reusing from Day 8/9) ---
@tool
def word_reverser(word: str) -> str:
    """Reverses a given word or string."""
    print(f"\n--- Tool Action: Executing word_reverser on '{word}' ---")
    return word[::-1]

@tool
def character_counter(text: str) -> int:
    """Counts the number of characters in a given string."""
    print(f"\n--- Tool Action: Executing character_counter on '{text}' ---")
    return len(text)

tools = [word_reverser, character_counter]
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
            # For Ollama, tool calling is often handled by specific models or custom parsing.
            # Here we'll use a general model and rely on custom parsing in the node.
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

def call_llm(state: AgentState) -> AgentState:
    """
    Node to call the LLM and get its response.
    The LLM will decide whether to call a tool or give a final answer.
    """
    print("--- Node: call_llm ---")
    messages = state['messages']
    # If using Ollama, we need to explicitly inject tool definitions into the prompt
    if LLM_PROVIDER == "ollama":
        tool_names = ", ".join([t.name for t in tools])
        tool_descriptions = "\n".join([f"Tool Name: {t.name}\nTool Description: {t.description}\nTool Schema: {t.args_schema.schema() if t.args_schema else 'No schema'}" for t in tools])
        # A simple prompt hint for Ollama to use tools.
        # More robust tool calling with Ollama might require specific models (e.g., function-calling fine-tunes)
        # or more sophisticated parsing.
        system_message = (
            "You are a helpful assistant. You have access to the following tools: "
            f"{tool_names}.\n\n"
            f"Here are their descriptions and schemas:\n{tool_descriptions}\n\n"
            "If you need to use a tool, respond with a JSON object like: "
            "```json\n{{\"tool_name\": \"<tool_name>\", \"tool_input\": {{...}}}}\n```. "
            "Otherwise, respond with your final answer."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            *messages # Pass all previous messages
        ])
        response = llm.invoke(prompt)
    else: # OpenAI handles tools automatically when bound
        response = llm.invoke(messages)

    # Return the LLM's response appended to the messages
    return {"messages": [response]}


def call_tool(state: AgentState) -> AgentState:
    """
    Node to execute a tool if the LLM has decided to call one.
    It takes the last AI message (which should contain tool calls) and executes them.
    """
    print("--- Node: call_tool ---")
    messages = state['messages']
    last_message = messages[-1]

    tool_outputs = []
    # OpenAI model with tool_calls
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.args
            print(f"Executing tool: {tool_name} with input: {tool_input}")
            # Find the tool by name and execute it
            selected_tool = next(t for t in tools if t.name == tool_name)
            output = selected_tool.invoke(tool_input)
            tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call.id))

    # Basic parsing for Ollama if it tried to output JSON tool call
    elif LLM_PROVIDER == "ollama" and isinstance(last_message.content, str) and "tool_name" in last_message.content:
        import json
        try:
            tool_call_data = json.loads(last_message.content.strip("`").strip("json").strip()) # Attempt to parse JSON
            tool_name = tool_call_data.get("tool_name")
            tool_input = tool_call_data.get("tool_input", {})
            print(f"Executing Ollama-parsed tool: {tool_name} with input: {tool_input}")
            selected_tool = next(t for t in tools if t.name == tool_name)
            output = selected_tool.invoke(tool_input)
            tool_outputs.append(AIMessage(content=f"Tool output: {output}")) # Represent as AI message for simplicity

        except (json.JSONDecodeError, StopIteration) as e:
            print(f"Ollama tool parsing failed or tool not found: {e}")
            tool_outputs.append(AIMessage(content=f"Error parsing tool call: {last_message.content}"))
    else:
        print("No tool calls detected or parsed for execution.")
        # If no tool calls, just return the state as is, or an error message
        # For simplicity, we assume an error or a direct answer was intended by LLM
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
    # Check if the LLM outputted a tool call
    if last_message.tool_calls: # For OpenAI's structured tool calls
        print("Decision: LLM wants to call a tool.")
        return "tool_call"
    # Basic check for Ollama's string output, might need more robust parsing for production
    if LLM_PROVIDER == "ollama" and isinstance(last_message.content, str) and "tool_name" in last_message.content:
        print("Decision: Ollama LLM seems to want to call a tool (based on string content).")
        return "tool_call"
    else:
        print("Decision: LLM has a final answer or no tool needed.")
        return "end"


# --- Step 6: Build the LangGraph with Conditional Edges ---
print("--- Building the LangGraph with Conditional Edges & Loops ---")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tool", call_tool)

# Set entry point
workflow.set_entry_point("call_llm")

# Add conditional edge from call_llm:
# If route_decision returns 'tool_call', go to 'call_tool'.
# If route_decision returns 'end', go to END.
workflow.add_conditional_edges(
    "call_llm", # Source node
    route_decision, # The function that decides the next step
    {
        "tool_call": "call_tool",
        "end": END # Use END to signify graph termination
    }
)

# Add a normal edge from call_tool back to call_llm
# This creates the loop: tool executes, then LLM re-evaluates with tool output
workflow.add_edge("call_tool", "call_llm")

# Compile the graph
app = workflow.compile()
print("LangGraph compiled successfully with conditional edges and looping logic.\n")

# --- Step 7: Invoke the Graph ---
print("--- Invoking the LangGraph (Verbose output below) ---")

# Question requiring a tool
print("\n=== Question 1: Reverse 'LangGraph' ===")
inputs_tool_req = {"messages": [HumanMessage(content="Reverse the word 'LangGraph'.")]}
result_tool_req = app.invoke(inputs_tool_req)
print(f"\nFinal State (Tool Req): {result_tool_req['messages'][-1].content}")


# Question not requiring a tool
print("\n=== Question 2: What is the capital of Japan? ===")
inputs_no_tool = {"messages": [HumanMessage(content="What is the capital of Japan?")]}
result_no_tool = app.invoke(inputs_no_tool)
print(f"\nFinal State (No Tool): {result_no_tool['messages'][-1].content}")

# Question requiring multiple steps (tool + follow-up)
print("\n=== Question 3: Reverse 'Python' and count characters in the reversed word ===")
inputs_multi_step = {"messages": [HumanMessage(content="Reverse the word 'Python' and then count characters in the reversed word.")]}
result_multi_step = app.invoke(inputs_multi_step)
print(f"\nFinal State (Multi-Step): {result_multi_step['messages'][-1].content}")
