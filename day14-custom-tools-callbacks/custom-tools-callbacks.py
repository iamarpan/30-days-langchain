import os
import requests # For making HTTP requests
from typing import TypedDict, Annotated, List, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field # For structured tool inputs
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.callbacks import BaseCallbackHandler # For custom callbacks
from langchain_core.outputs import LLMResult # For on_llm_end type hinting

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Step 1: Define Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Step 2: Define Advanced Custom Tool (Simulated External API Interaction) ---

# Define the input schema for the tool using Pydantic
class GetWeatherInput(BaseModel):
    location: str = Field(description="The city name for which to get the weather.")
    unit: str = Field(description="The unit of temperature, either 'celsius' or 'fahrenheit'.")

@tool("get_current_weather", args_schema=GetWeatherInput)
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Fetches the current weather information for a given location and unit.
    This is a simulated external API call.
    """
    print(f"\n--- Tool Action: Executing get_current_weather for {location} in {unit} ---")
    
    # Simulate an external API call
    # In a real scenario, you'd use requests.get() to an actual weather API
    # Example: response = requests.get(f"https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q={location}")
    # data = response.json()
    
    dummy_weather_data = {
        "London": {"celsius": "15°C, cloudy", "fahrenheit": "59°F, cloudy"},
        "Hyderabad": {"celsius": "30°C, sunny", "fahrenheit": "86°F, sunny"},
        "New York": {"celsius": "20°C, partly cloudy", "fahrenheit": "68°F, partly cloudy"},
        "Tokyo": {"celsius": "25°C, rainy", "fahrenheit": "77°F, rainy"}
    }

    try:
        if location in dummy_weather_data:
            weather = dummy_weather_data[location].get(unit.lower())
            if weather:
                return f"The weather in {location} is: {weather}."
            else:
                return f"Weather unit '{unit}' not supported for {location}. Please use 'celsius' or 'fahrenheit'."
        else:
            return f"Weather data not available for {location}. Please try a major city."
    except Exception as e:
        return f"An error occurred while fetching weather for {location}: {e}"

# Simple datetime tool from previous days
@tool
def get_current_datetime() -> str:
    """Returns the current date and time in a human-readable format."""
    print("\n--- Tool Action: Executing get_current_datetime ---")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_current_weather, get_current_datetime]
print(f"Available tools: {[tool.name for tool in tools]}\n")

# --- Step 3: Implement a Custom Callback Handler ---
class MyAgentLoggerCallback(BaseCallbackHandler):
    """
    A custom callback handler to log agent execution details.
    """
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print(f"\n--- Callback: LLM Start ---")
        # print(f"  Model: {serialized.get('lc_kwargs', {}).get('model_name', 'Unknown')}")
        print(f"  Prompts: {prompts[0][:100]}...") # Print first 100 chars of first prompt
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"--- Callback: LLM End ---")
        if response.generations and response.generations[0].text:
            print(f"  LLM Output: {response.generations[0].text[:100]}...") # Print first 100 chars of output
        else:
            print(f"  LLM Output: No text generation found.")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        print(f"\n--- Callback: Tool Start ---")
        print(f"  Tool Name: {serialized.get('name', 'Unknown')}")
        print(f"  Tool Input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        print(f"--- Callback: Tool End ---")
        print(f"  Tool Output: {output[:100]}...") # Print first 100 chars of output

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        # This callback is often redundant if on_llm_end or on_tool_start capture enough
        # but can be useful to see agent specific actions.
        # print(f"--- Callback: Agent Action ---")
        # print(f"  Action: {action}")
        pass

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
        print(f"\n--- Callback: Agent Finish ---")
        print(f"  Final Answer: {finish.return_values['output'][:100]}...")


# --- Step 4: Initialize LLM ---
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


# --- Step 5: Define Graph Nodes (similar to Day 12) ---
def call_llm_node(state: AgentState) -> AgentState:
    print("--- Node: call_llm_node (Agent Thinking) ---")
    messages = state['messages']

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
    return {"messages": [response]}


def call_tool_node(state: AgentState) -> AgentState:
    print("--- Node: call_tool_node (Executing Tool) ---")
    messages = state['messages']
    last_message = messages[-1]
    tool_outputs = []

    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.args
            print(f"  Attempting to execute tool: {tool_name} with input: {tool_input}")
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                try:
                    output = selected_tool.invoke(tool_input)
                    tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call.id))
                except Exception as e:
                    print(f"!!! Error executing {tool_name}: {e} !!!")
                    tool_outputs.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call.id))
            else:
                print(f"!!! Error: Tool '{tool_name}' not found. !!!")
                tool_outputs.append(ToolMessage(content=f"Tool '{tool_name}' not found.", tool_call_id=tool_call.id))
    elif LLM_PROVIDER == "ollama" and isinstance(last_message.content, str):
        import json
        try:
            json_str = last_message.content[last_message.content.find('{'):last_message.content.rfind('}')+1]
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("tool_name")
            tool_input = tool_call_data.get("tool_input", {})
            print(f"  Attempting to execute Ollama-parsed tool: {tool_name} with input: {tool_input}")
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool:
                output = selected_tool.invoke(tool_input)
                tool_outputs.append(AIMessage(content=f"Tool output for {tool_name}: {output}"))
            else:
                print(f"!!! Error: Ollama-parsed Tool '{tool_name}' not found. !!!")
                tool_outputs.append(AIMessage(content=f"Tool '{tool_name}' not found for Ollama.", tool_call_id=None))
        except (json.JSONDecodeError, StopIteration, ValueError) as e:
            print(f"!!! Error parsing Ollama tool call or executing: {e} !!!")
            tool_outputs.append(AIMessage(content=f"Error parsing or executing Ollama tool: {e}"))
    return {"messages": tool_outputs}

# Router Function
def route_decision(state: AgentState) -> str:
    print("--- Decider: route_decision ---")
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        print("  Decision: LLM wants to call a tool.")
        return "tool_call"
    if LLM_PROVIDER == "ollama" and isinstance(last_message.content, str):
        import json
        try:
            json_str = last_message.content[last_message.content.find('{'):last_message.content.rfind('}')+1]
            json.loads(json_str)
            print("  Decision: Ollama LLM seems to want to call a tool (parsed JSON).")
            return "tool_call"
        except json.JSONDecodeError:
            print("  Decision: Ollama LLM content looks like text (no tool call JSON).")
            pass # Fall through to 'end'
    print("  Decision: LLM has a final answer or no tool needed.")
    return "end"

# --- Step 6: Build the LangGraph with Callbacks ---
print("--- Building the LangGraph Agent ---")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("llm", call_llm_node)
workflow.add_node("tool", call_tool_node)

# Set entry point
workflow.set_entry_point("llm")

# Add conditional edge from 'llm' node
workflow.add_conditional_edges(
    "llm",
    route_decision,
    {
        "tool_call": "tool",
        "end": END
    }
)

# Add looping edge from 'tool' node back to 'llm' node
workflow.add_edge("tool", "llm")

# Compile the graph
langgraph_app = workflow.compile()
print("LangGraph agent compiled successfully.\n")

# --- Step 7: Invoke the Agent with Custom Callbacks ---
print("--- Invoking the Agent with MyAgentLoggerCallback ---")

agent_questions = [
    "What is the current date and time?",
    "What is the weather in Hyderabad in celsius?",
    "What is the weather in London in fahrenheit?",
    "Tell me about the capital of France.", # No tool needed
    "What is the weather in Mars?" # Tool called, but data not available
]

# Instantiate our custom callback handler
my_logger = MyAgentLoggerCallback()

for i, question in enumerate(agent_questions):
    print(f"\n{'='*20} Agent Turn {i+1} {'='*20}")
    print(f"User Question: {question}")
    
    initial_input = {"messages": [HumanMessage(content=question)]}

    try:
        # Invoke the graph with the custom callback handler
        final_state = langgraph_app.invoke(
            initial_input,
            config={"callbacks": [my_logger], "recursion_limit": 50}, # Pass the callback instance
            # verbose=True # You can add verbose=True for even more internal LangGraph logs
        )
        print(f"\nFinal Agent Response: {final_state['messages'][-1].content}")
    except Exception as e:
        print(f"!!! Agent encountered an unexpected error during invocation: {e} !!!")
    print(f"{'='*50}\n")
