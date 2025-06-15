import os
import requests
from typing import TypedDict, Annotated, List, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json # For parsing LLM's intent classification output

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- 1. Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # Add a field to store the identified intent for debugging/clarity
    intent: str
    user_input_for_task: str # To pass original user input to relevant branches

# --- 2. Tools (Reusing from Day 14) ---
class GetWeatherInput(BaseModel):
    location: str = Field(description="The city name for which to get the weather.")
    unit: str = Field(description="The unit of temperature, either 'celsius' or 'fahrenheit'.")

@tool("get_current_weather", args_schema=GetWeatherInput)
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Fetches the current weather information for a given location and unit.
    This is a simulated external API call.
    """
    print(f"\n--- Tool: get_current_weather for {location} in {unit} ---")
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

@tool
def simple_calculator(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.
    Example: '2 + 2', '(5 * 3) / 2'
    """
    print(f"\n--- Tool: simple_calculator for expression: {expression} ---")
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"

tools = [get_current_weather, simple_calculator]

# --- 3. LLM Initialization ---
def initialize_llm(provider, model_name=None, temp=0.7, bind_tools=False):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        llm_instance = ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp)
        if bind_tools:
            return llm_instance.bind_tools(tools)
        return llm_instance
    elif provider == "ollama":
        try:
            llm_instance = ChatOllama(model=model_name or OLLAMA_MODEL_CHAT, temperature=temp)
            llm_instance.invoke("Hello!") # Test connection
            return llm_instance
        except Exception as e:
            print(f"Error connecting to Ollama LLM or model '{model_name or OLLAMA_MODEL_CHAT}' not found: {e}")
            print("Please ensure Ollama is running and the specified model is pulled.")
            exit()
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'openai' or 'ollama'.")

# Two LLMs: one for intent classification (no tools), one for task execution (with tools)
llm_classifier = initialize_llm(LLM_PROVIDER, bind_tools=False)
llm_task_executor = initialize_llm(LLM_PROVIDER, bind_tools=True)

# --- 4. Define Nodes for Different Intents ---

# Node 1: Intent Classification
def classify_intent(state: AgentState) -> AgentState:
    print("--- Node: Classifying User Intent ---")
    user_message = state['messages'][-1].content

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an intelligent intent classifier. Analyze the user's request and classify its intent.
        Return your classification as a JSON object with a single key 'intent' and one of the following values:
        - 'summarize': If the user wants a summary of provided text.
        - 'answer_question': If the user is asking a general knowledge question that doesn't require specific tools.
        - 'perform_task': If the user's request requires using tools (e.g., asking for weather, calculations).

        Example for 'summarize': {"intent": "summarize"}
        Example for 'answer_question': {"intent": "answer_question"}
        Example for 'perform_task': {"intent": "perform_task"}
        """),
        ("human", user_message)
    ])
    
    response = llm_classifier.invoke(prompt)
    intent_raw = response.content.strip()
    
    try:
        # Attempt to parse JSON, sometimes LLMs add markdown fences
        if intent_raw.startswith("```json"):
            intent_raw = intent_raw[7:-3].strip()
        
        intent_json = json.loads(intent_raw)
        intent = intent_json.get("intent", "answer_question") # Default to answer if parsing fails
        if intent not in ["summarize", "answer_question", "perform_task"]:
            intent = "answer_question" # Fallback for invalid classification
        print(f"  Identified Intent: {intent}")
        return {"intent": intent, "user_input_for_task": user_message}
    except json.JSONDecodeError:
        print(f"  Warning: LLM returned non-JSON intent: {intent_raw[:50]}... Defaulting to 'answer_question'")
        return {"intent": "answer_question", "user_input_for_task": user_message}


# Node 2: Handle Summarization
def summarize_node(state: AgentState) -> AgentState:
    print("--- Node: Executing Summarization ---")
    user_input = state['user_input_for_task']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise summarizer. Summarize the following text clearly and briefly."),
        ("human", user_input)
    ])
    response = llm_classifier.invoke(prompt) # Using llm_classifier for simple summaries
    return {"messages": [AIMessage(content=response.content)]}

# Node 3: Handle General Questions
def answer_question_node(state: AgentState) -> AgentState:
    print("--- Node: Answering General Question ---")
    user_input = state['user_input_for_task']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the following question concisely."),
        ("human", user_input)
    ])
    response = llm_classifier.invoke(prompt) # Using llm_classifier for general questions
    return {"messages": [AIMessage(content=response.content)]}

# Node 4: Handle Task Execution (This is our existing tool-calling logic)
def call_llm_for_task(state: AgentState) -> AgentState:
    print("--- Node: LLM Thinking (Task Execution) ---")
    # Use the original user input for task processing
    current_messages = state['messages'] + [HumanMessage(content=state['user_input_for_task'])] if not state['messages'] else state['messages']
    
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
            *current_messages
        ])
        response = llm_task_executor.invoke(prompt)
    else:
        response = llm_task_executor.invoke(current_messages)
    return {"messages": [response]}


def call_tool_for_task(state: AgentState) -> AgentState:
    print("--- Node: Executing Tool for Task ---")
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


# --- 5. Routing Function (The Heart of Advanced Logic) ---
def route_agent_workflow(state: AgentState) -> str:
    print(f"--- Router: route_agent_workflow (Current Intent: {state.get('intent')}) ---")
    intent = state.get("intent")
    
    if intent == "summarize":
        return "summarize_node"
    elif intent == "answer_question":
        return "answer_question_node"
    elif intent == "perform_task":
        # The 'perform_task' branch itself needs further routing for tool calls
        last_message_from_task_llm = state['messages'][-1]
        if last_message_from_task_llm.tool_calls:
            print("  Sub-routing for 'perform_task': LLM wants to call a tool.")
            return "tool_node_for_task" # Route to tool execution within the task flow
        if LLM_PROVIDER == "ollama" and isinstance(last_message_from_task_llm.content, str):
            try:
                json_str = last_message_from_task_llm.content[last_message_from_task_llm.content.find('{'):last_message_from_task_llm.content.rfind('}')+1]
                json.loads(json_str)
                print("  Sub-routing for 'perform_task': Ollama LLM seems to want to call a tool.")
                return "tool_node_for_task"
            except json.JSONDecodeError:
                pass
        print("  Sub-routing for 'perform_task': LLM has final answer for task.")
        return END # Task complete
    else:
        print("  Error: Unknown intent. Defaulting to 'answer_question'.")
        return "answer_question_node" # Fallback

# --- 6. Build the LangGraph ---
print("--- Building the Multi-Intent LangGraph Agent ---")
workflow = StateGraph(AgentState)

# Add the initial intent classification node
workflow.add_node("classify_intent", classify_intent)

# Add nodes for each main intent branch
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("answer_question_node", answer_question_node)

# Add nodes for the 'perform_task' sub-workflow
workflow.add_node("llm_for_task", call_llm_for_task)
workflow.add_node("tool_node_for_task", call_tool_for_task)


# Set the entry point to the intent classifier
workflow.set_entry_point("classify_intent")

# Define the primary conditional routing from the classifier
workflow.add_conditional_edges(
    "classify_intent",
    route_agent_workflow, # Our main router
    {
        "summarize_node": "summarize_node",
        "answer_question_node": "answer_question_node",
        "perform_task": "llm_for_task" # Start the task execution flow
    }
)

# Define edges within the 'perform_task' branch (the tool-using agent logic)
workflow.add_conditional_edges(
    "llm_for_task",
    route_agent_workflow, # Re-use the router to check if tool call or END for task
    {
        "tool_node_for_task": "tool_node_for_task",
        END: END # Task flow ends here
    }
)
workflow.add_edge("tool_node_for_task", "llm_for_task") # Loop back for multi-tool use

# Define end points for the other branches
workflow.add_edge("summarize_node", END)
workflow.add_edge("answer_question_node", END)


# Compile the graph
langgraph_app = workflow.compile()
print("Multi-Intent LangGraph agent compiled successfully.\n")

# --- 7. Invoke the Agent with Different Intents ---
print("--- Invoking the Agent with Various User Intents ---")

agent_questions = [
    "Summarize this: Large language models (LLMs) are a type of artificial intelligence (AI) program that can recognize and generate text and other content based on massive datasets. They are trained on vast amounts of text data to learn patterns, grammar, facts, and reasoning abilities, enabling them to perform tasks like translation, summarization, question answering, and content generation. GPT-3.5 and GPT-4 are prominent examples of LLMs.",
    "What is the current date and time?", # Perform Task (datetime tool)
    "What is the weather in Tokyo in celsius?", # Perform Task (weather tool)
    "Who was the first person to walk on the moon?", # Answer Question
    "Calculate 150 * 3 / 2.", # Perform Task (calculator tool)
    "Tell me a short story about a brave knight and a dragon." # Answer Question
]

for i, question in enumerate(agent_questions):
    print(f"\n{'='*20} Agent Turn {i+1} {'='*20}")
    print(f"User Question: {question}")
    
    # Reset messages for each new invocation for independent runs
    initial_input = {"messages": [HumanMessage(content=question)], "intent": None, "user_input_for_task": None}

    try:
        final_state = langgraph_app.invoke(
            initial_input,
            config={"recursion_limit": 50}, # Add recursion limit for safety
            # verbose=True # Uncomment for even more detailed LangGraph internal logs
        )
        print(f"\nFinal Agent Response: {final_state['messages'][-1].content}")
        print(f"Identified Intent (from state): {final_state.get('intent')}")
    except Exception as e:
        print(f"!!! Agent encountered an unexpected error during invocation: {e} !!!")
    print(f"{'='*50}\n")
