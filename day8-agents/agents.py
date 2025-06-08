import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import tool # Correct import for @tool decorator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# LLM Provider (from Day 3)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Step 1: Define Custom Tools ---
# Use the @tool decorator to easily create a tool from a function
@tool
def word_reverser(word: str) -> str:
    """Reverses a given word or string."""
    print(f"\n--- Tool Action: Reversing '{word}' ---") # For demonstration visibility
    return word[::-1]

@tool
def character_counter(text: str) -> int:
    """Counts the number of characters in a given string."""
    print(f"\n--- Tool Action: Counting characters in '{text}' ---") # For demonstration visibility
    return len(text)

# List of tools available to the agent
tools = [word_reverser, character_counter]
print(f"Available tools: {[tool.name for tool in tools]}\n")


# --- Step 2: Initialize LLM ---
def initialize_llm(provider, model_name=None, temp=0.7):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp)
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

# --- Step 3: Create the Agent Prompt ---
# The prompt is crucial for guiding the LLM to act as an agent (ReAct style)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. You have access to the following tools: {tools}. "
               "You should use these tools to answer the user's questions. "
               "If a tool is suitable, first think about what tool to use and its input. "
               "Respond in the following format:\n\n"
               "Thought: (Your reasoning about what to do)\n"
               "Action: (The tool to call, exactly as specified, with input)\n"
               "Action Input: (The input to the tool)\n"
               "Observation: (The result of the tool)\n"
               "... (this Thought/Action/Observation can repeat multiple times)\n"
               "Thought: (Final thought before providing the answer)\n"
               "Final Answer: (The ultimate answer to the user's question)\n\n"
               "If you don't need a tool, just provide a direct answer."),
    MessagesPlaceholder(variable_name="chat_history"), # For future history (optional for this simple demo)
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") # Important for ReAct, stores thought/action/observation
])

# --- Step 4: Create the ReAct Agent ---
agent = create_react_agent(llm, tools, prompt)
print("Agent created using create_react_agent.\n")

# --- Step 5: Create the Agent Executor ---
# The AgentExecutor runs the agent's loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
# verbose=True shows the thought/action/observation steps

# --- Step 6: Invoke the Agent with Questions ---
questions = [
    "Reverse the word 'hello'.",
    "How many characters are in the string 'LangChain is amazing'?",
    "What is the capital of Germany?", # Should not use a tool
    "Reverse the word 'python' and then count characters in the reversed word."
]

print("--- Invoking Agent ---")
for q in questions:
    print(f"\nUser Question: {q}")
    try:
        # chat_history is empty for this simple, single-turn demo
        response = agent_executor.invoke({"input": q, "chat_history": []})
        print(f"Agent Final Answer: {response['output']}")
    except Exception as e:
        print(f"Agent encountered an error: {e}")
    print("\n" + "="*70 + "\n")
