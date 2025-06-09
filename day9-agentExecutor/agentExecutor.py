import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import tool
from langchain.memory import ConversationBufferMemory # Import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Step 1: Define Custom Tools (reusing from Day 8) ---
@tool
def word_reverser(word: str) -> str:
    """Reverses a given word or string."""
    print(f"\n--- Tool Action: Reversing '{word}' ---")
    return word[::-1]

@tool
def character_counter(text: str) -> int:
    """Counts the number of characters in a given string."""
    print(f"\n--- Tool Action: Counting characters in '{text}' ---")
    return len(text)

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

# --- Step 3: Create the Agent Prompt with Memory Placeholder ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. You have access to the following tools: {tools}. "
               "You should use these tools to answer the user's questions. "
               "Respond in the ReAct format: Thought, Action, Action Input, Observation, Final Answer. "
               "If you don't need a tool, just provide a direct answer. "
               "Maintain context from previous turns."),
    MessagesPlaceholder(variable_name="chat_history"), # THIS IS WHERE MEMORY WILL BE INJECTED
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Step 4: Create the ReAct Agent ---
agent = create_react_agent(llm, tools, prompt)
print("Agent created using create_react_agent.\n")

# --- Step 5: Initialize Memory ---
# We use a simple buffer memory for this example
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
print("ConversationBufferMemory initialized.\n")

# --- Step 6: Create the Agent Executor with Memory ---
# Now we pass the memory's chat history to the executor's input
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    # Here, we don't directly pass 'memory' to AgentExecutor for simple agents.
    # Instead, we'll manually pass the memory's content in the invoke call.
    # This setup is common for ReAct agents where memory is just another part of the prompt.
)
print("Agent Executor created.\n")

# --- Step 7: Conduct a Multi-Turn Conversation ---
print("--- Starting Multi-Turn Conversation with Agent ---")

conversation_turns = [
    "What is the capital of France?",
    "Reverse the word 'LangChain'.",
    "How many characters are in that reversed word?", # Follow-up question referencing previous turn
    "Tell me about large language models."
]

for i, q in enumerate(conversation_turns):
    print(f"\n--- Turn {i+1} ---")
    print(f"User Question: {q}")
    try:
        # Pass the current chat history from memory to the agent's prompt
        # The 'chat_history' key here matches the variable_name in MessagesPlaceholder
        response = agent_executor.invoke({"input": q, "chat_history": memory.load_memory_variables({})["chat_history"]})
        print(f"Agent Final Answer: {response['output']}")

        # Save the current turn's interaction to memory for the next turn
        memory.save_context({"input": q}, {"output": response["output"]})

    except Exception as e:
        print(f"Agent encountered an error: {e}")
    print("\n" + "="*70 + "\n")

# --- Optional: Demonstrate return_intermediate_steps ---
print("--- Demonstrating return_intermediate_steps ---")
query_return_intermediate = "Reverse the word 'LangChain'."
print(f"Query: {query_return_intermediate}")

# Note: AgentExecutor's invoke can return intermediate steps
# You might need to adjust the prompt or agent type for explicit intermediate steps
# returned as part of the output dictionary. For basic create_react_agent,
# verbose=True often gives enough insight. If return_intermediate_steps=True
# was added to AgentExecutor, the invoke output dict would contain 'intermediate_steps'.
# For now, verbose output itself shows intermediate steps.

# Let's show how memory looks after the conversation
print("--- Current Memory Content (from ConversationBufferMemory) ---")
print(memory.load_memory_variables({})["chat_history"])
print("-" * 70)
