import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Set to 'openai' or 'ollama' to choose your LLM
# You can set this in your .env file: LLM_PROVIDER=ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # Default to openai if not set

# --- Initialize LLM based on configuration ---
llm = None
if LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set. Please set it for OpenAI provider.")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    print("Using OpenAI GPT-3.5-Turbo.")
elif LLM_PROVIDER == "ollama":
    try:
        # Ensure Ollama server is running and model is pulled (e.g., ollama pull llama2)
        llm = ChatOllama(model="llama2", temperature=0.7)
        # Test connection by making a small call (optional, but good for debugging)
        llm.invoke("Hello!")
        print("Using local Ollama Llama 2.")
    except Exception as e:
        print(f"Error connecting to Ollama or model 'llama2' not found: {e}")
        print("Please ensure:")
        print("1. Ollama is installed and running (`ollama serve`).")
        print("2. The model 'llama2' is pulled (`ollama pull llama2`).")
        print("Exiting...")
        exit()
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")

# --- Define the Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise AI assistant. Respond briefly and to the point."),
    ("user", "{question}")
])

# --- Define the Output Parser ---
output_parser = StrOutputParser()

# --- Construct the LCEL Chain ---
# The chain structure remains identical regardless of the LLM provider
chain = prompt | llm | output_parser

# --- Invoke the Chain with questions ---
questions = [
    "What is the capital of Japan?",
    "Explain quantum entanglement in one sentence.",
    "What's a creative use case for LLMs?"
]

print("\n--- Conversational Responses ---")
for q in questions:
    response = chain.invoke({"question": q})
    print(f"Q: {q}")
    print(f"A: {response}\n")

# Example of changing model parameters (e.g., temperature) if desired
# For a specific invocation, you can use .with_config
print("\n--- Example with changed temperature for one invocation ---")
high_temp_response = chain.with_config(run_name="high_temp_query").invoke({"question": "Tell me a very creative and imaginative story idea in 2-3 sentences."}, config={"run_config": {"llm_config": {"temperature": 1.5}}})
print(f"Q: Tell me a very creative and imaginative story idea in 2-3 sentences.")
print(f"A: {high_temp_response}\n")
