# Save this as day7-lcel-advanced-patterns.py
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# LLM Provider for primary and fallback LLMs (you can mix and match)
# Set these in your .env file:
# PRIMARY_LLM_PROVIDER=openai
# FALLBACK_LLM_PROVIDER=ollama
# OLLAMA_MODEL_CHAT=llama2
PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "openai").lower()
FALLBACK_LLM_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()

# --- Initialize LLMs ---
def initialize_llm(provider, model_name=None, temp=0.7, timeout=None):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp, request_timeout=timeout)
    elif provider == "ollama":
        try:
            llm = ChatOllama(model=model_name or OLLAMA_MODEL_CHAT, temperature=temp)
            # Test connection (optional, but good for debugging)
            llm.invoke("Hello!")
            return llm
        except Exception as e:
            print(f"Error connecting to Ollama LLM or model '{model_name or OLLAMA_MODEL_CHAT}' not found: {e}")
            print("Please ensure Ollama is running and the specified model is pulled.")
            exit()
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'openai' or 'ollama'.")

# --- Example 1: Parallelism ---
print("--- Example 1: Parallelism with RunnableParallel ---")

# Define two different LLM instances or instructions
llm_fast = initialize_llm(PRIMARY_LLM_PROVIDER, temp=0.5, model_name="gpt-3.5-turbo" if PRIMARY_LLM_PROVIDER == "openai" else OLLAMA_MODEL_CHAT)
llm_creative = initialize_llm(PRIMARY_LLM_PROVIDER, temp=0.9, model_name="gpt-3.5-turbo" if PRIMARY_LLM_PROVIDER == "openai" else OLLAMA_MODEL_CHAT) # Can be same model, just different temp

# Define prompts for different outputs
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise summarizer. Summarize the following text briefly."),
    ("user", "{text}")
])

keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 3-5 keywords from the following text, separated by commas."),
    ("user", "{text}")
])

# Create parallel chains
parallel_chain = RunnableParallel(
    summary=summary_prompt | llm_fast | StrOutputParser(),
    keywords=keywords_prompt | llm_creative | StrOutputParser()
)

# Invoke the parallel chain
input_text = "LangChain is a framework for developing applications powered by language models. It enables chaining together different components to create more complex use cases around LLMs. This includes components for prompt management, LLMs, chat models, output parsers, retrievers, document loaders, and more. It emphasizes composability and supports the LangChain Expression Language (LCEL) for building flexible and robust chains."

print(f"\nInput Text: {input_text[:100]}...\n")
print("Running parallel chain...")
start_time = time.time()
parallel_output = parallel_chain.invoke({"text": input_text})
end_time = time.time()

print(f"Parallel execution took: {end_time - start_time:.2f} seconds")
print(f"Summary: {parallel_output['summary']}")
print(f"Keywords: {parallel_output['keywords']}\n")

# --- Example 2: Fallbacks ---
print("--- Example 2: Fallbacks with .with_fallbacks() ---")

# Define primary LLM (could be more expensive/prone to rate limits)
# We can simulate failure by setting a very short timeout or using a non-existent model
primary_llm = initialize_llm(PRIMARY_LLM_PROVIDER, temp=0.7, model_name="gpt-3.5-turbo" if PRIMARY_LLM_PROVIDER == "openai" else OLLAMA_MODEL_CHAT, timeout=0.01) # Simulate failure with a tiny timeout

# Define fallback LLM (could be cheaper/local/more reliable)
fallback_llm = initialize_llm(FALLBACK_LLM_PROVIDER, temp=0.7, model_name="gpt-3.5-turbo-0125" if FALLBACK_LLM_PROVIDER == "openai" else OLLAMA_MODEL_CHAT)

# Define a simple prompt
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# Create a chain with fallbacks
# The primary_llm will be tried first. If it fails, fallback_llm is used.
fallback_chain = (
    simple_prompt
    | primary_llm.with_fallbacks([fallback_llm])
    | StrOutputParser()
)

question_fallback = "What is the capital of France?"
print(f"Question for fallback: '{question_fallback}'")
print(f"Attempting to use primary LLM ({PRIMARY_LLM_PROVIDER}) first, falling back to ({FALLBACK_LLM_PROVIDER}) if needed...")

start_time_fallback = time.time()
try:
    response_fallback = fallback_chain.invoke({"question": question_fallback})
    print(f"Response: {response_fallback}")
except Exception as e:
    print(f"Fallback chain failed completely: {e}") # Should not happen if fallback is robust
finally:
    end_time_fallback = time.time()
    print(f"Fallback execution took: {end_time_fallback - start_time_fallback:.2f} seconds")

# Optional: Clean up dummy file if created (not applicable for this script)
