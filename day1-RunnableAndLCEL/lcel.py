import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv # Recommended for managing environment variables
 
# Load environment variables from .env file (if it exists)
load_dotenv()
 
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or use a .env file.")
 
 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful AI assistant. Respond concisely."),
    ("user", "{input}")
])
 
output_parser = StrOutputParser()
 
# The output of one Runnable becomes the input of the next.
# Flow: User Input -> Prompt (formats input) -> LLM (generates response) -> Output Parser (extracts string)
chain = prompt | llm | output_parser
 
 
def run_chain_example(query: str):
    """Invokes the LCEL chain with a given query and prints the response."""
    print(f"\n--- User Query: '{query}' ---")
    response = chain.invoke({"input": query}) # .invoke() is the Runnable method being called
    print(f"AI Response: {response}")
    print("-" * (len(query) + 20))
 
if __name__ == "__main__":
    print("Day 1: Hello, LangChain! - Your First LCEL Pipeline ")
    print("Understanding Runnables and LCEL for building robust GenAI applications.")
 
    # Example 1
    run_chain_example("What is the capital of Canada?")
 
    # Example 2
    run_chain_example("Tell me a short, interesting fact about the ocean.")
 
    # Example 3
    run_chain_example("Explain the concept of 'Runnable' in LangChain 0.3.")
