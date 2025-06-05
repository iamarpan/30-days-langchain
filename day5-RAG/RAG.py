import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Set to 'openai' or 'ollama' to choose your embedding model
# You can set this in your .env file: EMBEDDING_PROVIDER=ollama
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

# --- Create a dummy document file (same as Day 4) ---
document_content = """
LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).
It provides a standardized interface for chains, agents, and retrieval.
The core idea is to allow chaining together different components to create more complex use cases.

One of the most important concepts in LangChain 0.3 is the Runnable interface.
Nearly every component, from prompts to models to parsers, implements Runnable.
This standardization enables seamless composition using the LangChain Expression Language (LCEL) and the pipe operator (|).

LCEL allows for highly declarative and concurrent execution of chains.
It supports streaming, asynchronous operations, and is designed for production-grade applications.
Understanding Runnables and LCEL is fundamental to mastering modern LangChain.

When building Retrieval-Augmented Generation (RAG) systems, preparing your data is crucial.
This involves using Document Loaders to ingest data from various sources like text files, PDFs, or web pages.
Once loaded, Text Splitters break down these large documents into smaller, manageable chunks.
These chunks are then typically embedded and stored in a vector database for efficient retrieval based on semantic similarity.
This process ensures that only relevant information is passed to the LLM, optimizing context window usage and improving response quality.
"""

file_path = "sample_document.txt"
with open(file_path, "w") as f:
    f.write(document_content)
print(f"Created '{file_path}' for demonstration.\n")

# --- Step 1: Load and Split the Document ---
print("--- Loading and Splitting Document ---")
loader = TextLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
print(f"Original document split into {len(chunks)} chunks.\n")

# --- Step 2: Initialize Embedding Model ---
embeddings = None
if EMBEDDING_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set for OpenAI embedding provider.")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    print("Using OpenAIEmbeddings (text-embedding-ada-002).")
elif EMBEDDING_PROVIDER == "ollama":
    try:
        # Ensure Ollama server is running and model is pulled (e.g., ollama pull nomic-embed-text)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Test connection by embedding a small string
        _ = embeddings.embed_query("test")
        print("Using OllamaEmbeddings (nomic-embed-text).")
    except Exception as e:
        print(f"Error connecting to Ollama or model 'nomic-embed-text' not found: {e}")
        print("Please ensure:")
        print("1. Ollama is installed and running (`ollama serve`).")
        print("2. The model 'nomic-embed-text' is pulled (`ollama pull nomic-embed-text`).")
        print("Exiting...")
        exit()
else:
    raise ValueError(f"Invalid EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}. Must be 'openai' or 'ollama'.")

# --- Step 3: Create a Vector Store (Chroma in-memory) ---
print("--- Creating Chroma Vector Store and Adding Documents ---")
# Chroma.from_documents takes chunks and the embedding model to create and populate the store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_30_days_embeddings" # Optional: name for your collection
)
print(f"Vector store created with {vectorstore._collection.count()} documents.\n")

# --- Step 4: Perform a Similarity Search ---
print("--- Performing Similarity Search ---")
query = "What is LCEL and why is it important?"
print(f"Query: '{query}'")

# Perform a similarity search to retrieve the top 2 most relevant chunks
retrieved_docs = vectorstore.similarity_search(query, k=2)

print("\n--- Retrieved Documents (Top 2) ---")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1} (Score/Relevance not shown by default, but these are closest):")
    print(f"  Content (first 150 chars): {doc.page_content[:150]}...")
    print(f"  Metadata: {doc.metadata}")
    print("-" * 30)

# Clean up the dummy file
# os.remove(file_path)
# print(f"\nCleaned up '{file_path}'.")
