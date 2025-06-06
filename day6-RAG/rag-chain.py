# Save this as day6-basic-rag-chain.py
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# LLM and Embedding provider configuration (from Day 3 & 5)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

# --- Create a dummy document file (reusing from Day 4) ---
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

Retrievers are a core component in a RAG pipeline. They are responsible for fetching the most relevant documents
from a knowledge base, typically a vector store, based on a user's query.
LangChain's `VectorStoreRetriever` converts a vector store into a callable retriever.
You can configure retrievers with parameters like `k` (number of documents to retrieve) and `search_type`
(e.g., 'similarity' or 'mmr' for diversity).
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
    chunk_size=300, # Slightly increased chunk size for more coherent RAG context
    chunk_overlap=50,
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
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        _ = embeddings.embed_query("test") # Test connection
        print("Using OllamaEmbeddings (nomic-embed-text).")
    except Exception as e:
        print(f"Error connecting to Ollama or model 'nomic-embed-text' not found: {e}")
        print("Please ensure Ollama is running and 'nomic-embed-text' is pulled.")
        exit()
else:
    raise ValueError(f"Invalid EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}.")

# --- Step 3: Create Vector Store and Retriever ---
print("--- Creating Chroma Vector Store and Retriever ---")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_30_days_rag_collection"
)

# Convert the vectorstore into a retriever
# search_kwargs={"k": 2} means retrieve the top 2 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print(f"Retriever created with k=2.\n")


# --- Step 4: Initialize LLM ---
llm = None
if LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) # Lower temp for more factual RAG
    print("Using OpenAI GPT-3.5-Turbo as LLM.")
elif LLM_PROVIDER == "ollama":
    try:
        llm = ChatOllama(model="llama2", temperature=0.3) # Use a general chat model for RAG
        llm.invoke("Hello!") # Test connection
        print("Using local Ollama Llama 2 as LLM.")
    except Exception as e:
        print(f"Error connecting to Ollama LLM or model 'llama2' not found: {e}")
        print("Please ensure Ollama is running and 'llama2' is pulled.")
        exit()
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}.")


# --- Step 5: Define the RAG Prompt Template ---
# This prompt structure is crucial for RAG
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use the following retrieved context to answer the question. "
               "If the answer is not in the context, clearly state that you don't have enough information.\n\n"
               "Context: {context}"),
    ("user", "{question}")
])

# --- Step 6: Construct the LCEL RAG Chain ---
print("--- Constructing the LCEL RAG Chain ---")
rag_chain = (
    # This dictionary prepares the input for the prompt.
    # 'context' will be populated by the retriever.
    # 'question' will be the original user query, passed through.
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("RAG chain constructed.\n")

# --- Step 7: Invoke the RAG Chain with Questions ---
questions = [
    "What is LangChain and what does it simplify?",
    "Explain the Runnable interface in LangChain 0.3.",
    "What is the purpose of text splitters in RAG systems?",
    "Who invented the internet?", # Question not in context
    "What is LCEL designed for?"
]

print("--- Answering Questions using RAG Chain ---")
for q in questions:
    print(f"\nQ: {q}")
    response = rag_chain.invoke({"question": q})
    print(f"A: {response}")
    print("-" * 50)

# Optional: Clean up the dummy file
# os.remove(file_path)
# print(f"\nCleaned up '{file_path}'.")
