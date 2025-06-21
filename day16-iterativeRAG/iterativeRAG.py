import os
import json
from typing import TypedDict, Annotated, List, Union, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").lower()

# --- 1. Knowledge Base Setup ---
def create_knowledge_base(docs_dir="docs", embedding_provider="openai", ollama_embedding_model="nomic-embed-text"):
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        # Create dummy docs
        doc_contents = {
            "python_basics.txt": "Python is a high-level, interpreted, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented, and functional programming. It is often described as a 'batteries included' language due to its comprehensive standard library. Common uses for Python include web development (server-side), software development, mathematics, and system scripting. Popular frameworks like Django and Flask are built with Python.",
            "langchain_overview.txt": "LangChain is a framework designed to simplify the creation of applications using large language models (LLMs). It provides tools, components, and interfaces to build complex LLM applications that go beyond simple API calls. Key features include Chains (for sequential calls), Agents (for dynamic tool use), Retrieval (for RAG), and Callbacks (for observability). LangChain can be used for chatbots, summarization, data augmentation, code generation, and more. It supports various LLM providers, vector stores, and document loaders. The LangGraph library, built on LangChain, allows for building robust and stateful multi-actor applications by representing logic as a graph.",
            "ai_concepts.txt": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. AI can be categorized into narrow AI (designed for a specific task, like Siri or AlphaGo) and general AI (machines that can perform any intellectual task that a human being can). Machine learning, a subset of AI, involves training algorithms to learn patterns from data. Deep learning, a subset of machine learning, uses neural networks with many layers.",
            "internet_history.txt": "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and governmental networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies. The origins of the Internet date back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s to enable computer time-sharing and resource sharing among military researchers. The ARPANET, funded by DARPA, was an early packet switching network and the progenitor of the Internet. The commercialization of the ARPANET in the 1990s marked the beginning of the transition to the modern Internet.",
            "quantum_computing.txt": "Quantum computing is a rapidly emerging technology that uses the principles of quantum mechanics to solve problems too complex for classical computers. Unlike classical computers that store information as bits (0s or 1s), quantum computers use qubits, which can represent 0, 1, or both simultaneously (superposition), and also leverage entanglement and interference. This allows them to perform certain calculations exponentially faster. Key areas of application include drug discovery, material science, financial modeling, and breaking cryptography. Companies like IBM, Google, and Microsoft are actively investing in quantum computing research."
        }
        for filename, content in doc_contents.items():
            with open(os.path.join(docs_dir, filename), "w") as f:
                f.write(content)
        print(f"Created sample documents in '{docs_dir}' directory.")

    documents = []
    for f in os.listdir(docs_dir):
        if f.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_dir, f))
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    if embedding_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        embeddings = OpenAIEmbeddings()
    elif embedding_provider == "ollama":
        embeddings = OllamaEmbeddings(model=ollama_embedding_model)
    else:
        raise ValueError(f"Invalid embedding provider: {embedding_provider}. Must be 'openai' or 'ollama'.")

    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"Knowledge base (FAISS vector store) created successfully using {embedding_provider} embeddings.")
    return vectorstore

# Create the vector store
vectorstore = create_knowledge_base(embedding_provider=LLM_PROVIDER, ollama_embedding_model=OLLAMA_EMBEDDING_MODEL)
retriever = vectorstore.as_retriever(k=3) # Retrieve top 3 documents

# --- 2. LLM Initialization ---
def initialize_llm(provider, model_name=None, temp=0.7):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp)
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

llm = initialize_llm(LLM_PROVIDER)
print(f"Using LLM: {LLM_PROVIDER} ({llm.model_name if hasattr(llm, 'model_name') else OLLAMA_MODEL_CHAT})\n")

# --- 3. LangGraph State Definition ---
class RAGState(TypedDict):
    question: str  # The original user question
    documents: Annotated[List[str], add_messages] # Retrieved documents (as strings)
    generation: str # The final answer or current LLM output
    query_history: Annotated[List[str], add_messages] # Keep track of refined queries
    num_iterations: int # Track number of retrieval attempts
    
# --- 4. Define Tools (Retrieval) ---
@tool
def retrieve_documents(query: str) -> List[str]:
    """
    Retrieves relevant documents from the knowledge base based on the given query.
    """
    print(f"\n--- Tool Action: Retrieving documents for query: '{query}' ---")
    docs = retriever.invoke(query)
    # Convert Document objects to strings for easier handling in state
    return [doc.page_content for doc in docs]

tools = [retrieve_documents] # Our only tool for now

# --- 5. Define Graph Nodes ---

# Node 1: Initial Query Analysis / Refinement
def query_router_or_refine(state: RAGState) -> RAGState:
    print("--- Node: Query Analysis / Refinement ---")
    current_question = state['question']
    current_documents = state['documents']

    # On first run, or if previous documents were insufficient, refine/generate initial query
    if not current_documents or state['generation'] == "insufficient_documents":
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a query analysis and refinement assistant.
            The user is asking a question: {question}.
            If this is the first search, output the most effective search query for a knowledge base.
            If previous documents were found insufficient, output a refined search query that might yield better results.
            Always output only the refined/initial query string. Do NOT add any other text.
            """),
            ("human", f"Original question: {current_question}\nPrevious documents (if any):\n{state['documents']}\nPrevious attempt status: {state['generation']}")
        ])
        response = llm.invoke(prompt)
        refined_query = response.content.strip()
        print(f"  Generated/Refined Query: '{refined_query}'")
        return {"query_history": [refined_query], "generation": None} # Clear generation, add to history
    
    return {"query_history": [current_question]} # For direct initial retrieval if no refinement needed

# Node 2: Retrieve Documents
def retrieve(state: RAGState) -> RAGState:
    print("--- Node: Retrieving Documents ---")
    current_query = state['query_history'][-1] # Use the most recent query
    retrieved_docs = retrieve_documents.invoke({"query": current_query}) # Call the tool
    print(f"  Retrieved {len(retrieved_docs)} documents.")
    return {"documents": retrieved_docs, "num_iterations": state.get("num_iterations", 0) + 1}

# Node 3: Evaluate Retrieved Documents
def evaluate_and_decide(state: RAGState) -> RAGState:
    print("--- Node: Evaluating Documents and Deciding Next Step ---")
    question = state['question']
    documents = state['documents']
    
    if not documents:
        print("  Decision: No documents retrieved. Cannot answer.")
        return {"generation": "no_documents_found"}

    # Use LLM to decide if documents are sufficient
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a document evaluator for a RAG system.
        Based on the original question and the provided documents, determine if the documents are sufficient to fully answer the question.
        Respond with 'SUFFICIENT' if they are, or 'INSUFFICIENT' if more information or a refined search is needed.
        Also, provide a brief reasoning.
        Format your response as a JSON object: {"decision": "SUFFICIENT" | "INSUFFICIENT", "reasoning": "..."}
        """),
        ("human", f"Original Question: {question}\n\nDocuments:\n{'\n---\n'.join(documents)}\n\nDecision:")
    ])
    
    response = llm.invoke(eval_prompt)
    decision_raw = response.content.strip()
    
    try:
        if decision_raw.startswith("```json"):
            decision_raw = decision_raw[7:-3].strip()
        decision_json = json.loads(decision_raw)
        decision = decision_json.get("decision", "INSUFFICIENT").upper()
        reasoning = decision_json.get("reasoning", "No specific reason provided.")
        print(f"  Evaluation Decision: {decision} - Reason: {reasoning}")

        if decision == "SUFFICIENT":
            return {"generation": "sufficient_documents"}
        else:
            # If insufficient, signal for refinement or limit iterations
            if state.get("num_iterations", 0) >= 3: # Limit to 3 retrieval iterations
                print("  Reached max iterations. Cannot refine further.")
                return {"generation": "max_iterations_reached"}
            return {"generation": "insufficient_documents"} # Signal to refine

    except json.JSONDecodeError:
        print(f"  Warning: LLM returned non-JSON decision: {decision_raw[:50]}... Defaulting to INSUFFICIENT.")
        if state.get("num_iterations", 0) >= 3:
            return {"generation": "max_iterations_reached"}
        return {"generation": "insufficient_documents"}


# Node 4: Generate Final Answer
def generate_answer(state: RAGState) -> RAGState:
    print("--- Node: Generating Final Answer ---")
    question = state['question']
    documents = state['documents']
    
    if not documents or state['generation'] in ["no_documents_found", "max_iterations_reached"]:
        final_answer = "I'm sorry, I couldn't find enough relevant information to answer your question based on the available knowledge."
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful assistant. Use the provided documents to answer the question comprehensively.
            If the documents do not contain enough information, state that clearly.
            """),
            ("human", f"Question: {question}\n\nDocuments:\n{'\n---\n'.join(documents)}\n\nAnswer:")
        ])
        response = llm.invoke(prompt)
        final_answer = response.content.strip()

    return {"generation": final_answer, "messages": [AIMessage(content=final_answer)]} # Add final answer to messages for output


# --- 6. Define Routing Logic ---
def route_rag_workflow(state: RAGState) -> str:
    print(f"--- Router: RAG Workflow Router (Current Gen State: {state.get('generation')}) ---")
    current_generation_state = state.get('generation')

    if current_generation_state is None:
        # Initial state or after query refinement, go to retrieve
        print("  Decision: Initial retrieval or post-refinement. Going to 'retrieve_node'.")
        return "retrieve_node"
    elif current_generation_state == "insufficient_documents":
        # Docs were insufficient, need to refine query and re-retrieve
        print("  Decision: Insufficient documents. Going to 'query_router_or_refine' to refine query.")
        return "query_router_or_refine_node"
    elif current_generation_state in ["sufficient_documents", "no_documents_found", "max_iterations_reached"]:
        # We have enough docs, no docs, or hit max iterations, so generate final answer
        print(f"  Decision: {current_generation_state}. Going to 'generate_answer_node'.")
        return "generate_answer_node"
    else:
        # Fallback for unexpected state
        print("  Decision: Unexpected state. Going to 'generate_answer_node'.")
        return "generate_answer_node"


# --- 7. Build the LangGraph ---
print("--- Building the Iterative RAG Agent Graph ---")
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("query_router_or_refine_node", query_router_or_refine)
workflow.add_node("retrieve_node", retrieve)
workflow.add_node("evaluate_and_decide_node", evaluate_and_decide)
workflow.add_node("generate_answer_node", generate_answer)

# Set entry point
workflow.set_entry_point("query_router_or_refine_node") # Start by analyzing/refining query

# Define edges
# From initial query analysis to retrieval
workflow.add_edge("query_router_or_refine_node", "retrieve_node")

# From retrieval to evaluation
workflow.add_edge("retrieve_node", "evaluate_and_decide_node")

# From evaluation, route conditionally
workflow.add_conditional_edges(
    "evaluate_and_decide_node",
    route_rag_workflow, # Use our router
    {
        "query_router_or_refine_node": "query_router_or_refine_node", # Loop back to refine
        "generate_answer_node": "generate_answer_node" # Proceed to answer
    }
)

# From generate answer, end the workflow
workflow.add_edge("generate_answer_node", END)

# Compile the graph
langgraph_app = workflow.compile()
print("Iterative RAG agent compiled successfully.\n")

# --- 8. Invoke the Agent ---
print("--- Invoking the Iterative RAG Agent with Various Questions ---")

rag_questions = [
    "What are the main features of LangChain, and what is LangGraph?",
    "When was Python created and what are its key features?",
    "Tell me about the early history and origins of the internet.",
    "What is quantum computing and what are its potential applications?",
    "What is the capital of France?" # Question outside the knowledge base
]

for i, question in enumerate(rag_questions):
    print(f"\n{'='*20} RAG Agent Turn {i+1} {'='*20}")
    print(f"User Question: {question}")
    
    initial_input = {
        "question": question,
        "documents": [],
        "generation": None, # Initial state for generation
        "query_history": [],
        "num_iterations": 0
    }

    try:
        final_state = langgraph_app.invoke(
            initial_input,
            config={"recursion_limit": 50}, # Safety limit
            # verbose=True # Uncomment for verbose LangGraph internal logs
        )
        print(f"\nFinal Answer: {final_state['generation']}")
        print(f"Total Retrieval Iterations: {final_state.get('num_iterations', 0)}")
        print(f"Query History: {final_state.get('query_history', [])}")

    except Exception as e:
        print(f"!!! Agent encountered an unexpected error during invocation: {e} !!!")
    print(f"{'='*50}\n")
