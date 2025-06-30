import os
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# For evaluation
from langchain_evaluation import load_evaluator
from langchain_evaluation.schema import EvaluatorType

load_dotenv()

# --- Configuration for LLM and Embeddings ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()
OLLAMA_MODEL_EMBED = os.getenv("OLLAMA_MODEL_EMBED", "nomic-embed-text").lower()
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo")
OPENAI_MODEL_EMBED = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small") # text-embedding-ada-002 also common

# --- Initialize LLM and Embeddings ---
def get_llm_and_embeddings():
    llm = None
    embeddings = None

    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0) # Lower temp for evaluation
        embeddings = OpenAIEmbeddings(model=OPENAI_MODEL_EMBED)
    elif LLM_PROVIDER == "ollama":
        try:
            llm = ChatOllama(model=OLLAMA_MODEL_CHAT, temperature=0) # Lower temp for evaluation
            llm.invoke("test", config={"stream": False})
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama chat LLM '{OLLAMA_MODEL_CHAT}': {e}") from e
        
        try:
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_EMBED)
            embeddings.embed_query("test")
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama embedding model '{OLLAMA_MODEL_EMBED}': {e}") from e
    else:
        raise ValueError(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")
    
    return llm, embeddings

# Get the LLM for the RAG chain and a separate LLM for evaluation (can be the same or different model)
rag_llm, embeddings_model = get_llm_and_embeddings()

# For evaluation, it's often good to use a more capable LLM, potentially even gpt-4 for robust evaluation
# If using OpenAI as LLM_PROVIDER, use a specific model for evaluation if desired.
eval_llm = ChatOpenAI(model="gpt-4o", temperature=0) if LLM_PROVIDER == "openai" else rag_llm # Use a strong LLM for evaluation if available


# --- 1. Define a small, hardcoded knowledge base ---
knowledge_base_text = """
The LangChain framework simplifies the development of applications powered by large language models (LLMs).
It provides a standard interface for chains, over 50 tools, and integrates with various LLMs, vector stores, and agents.
LangChain supports Python and JavaScript. It was initially released in October 2022.
The core components include Models, Prompts, Chains, Retrieval, Agents, and Callbacks.
Models refer to the LLMs or ChatModels. Prompts are templates for inputs to models.
Chains combine LLMs with other components. Retrieval focuses on RAG for grounding LLMs.
Agents enable LLMs to choose and use tools. Callbacks allow logging and monitoring.
LangGraph is a library built on LangChain that enables building robust and stateful multi-actor applications with LLMs.
It uses stateful FSMs (Finite State Machines) to orchestrate complex sequences of LLM calls, tool uses, and human interventions.
LangGraph is particularly useful for building agents that can reliably perform multiple steps.
"""

# --- 2. Create a RAG chain ---
# Split the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([knowledge_base_text])

# Create a simple in-memory vector store
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings_model)
retriever = vectorstore.as_retriever()

# RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use the following retrieved context to answer the question. "
               "If you don't know the answer based *only* on the context, state that you don't know. "
               "Context: {context}"),
    ("human", "{question}")
])

# RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)

# --- 3. Define a small set of ground truth questions and answers for evaluation ---
# In a real scenario, this dataset would be much larger and more diverse.
# We also include 'expected_contexts' to help with manual context recall checks
# and to provide a reference for automated evaluators if they supported it.
evaluation_dataset = [
    {
        "question": "When was LangChain first released?",
        "ground_truth_answer": "LangChain was initially released in October 2022.",
        "expected_contexts": ["It was initially released in October 2022."]
    },
    {
        "question": "What is LangGraph used for?",
        "ground_truth_answer": "LangGraph is a library built on LangChain that enables building robust and stateful multi-actor applications with LLMs, using stateful FSMs to orchestrate complex sequences of LLM calls, tool uses, and human interventions.",
        "expected_contexts": ["LangGraph is a library built on LangChain that enables building robust and stateful multi-actor applications with LLMs.", "It uses stateful FSMs (Finite State Machines) to orchestrate complex sequences of LLM calls, tool uses, and human interventions.", "LangGraph is particularly useful for building agents that can reliably perform multiple steps."]
    },
    {
        "question": "What programming languages does LangChain support?",
        "ground_truth_answer": "LangChain supports Python and JavaScript.",
        "expected_contexts": ["LangChain supports Python and JavaScript."]
    },
    {
        "question": "What is the capital of France?",
        "ground_truth_answer": "I don't know the answer based on the provided context.", # Expected out-of-context answer
        "expected_contexts": [] # No relevant contexts in our document
    }
]

# --- 4. Set up evaluators using LLM-as-a-Judge ---
# Using the "criteria" type evaluator to check faithfulness and relevancy
# For faithfulness, we ask if the answer is supported by the context.
# For answer relevancy, we ask if the answer directly addresses the question.

# Evaluator for Faithfulness
evaluator_faithfulness = load_evaluator(
    EvaluatorType.CRITERIA,
    llm=eval_llm,
    criteria={
        "faithfulness": "Is the generated answer fully supported by the provided context? Rate YES or NO."
    },
    requires_input=True, # Need the input (question)
    requires_reference=False, # Don't need ground truth answer for faithfulness to context
    requires_context=True # Need the retrieved context for faithfulness
)

# Evaluator for Answer Relevancy
evaluator_relevancy = load_evaluator(
    EvaluatorType.CRITERIA,
    llm=eval_llm,
    criteria={
        "answer_relevancy": "Is the generated answer relevant to the question? Rate YES or NO."
    },
    requires_input=True, # Need the question
    requires_reference=False, # Don't need ground truth answer for relevancy to question
    requires_context=False # Don't strictly need context for relevancy to question
)

# --- 5. Run the evaluation pipeline ---
print("--- Starting RAG Evaluation ---")
evaluation_results = []

for i, example in enumerate(evaluation_dataset):
    question = example["question"]
    ground_truth = example["ground_truth_answer"]
    
    print(f"\n--- Evaluating Question {i+1}: '{question}' ---")

    # Get retrieved documents separately to pass to evaluator
    retrieved_docs = retriever.invoke(question)
    retrieved_context_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Invoke the RAG chain
    generated_answer = rag_chain.invoke(question)

    print(f"  Generated Answer: {generated_answer}")
    print(f"  Expected Answer : {ground_truth}")
    print(f"  Retrieved Contexts: {len(retrieved_docs)} chunks")
    # for doc in retrieved_docs:
    #     print(f"    - ...{doc.page_content[-100:]}") # Print last 100 chars of each doc

    # Evaluate Faithfulness
    faithfulness_result = evaluator_faithfulness.evaluate_strings(
        input=question,
        prediction=generated_answer,
        context=retrieved_context_content,
    )
    faithfulness_score = faithfulness_result.get("score")
    faithfulness_reasoning = faithfulness_result.get("reasoning")
    print(f"  Faithfulness: {faithfulness_score} ({faithfulness_reasoning})")

    # Evaluate Answer Relevancy
    relevancy_result = evaluator_relevancy.evaluate_strings(
        input=question,
        prediction=generated_answer,
    )
    relevancy_score = relevancy_result.get("score")
    relevancy_reasoning = relevancy_result.get("reasoning")
    print(f"  Answer Relevancy: {relevancy_score} ({relevancy_reasoning})")

    evaluation_results.append({
        "question": question,
        "ground_truth_answer": ground_truth,
        "generated_answer": generated_answer,
        "retrieved_context_count": len(retrieved_docs),
        "faithfulness_score": faithfulness_score,
        "faithfulness_reasoning": faithfulness_reasoning,
        "relevancy_score": relevancy_score,
        "relevancy_reasoning": relevancy_reasoning
    })

print("\n--- Evaluation Summary ---")
total_faithfulness_yes = sum(1 for res in evaluation_results if res["faithfulness_score"] == "YES")
total_relevancy_yes = sum(1 for res in evaluation_results if res["relevancy_score"] == "YES")
total_examples = len(evaluation_dataset)

print(f"Total Examples Evaluated: {total_examples}")
print(f"Faithfulness Score (YES): {total_faithfulness_yes}/{total_examples} ({total_faithfulness_yes/total_examples:.2%})")
print(f"Answer Relevancy Score (YES): {total_relevancy_yes}/{total_examples} ({total_relevancy_yes/total_examples:.2%})")

# Clean up Chroma (optional, if using persistent directory)
# vectorstore.delete_collection()
