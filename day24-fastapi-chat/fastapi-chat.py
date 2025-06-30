from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration for LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower() # e.g., 'llama2', 'mistral'
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo") # e.g., 'gpt-4o', 'gpt-3.5-turbo'

# --- Initialize LLM ---
def get_llm():
    """Initializes and returns the ChatLargeLanguageModel based on provider."""
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider. Please set it.")
        return ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7)
    elif LLM_PROVIDER == "ollama":
        try:
            llm_instance = ChatOllama(model=OLLAMA_MODEL_CHAT, temperature=0.7)
            # Test connection (optional but good practice)
            llm_instance.invoke("test", config={"stream": False})
            return llm_instance
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama LLM '{OLLAMA_MODEL_CHAT}' or model not found: {e}. "
                               f"Please ensure Ollama is running and you have pulled the model: `ollama pull {OLLAMA_MODEL_CHAT}`") from e
    else:
        raise ValueError(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")

llm = get_llm()

# --- In-memory store for chat histories ---
# In a real application, this would be a persistent database (e.g., Redis, Postgres)
store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns a new BaseChatMessageHistory instance for a given session ID."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- LangChain Runnable with Message History ---
# Define the prompt template with a placeholder for messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer user questions concisely.",
        ),
        MessagesPlaceholder(variable_name="history"), # Placeholder for chat history
        ("human", "{input}"), # User's current input
    ]
)

# Create the base chain: prompt -> LLM -> output parser
chain = prompt | llm | StrOutputParser()

# Wrap the chain with RunnableWithMessageHistory
# `get_session_history` is a function that returns the history object for a given session_id
# `input_messages_key` tells LangChain which key in the input dictionary corresponds to the new message
# `history_messages_key` tells LangChain which key in the prompt expects the history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- FastAPI App Setup ---
app = FastAPI(
    title="LangChain Chatbot API",
    description="A simple FastAPI endpoint for a LangChain chat bot with conversational memory.",
    version="0.1.0",
)

# --- Pydantic Models for Request and Response ---
class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""
    session_id: str
    message: str

class ChatResponse(BaseModel):
    """Response schema for the chat endpoint."""
    session_id: str
    response: str
    message_count: int # For demonstration, show how many messages in history

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat messages, maintains conversation history, and returns AI response.
    """
    try:
        # Invoke the chain with the current input and session configuration
        # The session_id from the request is used by `get_session_history`
        response = await chain_with_history.ainvoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Retrieve the updated message count for the session
        current_history = get_session_history(request.session_id)
        message_count = len(current_history.messages)

        return ChatResponse(
            session_id=request.session_id,
            response=response,
            message_count=message_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "llm_provider": LLM_PROVIDER}

# --- How to run this app ---
# To run this file directly (for development with auto-reload):
# uvicorn day24-fastapi-chat:app --reload --host 0.0.0.0 --port 8000
#
# Open your browser to http://localhost:8000/docs for interactive API documentation.
# Test with a tool like curl:
# curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"session_id": "test_session_123", "message": "Hi there!"}'
# curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"session_id": "test_session_123", "message": "What did I just ask you?"}'
