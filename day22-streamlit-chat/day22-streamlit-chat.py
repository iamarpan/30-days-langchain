import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (for OpenAI API key or Ollama model names)
from dotenv import load_dotenv
load_dotenv()

# --- Configuration for LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower() # e.g., 'llama2', 'mistral'
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo") # e.g., 'gpt-4o', 'gpt-3.5-turbo'

# --- LLM Initialization ---
def get_llm():
    """Initializes and returns the ChatLargeLanguageModel based on provider."""
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set for OpenAI provider. Please set it in your .env file or environment variables.")
            st.stop() # Stop the app if API key is missing
        return ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7)
    elif LLM_PROVIDER == "ollama":
        try:
            llm_instance = ChatOllama(model=OLLAMA_MODEL_CHAT, temperature=0.7)
            # Test connection (optional but good practice)
            llm_instance.invoke("test", config={"stream": False})
            return llm_instance
        except Exception as e:
            st.error(f"Error connecting to Ollama LLM '{OLLAMA_MODEL_CHAT}' or model not found: {e}")
            st.info(f"Please ensure Ollama is running and you have pulled the model: `ollama pull {OLLAMA_MODEL_CHAT}`")
            st.stop() # Stop the app if Ollama fails
    else:
        st.error(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")
        st.stop()

llm = get_llm()

# --- Streamlit App Setup ---
st.set_page_config(page_title="LangChain Chatbot", page_icon="💬")
st.title("LangChain Chatbot")
st.markdown(f"*{LLM_PROVIDER.capitalize()} model: {OPENAI_MODEL_CHAT if LLM_PROVIDER == 'openai' else OLLAMA_MODEL_CHAT}*")
st.markdown("---")

# --- Initialize chat history in session state ---
# This ensures messages persist across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores list of {"role": "user" or "assistant", "content": "message text"}

# --- Display chat messages from history ---
# Iterate through the messages stored in session state and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle user input ---
# st.chat_input creates an input box at the bottom of the page
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for the LLM
    # LangChain models typically expect BaseMessage objects, so convert if needed
    # For this simple chat, we'll just pass the whole history
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    # Call the LLM
    with st.chat_message("assistant"):
        # Use a spinner to indicate thinking
        with st.spinner("Thinking..."):
            # A simple chain: prompt -> LLM -> string output parser
            # We construct a simple prompt to include chat history for context
            chat_prompt = ChatPromptTemplate.from_messages(langchain_messages)
            
            chain = chat_prompt | llm | StrOutputParser()
            
            # Stream the response for a better UX
            full_response = ""
            response_container = st.empty() # Placeholder for streaming text
            for chunk in chain.stream({}): # No input vars needed if prompt is already full messages
                full_response += chunk
                response_container.markdown(full_response + "▌") # Add a blinking cursor effect
            
            response_container.markdown(full_response) # Display final response without cursor

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- How to run this app ---
st.sidebar.markdown("### How to run")
st.sidebar.markdown("1. Save this code as `day22-streamlit-chat.py`")
st.sidebar.markdown("2. Open your terminal in the same directory.")
st.sidebar.markdown("3. Run the command: `streamlit run day22-streamlit-chat.py`")
st.sidebar.markdown("4. Your browser will open with the chat application.")
st.sidebar.markdown("---")
st.sidebar.markdown("#### LLM Configuration")
st.sidebar.markdown(f"**Provider:** `{LLM_PROVIDER.capitalize()}`")
if LLM_PROVIDER == 'openai':
    st.sidebar.markdown(f"**Model:** `{OPENAI_MODEL_CHAT}`")
else:
    st.sidebar.markdown(f"**Model:** `{OLLAMA_MODEL_CHAT}`")
st.sidebar.markdown("*Set `LLM_PROVIDER` and model names in your `.env` file.*")
