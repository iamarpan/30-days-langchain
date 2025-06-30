import streamlit as st
import os
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings # For local LLM and embeddings
from langchain.document_loaders import PyPDFLoader # For loading PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter # For chunking
from langchain_community.vectorstores import Chroma # Our vector store
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration for LLM and Embeddings ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()
OLLAMA_MODEL_EMBED = os.getenv("OLLAMA_MODEL_EMBED", "nomic-embed-text").lower()
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo")
OPENAI_MODEL_EMBED = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-ada-002")

# --- Initialize LLM and Embeddings ---
@st.cache_resource
def get_llm_and_embeddings():
    """Initializes and returns LLM and Embeddings based on provider."""
    llm = None
    embeddings = None

    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set for OpenAI provider. Please set it.")
            st.stop()
        llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.3)
        embeddings = OpenAIEmbeddings(model=OPENAI_MODEL_EMBED)
    elif LLM_PROVIDER == "ollama":
        try:
            llm = ChatOllama(model=OLLAMA_MODEL_CHAT, temperature=0.3)
            # Test chat LLM connection
            llm.invoke("test", config={"stream": False})
            st.success(f"Successfully connected to Ollama chat model: {OLLAMA_MODEL_CHAT}")
        except Exception as e:
            st.error(f"Error connecting to Ollama chat LLM '{OLLAMA_MODEL_CHAT}': {e}")
            st.info(f"Please ensure Ollama is running and you have pulled the model: `ollama pull {OLLAMA_MODEL_CHAT}`")
            st.stop()
        
        try:
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_EMBED)
            # Test embedding model connection
            embeddings.embed_query("test")
            st.success(f"Successfully connected to Ollama embedding model: {OLLAMA_MODEL_EMBED}")
        except Exception as e:
            st.error(f"Error connecting to Ollama embedding model '{OLLAMA_MODEL_EMBED}': {e}")
            st.info(f"Please ensure Ollama is running and you have pulled the embedding model: `ollama pull {OLLAMA_MODEL_EMBED}`")
            st.stop()
    else:
        st.error(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")
        st.stop()
    
    return llm, embeddings

llm, embeddings = get_llm_and_embeddings()


# --- Streamlit App Setup ---
st.set_page_config(page_title="RAG Chat with Your Documents", page_icon="📚")
st.title("📚 RAG Chat with Your Documents")
st.markdown(f"*(LLM: {LLM_PROVIDER.capitalize()} {OPENAI_MODEL_CHAT if LLM_PROVIDER == 'openai' else OLLAMA_MODEL_CHAT}, Embeddings: {OPENAI_MODEL_EMBED if LLM_PROVIDER == 'openai' else OLLAMA_MODEL_EMBED})*")
st.markdown("---")

# --- Initialize chat history in session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores list of {"role": "user" or "assistant", "content": "message text", "sources": []}

# --- Initialize vector store in session state ---
# This will hold our document embeddings
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Document Upload and Processing ---
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document",
    type="pdf",
    accept_multiple_files=False,
    key="pdf_uploader"
)

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing document... This may take a moment."):
        try:
            # 1. Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # 2. Load the document
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            if not docs:
                st.warning("Could not extract text from the PDF. Please try another file.")
                os.unlink(tmp_file_path) # Clean up temp file
                st.stop()

            # 3. Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)

            # 4. Create embeddings and store in Chroma
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            st.sidebar.success(f"Document '{uploaded_file.name}' processed and ready for questions!")
            # Clean up temporary file after processing
            os.unlink(tmp_file_path)
        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")
            st.session_state.vectorstore = None # Reset vectorstore on error
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
# --- Display chat messages from history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.text(source)

# --- Handle user input ---
if prompt := st.chat_input("Ask a question about the document..."):
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF document first to enable RAG.")
    else:
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare RAG chain
        retriever = st.session_state.vectorstore.as_retriever()
        
        # Define RAG prompt
        rag_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Use the following retrieved context to answer the question. "
                       "If you don't know the answer, state that you don't know. Keep your answer concise and to the point. "
                       "Context: {context}"),
            ("human", "{question}")
        ])

        # Define RAG chain with source retrieval
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt_template
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and Generating..."):
                full_response = ""
                response_container = st.empty()
                
                # We need to invoke the retriever separately to get sources
                retrieved_docs = retriever.invoke(prompt)
                context_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Now, invoke the full chain with the context and question
                # For streaming, we pass the context and question to the prompt directly
                # and then stream the LLM response.
                chain_with_context = (
                    rag_prompt_template | llm | StrOutputParser()
                )
                
                for chunk in chain_with_context.stream({
                    "context": context_content,
                    "question": prompt
                }):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)

                # Collect and display sources
                source_texts = []
                for i, doc in enumerate(retrieved_docs):
                    source_texts.append(f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content[:200]}...") # Display first 200 chars

                st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": source_texts})
                
                if source_texts:
                    with st.expander("Sources"):
                        for source in source_texts:
                            st.text(source)


# --- How to run this app ---
st.sidebar.markdown("---")
st.sidebar.markdown("### How to run")
st.sidebar.markdown("1. Save this code as `day23-rag-file-upload.py`")
st.sidebar.markdown("2. Open your terminal in the same directory.")
st.sidebar.markdown("3. Run the command: `streamlit run day23-rag-file-upload.py`")
st.sidebar.markdown("4. Your browser will open with the RAG application.")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Dependencies")
st.sidebar.markdown("`pip install streamlit langchain-openai langchain-ollama chromadb pypdf unstructured tiktoken python-dotenv`")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Ollama Setup")
st.sidebar.markdown(f"Ensure Ollama is running and models pulled:")
st.sidebar.markdown(f"`ollama pull {OLLAMA_MODEL_CHAT}`")
st.sidebar.markdown(f"`ollama pull {OLLAMA_MODEL_EMBED}`")
