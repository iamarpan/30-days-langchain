from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import json
from typing import Dict, Any, AsyncGenerator, Tuple, List, Optional
import asyncio

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor, ToolNode

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration for LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo")

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
            llm_instance.invoke("test", config={"stream": False}) # Test connection
            return llm_instance
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama LLM '{OLLAMA_MODEL_CHAT}': {e}. "
                               f"Ensure Ollama is running: `ollama run {OLLAMA_MODEL_CHAT}`") from e
    else:
        raise ValueError(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")

llm = get_llm().bind_tools([]) # Initially bind no tools, will update for agent

# --- Define a simple tool for the agent ---
@tool
def search_web(query: str) -> str:
    """Simulates searching the web for information."""
    # In a real app, you'd use a real search API (e.g., DuckDuckGoSearch, Google Search)
    if "current time" in query.lower():
        return f"The current time in Hyderabad, India is {asyncio.get_event_loop().time()}" # placeholder for current time
    return f"Simulated search result for '{query}': Information about '{query}' found here: [link to relevant info]"

tools = [search_web]
tool_names = [tool.name for tool in tools]
tool_executor = ToolExecutor(tools)

# --- Define the LangGraph Agent State ---
class AgentState(BaseModel):
    messages: List[BaseMessage]

# --- Define the agent nodes ---
def call_model(state: AgentState) -> dict:
    messages = state.messages
    response = llm.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState) -> dict:
    last_message = state.messages[-1]
    tool_calls = last_message.tool_calls
    
    if not tool_calls:
        # This should ideally not happen if the agent decided to call a tool
        return {"messages": [AIMessage(content="Agent decided to call a tool but no tool calls found.")]}

    tool_outputs = []
    for tool_call in tool_calls:
        tool_output = tool_executor.invoke(tool_call)
        tool_outputs.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call.id))
    return {"messages": tool_outputs}

# --- Define the agent graph ---
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("llm", call_model)
workflow.add_node("tool", call_tool)

# Define the edges
workflow.add_edge(START, "llm")
workflow.add_edge("tool", "llm") # After tool execution, go back to LLM to summarize/continue

# Define the conditional edge for the LLM node
def should_continue(state: AgentState) -> str:
    messages = state.messages
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end" # If no tool calls, it's the final answer
    return "tool" # If there are tool calls, execute the tool

workflow.add_conditional_edges(
    "llm", # From LLM node
    should_continue,
    {"tool": "tool", "end": END}
)

# Compile the graph
agent_app = workflow.compile()
# Re-bind tools to the LLM now that the graph is defined
llm = get_llm().bind_tools(tools)


# --- In-memory store for chat histories (for RunnableWithMessageHistory) ---
# In a real application, this would be a persistent database
store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- LangChain Runnable with Message History (for the LangGraph agent) ---
# Prompt template for the LangGraph agent
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant with access to tools. If the user asks a question "
            "that requires external information (like current events, factual lookup, "
            "or specific data that you don't have), use the 'search_web' tool. "
            "Always respond to the user based on the tool results or your knowledge. "
            "Do not just repeat tool calls. "
            "History: {history}", # Placeholder for history
        ),
        ("human", "{input}"), # User's current input
        MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's internal monologue/tool calls
    ]
)

# Replace the base LLM with a version that has tools bound
# The LLM inside the graph (call_model) also needs tools bound if it's the one making tool_calls.
# This ensures that the agent can generate tool call messages.
# We ensure the LLM used in the agent graph also has tools bound.
llm_with_tools = get_llm().bind_tools(tools)

# The agent itself is the combination of prompt, LLM with tools, and its graph execution.
# We need to explicitly pass the tools to the LLM that's part of the prompt in the agent workflow.
# This needs careful integration if the agent's prompt directly involves tools,
# but for a simple agent like this, the 'should_continue' logic relies on the LLM's output.

# Compile the graph again if LLM binding changes, or ensure LLM is bound before compiling.
# For simplicity, we ensure the `llm` variable (which `call_model` uses) is bound.
# The graph is compiled once, and then `llm` used in `call_model` should be the one with tools.


# The actual chain that RunnableWithMessageHistory will wrap
# This is a simplified chain for the RunnableWithMessageHistory; the LangGraph agent will manage the internal steps.
# The `agent_app` (compiled graph) will be the runnable being invoked.
# We need `RunnableWithMessageHistory` to manage `history` for the *entire* agent execution.
# The `agent_app` expects `messages` as input.

# The input to the agent_app needs to be a list of BaseMessages.
# RunnableWithMessageHistory will feed the `history` as part of the `messages` list.
# `input_messages_key` will specify what the new message is.
# `history_messages_key` will specify where to put the history in the overall input.

# Let's adjust the agent_app's input structure to work well with RunnableWithMessageHistory
# LangGraph state typically expects `messages: List[BaseMessage]`.
# `RunnableWithMessageHistory` expects `input_messages_key` and `history_messages_key`.
# The `input` to `agent_app.invoke` will be a dictionary `{"messages": [HumanMessage(content=input_str)] + history_messages}`
# No, `RunnableWithMessageHistory` wraps a runnable and injects history.
# The internal state of the agent will have `messages`.
# The input to `chain_with_history` will be `{"input": user_message_str}`.
# `RunnableWithMessageHistory` will convert `user_message_str` to `HumanMessage` and append history.

# The overall agent chain that `RunnableWithMessageHistory` wraps
# It expects `input` and `history`.
# The LangGraph app itself operates on `messages` in its `AgentState`.
# We need a small adapter chain.

# Adapter from RunnableWithMessageHistory input format to LangGraph agent state format
def _format_for_agent_app(input_dict: dict) -> AgentState:
    # input_dict will have 'input' (current message) and 'history' (chat history)
    # The LangGraph agent_app expects AgentState(messages=[history + current_message])
    history_messages = input_dict.get("history", [])
    current_message = HumanMessage(content=input_dict["input"])
    return {"messages": history_messages + [current_message]} # Use dict for TypedDict compatible input

# The LangGraph agent app itself is the core runnable here.
# It returns the final AIMessage, so we can pass it directly.
final_chain = agent_app # Our compiled LangGraph agent

chain_with_history = RunnableWithMessageHistory(
    final_chain,
    get_session_history,
    input_messages_key="input", # Key for the current user message
    history_messages_key="history", # Key where history will be injected by RWMH
    # Ensure the history is passed correctly to the agent_app
    # The agent_app expects messages in its state.
    # We need to map `input` and `history` from RWMH into `messages` for agent_app.
    # Let's use `chain` to combine history and input before passing to agent_app.
).with_types(input_type={"input": str}, output_type=Any) # Define input/output types for clarity


# --- FastAPI App Setup ---
app = FastAPI(
    title="LangGraph Streaming Agent API",
    description="A FastAPI endpoint for a LangGraph agent with real-time streaming responses (SSE).",
    version="0.1.0",
)

# --- Pydantic Models for Request ---
class AgentRequest(BaseModel):
    """Request schema for the agent endpoint."""
    session_id: str
    message: str

# --- API Endpoint ---
@app.post("/agent_stream")
async def stream_agent_response(request: AgentRequest):
    """
    Streams responses from the LangGraph agent, including thoughts and actions,
    using Server-Sent Events (SSE).
    """
    async def event_generator():
        try:
            # We need to manually update history for streaming events from LangGraph,
            # as `RunnableWithMessageHistory.astream_events` is not directly supported.
            # Instead, we'll get the history, pass it to the agent_app, and then
            # manage history updates manually based on the agent's output.

            session_history = get_session_history(request.session_id)
            current_messages = session_history.messages + [HumanMessage(content=request.message)]

            # Astream events from the LangGraph agent
            # We directly stream the agent_app (the compiled graph)
            async for event in agent_app.astream_events(
                {"messages": current_messages}, # AgentState input
                config={"configurable": {"session_id": request.session_id}},
                version="v2" # Recommended for more consistent event structure
            ):
                event_name = event["event"]
                event_data = event["data"]
                
                # Filter events for what we want to send to the client
                # You can customize this to send more or less detail
                payload = None
                if event_name == "on_chat_model_start":
                    payload = {"type": "llm_start", "name": event["name"], "input": event_data.get("input")}
                elif event_name == "on_chat_model_stream":
                    chunk_content = event_data["chunk"].content
                    if chunk_content: # Only send if there's actual content
                        payload = {"type": "llm_stream", "content": chunk_content}
                elif event_name == "on_tool_start":
                    payload = {"type": "tool_start", "name": event["name"], "input": event_data.get("input")}
                elif event_name == "on_tool_end":
                    payload = {"type": "tool_end", "output": str(event_data.get("output"))}
                elif event_name == "on_chain_end" and event["name"] == "agent_app": # Final output of the agent
                    final_messages = event_data.get("output", {}).get("messages", [])
                    if final_messages:
                        final_response = final_messages[-1].content
                        payload = {"type": "final_answer", "content": final_response}
                        # Update the session history with the final exchange
                        session_history.add_user_message(request.message)
                        session_history.add_ai_message(final_response)

                if payload:
                    yield f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"
            
            # After the stream ends, ensure the connection is closed gracefully
            # A final event can signal the end of the response
            yield "event: stream_end\ndata: {}\n\n"

        except Exception as e:
            # Log the error for server-side debugging
            print(f"Error in streaming: {e}")
            # Send an error event to the client
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            # Ensure the connection is closed
            await asyncio.sleep(0.1) # Small delay before closing for client to receive error
            # Re-raise the exception or handle as needed for FastAPI error logging
            raise HTTPException(status_code=500, detail="Internal server error during streaming.")


    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "llm_provider": LLM_PROVIDER}

# --- How to run this app ---
# To run this file directly (for development with auto-reload):
# uvicorn day25-fastapi-streaming-agent:app --reload --host 0.0.0.0 --port 8000
#
# Open your browser to http://localhost:8000/docs for interactive API documentation.
# Test with a tool like curl (requires --no-buffer for real-time streaming):
# curl --no-buffer -X POST "http://localhost:8000/agent_stream" -H "Content-Type: application/json" -d '{"session_id": "user123", "message": "What is the current time?"}'
# curl --no-buffer -X POST "http://localhost:8000/agent_stream" -H "Content-Type: application/json" -d '{"session_id": "user123", "message": "Tell me a joke."}'

# Example of a simple HTML client (save as `index.html` and open in browser):
"""
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Streaming Agent</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .chat-container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 15px; border-radius: 8px; }
        .message { margin-bottom: 10px; padding: 8px; border-radius: 5px; }
        .user-message { background-color: #e0f7fa; text-align: right; }
        .ai-message { background-color: #f1f8e9; text-align: left; }
        #output { border: 1px solid #eee; padding: 10px; min-height: 150px; overflow-y: auto; background-color: #f9f9f9; }
        #controls { margin-top: 15px; }
        textarea { width: calc(100% - 20px); padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ddd; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .event-log { font-family: monospace; font-size: 0.8em; color: #555; background-color: #f0f0f0; padding: 5px; margin-top: 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>LangGraph Streaming Agent</h1>
        <p>Ask a question, and see the agent's thoughts and actions in real-time!</p>
        <div id="output"></div>
        <div id="controls">
            <input type="text" id="sessionId" value="user_session_1" placeholder="Enter Session ID">
            <textarea id="userInput" placeholder="Type your message..."></textarea>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const outputDiv = document.getElementById('output');
        const userInput = document.getElementById('userInput');
        const sessionIdInput = document.getElementById('sessionId');
        let currentEventSource = null;
        let aiMessageBuffer = ''; // Buffer for accumulating LLM stream chunks

        function appendMessage(role, content) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}-message`;
            msgDiv.innerHTML = content; // Use innerHTML to allow for formatting
            outputDiv.appendChild(msgDiv);
            outputDiv.scrollTop = outputDiv.scrollHeight; // Auto-scroll
        }

        function appendEventLog(message) {
            const logDiv = document.createElement('div');
            logDiv.className = 'event-log';
            logDiv.textContent = message;
            outputDiv.appendChild(logDiv);
            outputDiv.scrollTop = outputDiv.scrollHeight;
        }

        function sendMessage() {
            const message = userInput.value;
            const sessionId = sessionIdInput.value;
            if (!message.trim() || !sessionId.trim()) {
                alert("Please enter a message and a Session ID.");
                return;
            }

            appendMessage('user', message);
            userInput.value = '';
            aiMessageBuffer = ''; // Reset buffer for new message

            if (currentEventSource) {
                currentEventSource.close(); // Close any existing connection
            }

            // Using fetch with EventSource is cleaner for POST requests
            fetch('http://localhost:8000/agent_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let processedMessageElement = null; // Element to update for AI's final message

                async function processStream() {
                    let buffer = '';
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            console.log('Stream complete');
                            if (processedMessageElement) {
                                processedMessageElement.innerHTML = aiMessageBuffer; // Final update
                            }
                            break;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        let lines = buffer.split('\n');
                        buffer = lines.pop(); // Keep incomplete last line in buffer

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    if (data.type === 'llm_stream') {
                                        if (!processedMessageElement) {
                                            processedMessageElement = document.createElement('div');
                                            processedMessageElement.className = 'message ai-message';
                                            outputDiv.appendChild(processedMessageElement);
                                            outputDiv.scrollTop = outputDiv.scrollHeight;
                                        }
                                        aiMessageBuffer += data.content;
                                        processedMessageElement.innerHTML = aiMessageBuffer; // Update in place
                                    } else if (data.type === 'final_answer') {
                                        // This is redundant if llm_stream already built the answer,
                                        // but good for explicit finalization or if no llm_stream events occurred.
                                        if (!processedMessageElement) {
                                            appendMessage('ai', data.content);
                                        } else {
                                            processedMessageElement.innerHTML = data.content;
                                        }
                                        appendEventLog(`Agent Finished: ${data.content.substring(0, 50)}...`);
                                        aiMessageBuffer = ''; // Clear buffer
                                    } else if (data.type === 'llm_start') {
                                        appendEventLog(`LLM Thinking (Node: ${data.name})...`);
                                    } else if (data.type === 'tool_start') {
                                        appendEventLog(`Calling Tool: ${data.name} with input: ${JSON.stringify(data.input)}`);
                                    } else if (data.type === 'tool_end') {
                                        appendEventLog(`Tool Output: ${data.output.substring(0, 100)}...`);
                                    } else if (data.type === 'error') {
                                        appendEventLog(`ERROR: ${data.error}`);
                                    } else if (data.type === 'stream_end') {
                                        appendEventLog('Stream closed by server.');
                                    } else {
                                        appendEventLog(`Received event: ${JSON.stringify(data)}`);
                                    }
                                } catch (e) {
                                    console.error('Error parsing JSON:', e, line);
                                    appendEventLog(`Malformed event: ${line}`);
                                }
                            } else if (line.startsWith('event: ')) {
                                // This block could be used to directly read event types if 'data:' isn't always present
                                // but our JSON payload includes 'type' key, so it's less critical.
                            }
                        }
                    }
                }
                processStream().catch(error => {
                    console.error('Stream processing error:', error);
                    appendEventLog(`Stream processing error: ${error.message}`);
                });
            })
            .catch(error => {
                console.error('Fetch error:', error);
                appendEventLog(`Failed to connect to streaming endpoint: ${error.message}`);
                appendMessage('error', `Failed to get response: ${error.message}`);
            });
        }
    </script>
</body>
</html>
"""
