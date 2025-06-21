import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama # Import ChatOllama for local models
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables from a .env file (for LLM_PROVIDER if used)
from dotenv import load_dotenv
load_dotenv()

# --- Configuration for Local LLMs ---
# IMPORTANT: Ensure Ollama is installed and running, and you've pulled the models.
# Example: ollama pull llama2
# Example: ollama pull mistral
# Example: ollama pull phi3:mini (smaller, might run faster on less powerful hardware)

WRITER_MODEL = os.getenv("WRITER_MODEL", "llama2") # Model for the writer agent
EDITOR_MODEL = os.getenv("EDITOR_MODEL", "mistral") # Model for the editor agent

# --- LLM Initialization (using ChatOllama for local models) ---
def initialize_local_llm(model_name: str, temp: float = 0.7):
    """Initializes and returns a ChatOllama instance for a local LLM."""
    try:
        llm_instance = ChatOllama(model=model_name, temperature=temp)
        # Test connection by making a dummy call
        llm_instance.invoke("Hello!", config={"stream": False})
        print(f"Successfully connected to Ollama model: {model_name}")
        return llm_instance
    except Exception as e:
        print(f"Error connecting to Ollama LLM '{model_name}' or model not found: {e}")
        print(f"Please ensure Ollama is running and you have pulled the model:")
        print(f"  ollama pull {model_name}")
        exit()

# Initialize our local LLMs for the writer and editor
writer_llm = initialize_local_llm(WRITER_MODEL)
editor_llm = initialize_local_llm(EDITOR_MODEL)

print(f"\nWriter Agent using local LLM: {WRITER_MODEL}")
print(f"Editor Agent using local LLM: {EDITOR_MODEL}\n")


# --- 1. Agent State Definition (reusing from Day 18) ---
class AgentState(TypedDict):
    """
    Represents the shared memory for our multi-agent workflow.
    - messages: Conversation history.
    - draft: The current draft of the content.
    - feedback: Feedback on the draft.
    - iterations: Counter for the number of editing rounds.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    draft: str
    feedback: str
    iterations: int


# --- 2. Define Agent Nodes (reusing from Day 18, but with local LLMs) ---

# Node 1: Writer Agent
def writer_node(state: AgentState) -> AgentState:
    """
    Generates an initial draft or revises an existing draft based on feedback.
    Uses the local writer_llm.
    """
    print("--- Node: Writer Agent ---")
    messages = state['messages']
    user_query = messages[0].content # Initial user query

    if state['draft']: # If there's an existing draft, it's a revision
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content writer. You will revise the existing draft based on the editor's feedback. Focus only on the content and quality, do not add meta-commentary like 'I have revised the draft based on the feedback.'"),
            ("human", f"Original request: {user_query}\n\nExisting draft:\n{state['draft']}\n\nEditor feedback:\n{state['feedback']}\n\nRevise the draft:")
        ])
        print(f"  Revising draft based on feedback (Iteration: {state['iterations']})...")
    else: # First time, generate initial draft
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content writer. Your task is to write a concise article draft based on the user's request. Keep it focused and to the point."),
            ("human", f"Write an article draft on: {user_query}")
        ])
        print("  Generating initial draft...")
    
    response = writer_llm.invoke(prompt)
    new_draft = response.content.strip()
    
    print(f"  Draft generated/revised (first 100 chars): {new_draft[:100]}...")
    return {"draft": new_draft, "messages": [AIMessage(content=f"Writer generated draft.")]}

# Node 2: Editor Agent
def editor_node(state: AgentState) -> AgentState:
    """
    Reviews the draft and provides constructive feedback.
    Uses the local editor_llm.
    """
    print("\n--- Node: Editor Agent ---")
    draft = state['draft']
    iterations = state['iterations']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a meticulous content editor. Review the following draft. If it meets high quality standards (clear, concise, accurate, directly addresses the prompt), say 'APPROVED'. Otherwise, provide specific, actionable feedback for revision. Focus on improvements, not just general praise. Limit your feedback to 2-3 concise points."),
        ("human", f"Here is the draft:\n{draft}\n\nProvide feedback or approve:")
    ])

    response = editor_llm.invoke(prompt)
    feedback_content = response.content.strip()

    if "APPROVED" in feedback_content.upper():
        print("  Draft APPROVED by Editor.")
        return {"feedback": feedback_content, "status": "approved", "messages": [AIMessage(content=f"Editor approved the draft.")]}
    else:
        print(f"  Editor provided feedback: {feedback_content[:100]}...")
        # Increment iterations and set status for revision
        return {"feedback": feedback_content, "iterations": iterations + 1, "status": "needs_revision", "messages": [AIMessage(content=f"Editor provided feedback for revision.")]}


# --- 3. Define Graph Structure and Conditional Logic ---

# Define the function that determines the next step based on the editor's output
def should_continue(state: AgentState) -> str:
    """
    Decides whether the workflow should continue (needs revision) or end (approved).
    """
    if state["status"] == "approved":
        print("\n--- Router: Draft Approved. Ending Workflow. ---")
        return "end"
    elif state["iterations"] >= 3: # Max 3 revision cycles
        print("\n--- Router: Max Iterations Reached. Ending Workflow. ---")
        return "end"
    else:
        print("\n--- Router: Draft Needs Revision. Looping back to Writer. ---")
        return "continue"


# --- 4. Build the LangGraph Workflow ---
print("--- Building the Local LLM Agent Workflow ---")

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

# Set entry point
workflow.set_entry_point("writer")

# Define edges
workflow.add_edge("writer", "editor")

# Define conditional edge from editor based on 'should_continue'
workflow.add_conditional_edges(
    "editor",
    should_continue,
    {
        "continue": "writer",  # If needs revision, go back to writer
        "end": END             # If approved or max iterations, end the graph
    }
)

# Compile the graph
local_llm_app = workflow.compile()
print("Local LLM Agent workflow compiled successfully.\n")


# --- 5. Run the Workflow with Local LLMs ---
print("--- Running Multi-Agent Workflow with Local LLMs ---")

user_input = "Write a short article about the benefits of local LLMs for developers."

print(f"USER REQUEST: {user_input}\n")

# Initial state for the workflow
initial_state = {
    "messages": [HumanMessage(content=user_input)],
    "draft": "",
    "feedback": "",
    "iterations": 0,
    "status": "" # Will be set by nodes
}

# Run the graph
final_state = local_llm_app.invoke(initial_state)

print("\n" + "="*60)
print("FINAL ARTICLE DRAFT:")
print(final_state['draft'])
print("="*60)
print(f"FINAL STATUS: {final_state['status'].upper()}")
print(f"TOTAL ITERATIONS: {final_state['iterations']}")

# Optional: Print all messages to see the conversation flow
# print("\n--- Full Conversation History ---")
# for msg in final_state['messages']:
#     print(f"{msg.type.upper()}: {msg.content}")

print("\n--- Local LLM Agent Workflow Complete ---")
print(f"Observe the power of LangGraph running entirely on local models ({WRITER_MODEL} & {EDITOR_MODEL}).")
