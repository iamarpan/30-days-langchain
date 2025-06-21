import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json # For parsing editor's structured output

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower() # e.g., 'llama2', 'mistral'

# --- LLM Initialization ---
def initialize_llm(provider: str, model_name: str = None, temp: float = 0.7):
    """Initializes and returns the ChatLargeLanguageModel based on provider."""
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=model_name or "gpt-3.5-turbo", temperature=temp)
    elif provider == "ollama":
        try:
            llm_instance = ChatOllama(model=model_name or OLLAMA_MODEL_CHAT, temperature=temp)
            # Test connection to ensure Ollama is running and model is available
            llm_instance.invoke("Hello!", config={"stream": False})
            return llm_instance
        except Exception as e:
            print(f"Error connecting to Ollama LLM or model '{model_name or OLLAMA_MODEL_CHAT}' not found: {e}")
            print("Please ensure Ollama is running and the specified model is pulled (e.g., 'ollama pull llama2').")
            exit()
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'openai' or 'ollama'.")

# Initialize the chosen LLM
llm = initialize_llm(LLM_PROVIDER)
print(f"Using LLM: {LLM_PROVIDER} ({llm.model_name if hasattr(llm, 'model_name') else OLLAMA_MODEL_CHAT})\n")


# --- 1. Agent State Definition ---
class CollaborativeAgentState(TypedDict):
    """
    Represents the shared memory for the collaborative agents (Writer and Editor).
    """
    messages: Annotated[List[BaseMessage], add_messages] # Conversation history / internal messages log
    draft: str # The current content being written/revised by the Writer
    feedback: str # Editor's specific feedback for the Writer
    revision_count: int # Tracks how many times the draft has been revised
    status: str # Current state of the draft: "drafting", "reviewing", "revising", "completed", "error"
    topic: str # The original topic or task for the writer to address


# --- 2. Define Agent Nodes ---

# Node for the Writer Agent
def writer_node(state: CollaborativeAgentState) -> CollaborativeAgentState:
    """
    Writer Agent: Generates an initial draft or revises an existing one based on feedback.
    """
    print(f"\n--- Node: Writer Agent (Revision Count: {state['revision_count']}) ---")
    topic = state['topic']
    current_draft = state['draft']
    feedback = state['feedback']
    revision_count = state['revision_count']

    if revision_count == 0:
        # Initial draft generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a creative and engaging writer. Write a concise and clear paragraph about '{topic}'. Focus on capturing the reader's interest quickly."),
            ("human", f"Please write an initial draft about: {topic}")
        ])
        print("  Writing initial draft...")
    else:
        # Revise existing draft based on editor's feedback
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a meticulous writer. Revise the following draft about '{topic}' based on the specific feedback provided by the editor. Ensure you address the feedback comprehensively and improve the content's quality, clarity, and conciseness."),
            ("human", f"Current Draft:\n{current_draft}\n\nEditor's Feedback:\n{feedback}\n\nPlease provide a revised draft.")
        ])
        print(f"  Revising draft based on editor's feedback (Revision #{revision_count})...")

    response = llm.invoke(prompt)
    new_draft = response.content.strip()
    
    # Update the state with the new draft, increment revision count, and set status to reviewing
    print(f"  New Draft (excerpt): {new_draft[:150]}...") # Print a snippet of the new draft
    return {
        "draft": new_draft,
        "revision_count": revision_count + 1,
        "status": "reviewing",
        "messages": [AIMessage(content=f"Writer: Draft created/revised (Revision #{revision_count + 1}). Ready for editor.")]
    }

# Node for the Editor Agent
def editor_node(state: CollaborativeAgentState) -> CollaborativeAgentState:
    """
    Editor Agent: Reviews the draft, provides feedback, and decides if it's satisfactory or needs revision.
    """
    print(f"\n--- Node: Editor Agent ---")
    topic = state['topic']
    current_draft = state['draft']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a highly critical and constructive editor. Your primary task is to review the provided draft about '{topic}'.
        
        Evaluate the draft's:
        - Clarity: Is the message clear and easy to understand?
        - Conciseness: Is there any unnecessary jargon or repetition?
        - Engagement: Does it capture interest and convey the topic effectively?
        - Adherence to Topic: Does it fully address the stated topic?
        
        Based on your evaluation, decide if the draft is 'SATISFACTORY' (no more revisions are needed, it's ready) or 'NEEDS_REVISION'.
        
        If you decide 'NEEDS_REVISION', you MUST provide specific, actionable, and constructive feedback for the writer. Guide them on exactly what needs improvement.
        
        Output your decision as a JSON object with two keys: 'decision' and 'feedback'.
        
        Example for SATISFACTORY: 
        {{"decision": "SATISFACTORY", "feedback": "Excellent work! The draft is clear, concise, and engaging. No revisions needed."}}
        
        Example for NEEDS_REVISION:
        {{"decision": "NEEDS_REVISION", "feedback": "The introduction is too general. Please make the hook more specific to '{topic}' and add a clear thesis statement. Also, shorten the second sentence for better flow."}}
        """),
        ("human", f"Topic: {topic}\n\nDraft to review:\n{current_draft}")
    ])
    
    response = llm.invoke(prompt)
    editor_output_raw = response.content.strip()
    
    decision = "NEEDS_REVISION" # Default to revision in case of parsing errors or ambiguity
    feedback = "Editor could not parse response or provided generic feedback. Please revise for clarity and specific improvements."

    try:
        if editor_output_raw.startswith("```json"):
            # Attempt to strip markdown code block if present
            editor_output_raw = editor_output_raw[7:-3].strip()
        editor_json = json.loads(editor_output_raw)
        
        decision = editor_json.get("decision", "NEEDS_REVISION").upper()
        feedback = editor_json.get("feedback", feedback)
        
    except json.JSONDecodeError:
        print(f"  Warning: Editor LLM returned non-JSON. Raw output: {editor_output_raw[:100]}... Defaulting to 'NEEDS_REVISION'.")
    except Exception as e:
        print(f"  An error occurred during editor's output processing: {e}. Defaulting to 'NEEDS_REVISION'.")
    
    print(f"  Editor Decision: {decision}")
    print(f"  Editor Feedback: {feedback}")

    # Update the state with editor's feedback and decision status
    return {
        "feedback": feedback,
        "status": decision.lower(), # "satisfactory" or "needs_revision"
        "messages": [AIMessage(content=f"Editor: Decision: {decision}. Feedback: {feedback}")]
    }

# --- 3. Routing Logic ---

def route_editor_decision(state: CollaborativeAgentState) -> str:
    """
    Router function based on the Editor's decision.
    Decides whether to send the draft back to the Writer or end the workflow.
    """
    print(f"\n--- Router: Editor Decision (Current Status: '{state['status']}') ---")
    if state['status'] == "satisfactory":
        print("  Decision: Draft is SATISFACTORY. Workflow will END.")
        return END
    elif state['status'] == "needs_revision":
        print("  Decision: Draft NEEDS_REVISION. Routing back to 'writer_node' for revisions.")
        return "writer_node"
    else:
        # Fallback for any unexpected status
        print(f"  Decision: Unexpected status '{state['status']}'. Ending workflow for safety.")
        return END

# --- 4. Build the LangGraph Workflow ---
print("--- Building the Collaborative Multi-Agent Workflow (Writer-Editor) ---")
workflow = StateGraph(CollaborativeAgentState)

# Add nodes representing our agents
workflow.add_node("writer_node", writer_node)
workflow.add_node("editor_node", editor_node)

# Set the entry point of the workflow. We start with the writer.
workflow.set_entry_point("writer_node")

# Define the edges (transitions between nodes)
# After the writer produces a draft, it always goes to the editor for review
workflow.add_edge("writer_node", "editor_node")

# From the editor, the flow is conditional based on the editor's decision
workflow.add_conditional_edges(
    "editor_node",
    route_editor_decision, # Use our custom routing function
    {
        "writer_node": "writer_node", # If 'needs_revision', loop back to writer
        END: END # If 'satisfactory', end the workflow
    }
)

# Compile the graph into a runnable application
collaborative_app = workflow.compile()
print("Collaborative Multi-Agent workflow compiled successfully.\n")

# --- 5. Invoke the Workflow ---
print("--- Invoking the Writer-Editor Collaboration ---")

# Define the initial topic for the collaboration
initial_topic = "the importance of lifelong learning in the 21st century"
print(f"Starting collaboration on topic: '{initial_topic}'")

# Set the initial state for the workflow
initial_input = {
    "messages": [HumanMessage(content=f"Start writing about: {initial_topic}")],
    "draft": "",          # Initial empty draft
    "feedback": "",       # Initial empty feedback
    "revision_count": 0,  # Start with 0 revisions
    "status": "drafting", # Initial status
    "topic": initial_topic # The topic for the writer
}

try:
    # Invoke the workflow. Set a recursion_limit to prevent infinite loops
    # if the editor continuously requests revisions.
    final_state = collaborative_app.invoke(
        initial_input,
        config={"recursion_limit": 10}, # Allow up to 10 node transitions
        # verbose=True # Uncomment this line to see detailed LangGraph internal logs
    )
    print("\n" + "="*50)
    print("--- Final Collaboration State ---")
    print(f"Final Status: {final_state['status'].upper()}")
    print(f"Final Draft (after {final_state['revision_count'] - 1} revisions):")
    print(final_state['draft'])
    print(f"Last Feedback: {final_state['feedback']}")
    print("="*50 + "\n")

except Exception as e:
    print(f"\n!!! Workflow encountered an unexpected error: {e} !!!")
    if "recursion_limit" in str(e):
        print("The workflow likely hit the recursion limit. This means the editor kept requesting revisions without reaching a 'satisfactory' state within the limit.")
    print("Please check LLM responses and prompt engineering if this persists.")

print("Note: The quality and duration of revisions depend heavily on LLM capabilities, prompt clarity, and the complexity of the topic.")
