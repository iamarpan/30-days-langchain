import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Configuration for LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # 'openai' or 'ollama'
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "llama2").lower()
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo")

# --- Initialize LLM ---
def get_llm():
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider.")
        return ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7) # Higher temp for creativity
    elif LLM_PROVIDER == "ollama":
        try:
            llm = ChatOllama(model=OLLAMA_MODEL_CHAT, temperature=0.7) # Higher temp for creativity
            llm.invoke("test", config={"stream": False})
            return llm
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama chat LLM '{OLLAMA_MODEL_CHAT}': {e}") from e
    else:
        raise ValueError(f"Invalid LLM provider: {LLM_PROVIDER}. Must be 'openai' or 'ollama'.")

llm = get_llm()

# --- Define the Agent State ---
# This state will be passed between agents and updated by them.
class AgentState(TypedDict):
    """
    Represents the state of our multi-agent workflow.
    - initial_topic: The user's initial request/topic.
    - ideas: A list of generated ideas (strings).
    - critiques: A list of critiques for the current ideas.
    - iteration: Current iteration count for refinement.
    - max_iterations: Maximum allowed iterations for refinement.
    - messages: Langchain messages for conversation history (optional, but good for context)
    """
    initial_topic: str
    ideas: Annotated[List[str], operator.add]
    critiques: Annotated[List[str], operator.add]
    iteration: int
    max_iterations: int
    messages: Annotated[List[BaseMessage], operator.add]


# --- Agent Node Functions ---

def brainstorm_ideas(state: AgentState):
    """
    Brainstormer Agent: Generates initial ideas.
    """
    print("---BRAINSTORMING IDEAS---")
    topic = state["initial_topic"]
    current_ideas = state.get("ideas", [])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a creative brainstorming expert. Generate 3-5 distinct and innovative ideas based on the user's topic. Each idea should be a concise sentence or two. If there are existing ideas, try to generate new ones or variations, but don't just repeat them."),
        ("human", f"Topic: {topic}\nExisting ideas (avoid repeating): {current_ideas if current_ideas else 'None'}\nGenerate new ideas:")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    new_ideas_str = chain.invoke({"topic": topic, "current_ideas": current_ideas})
    
    # Simple parsing: split by lines, filter empty ones
    new_ideas = [idea.strip() for idea in new_ideas_str.split('\n') if idea.strip()]
    
    print(f"Generated Ideas: {new_ideas}")
    return {"ideas": new_ideas, "messages": [AIMessage(content=f"Generated initial ideas for '{topic}'.")]}

def critique_ideas(state: AgentState):
    """
    Critique Agent: Evaluates the generated ideas for flaws or areas of improvement.
    """
    print("---CRITIQUING IDEAS---")
    topic = state["initial_topic"]
    ideas = state["ideas"]
    
    # Combine all current ideas into a single string for critiquing
    ideas_text = "\n".join([f"- {idea}" for idea in ideas])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a critical thinking agent. Review the following ideas for a given topic. "
                   "Identify weaknesses, potential issues, missing aspects, or areas for improvement. "
                   "Provide constructive feedback for each idea or overall. Be specific and concise. "
                   "If ideas are generally good, suggest ways to make them even better or more unique."),
        ("human", f"Topic: {topic}\nIdeas to critique:\n{ideas_text}\n\nProvide your critique:")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    critique_str = chain.invoke({"topic": topic, "ideas_text": ideas_text})
    
    print(f"Critique: {critique_str}")
    return {"critiques": [critique_str], "messages": [AIMessage(content=f"Provided critique for ideas.")]}

def refine_ideas(state: AgentState):
    """
    Refiner Agent: Incorporates critiques to improve the ideas.
    """
    print("---REFINING IDEAS---")
    topic = state["initial_topic"]
    ideas = state["ideas"]
    critiques = state["critiques"]
    iteration = state["iteration"] + 1 # Increment iteration count

    # Combine all current ideas and critiques
    ideas_text = "\n".join([f"- {idea}" for idea in ideas])
    critiques_text = "\n".join([f"- {critique}" for critique in critiques])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"You are an idea refiner. Your goal is to improve the provided ideas based on the critiques. "
                   f"Generate 3-5 refined and enhanced ideas for the topic, taking into account all feedback. "
                   f"Focus on addressing the weaknesses and incorporating suggestions. "
                   f"Current Iteration: {iteration}"),
        ("human", f"Topic: {topic}\nOriginal Ideas:\n{ideas_text}\n\nCritiques:\n{critiques_text}\n\nGenerate Refined Ideas (3-5 concise sentences):")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    refined_ideas_str = chain.invoke({"topic": topic, "ideas_text": ideas_text, "critiques_text": critiques_text, "iteration": iteration})

    refined_ideas = [idea.strip() for idea in refined_ideas_str.split('\n') if idea.strip()]
    
    print(f"Refined Ideas: {refined_ideas}")
    # Clear critiques for the next round, and update ideas and iteration count
    return {"ideas": refined_ideas, "critiques": [], "iteration": iteration, "messages": [AIMessage(content=f"Refined ideas based on critique.")]}

def decide_next_step(state: AgentState) -> str:
    """
    Decides the next step in the workflow based on iteration count.
    """
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    
    print(f"---DECIDING NEXT STEP (Iteration: {iteration}/{max_iterations})---")
    if iteration >= max_iterations:
        print("Max iterations reached. Ending.")
        return "end"
    else:
        print("Continuing to critique.")
        return "critique"


# --- Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)

# Add nodes for each agent/step
workflow.add_node("brainstorm", brainstorm_ideas)
workflow.add_node("critique", critique_ideas)
workflow.add_node("refine", refine_ideas)

# Set the entry point
workflow.set_entry_point("brainstorm")

# Define edges
workflow.add_edge("brainstorm", "critique") # After brainstorming, always critique
workflow.add_edge("critique", "refine")    # After critique, always refine

# Define conditional edge for refinement loop
# After refinement, decide whether to loop back for more critique or end
workflow.add_conditional_edges(
    "refine",
    decide_next_step, # Function to determine next node
    {
        "critique": "critique", # Loop back to critique for another round
        "end": END              # End the graph
    }
)

# Compile the graph
app = workflow.compile()

# --- Run the Multi-Agent Team ---
if __name__ == "__main__":
    initial_topic = "Marketing campaign ideas for a new eco-friendly smart home device."
    max_iterations = 2 # Number of refinement cycles

    print(f"\n--- Starting Multi-Agent Idea Generation for: '{initial_topic}' ---")

    # The initial state passed to the graph
    inputs = {
        "initial_topic": initial_topic,
        "ideas": [],
        "critiques": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "messages": [HumanMessage(content=f"Generate ideas for: {initial_topic}")]
    }

    # Stream the output for better visibility of steps
    for s in app.stream(inputs):
        print(s)
        print("------")

    print("\n--- Final Ideas After Multi-Agent Collaboration ---")
    final_state = app.invoke(inputs) # Get the final state after execution
    for i, idea in enumerate(final_state["ideas"]):
        print(f"Idea {i+1}: {idea}")
    print(f"\nTotal Iterations: {final_state['iteration']}")
