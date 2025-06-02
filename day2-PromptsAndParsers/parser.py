# Save this as day2-prompts-parsers.py
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Define the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

# --- Example 1: Basic Joke with StrOutputParser ---

print("--- Example 1: Simple Joke (String Output) ---")

# Define the prompt for a simple joke
joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty comedian specialized in short, clean jokes."),
    ("user", "Tell me a joke about {topic}."),
])

# Build the simple joke chain
simple_joke_chain = joke_prompt | llm | StrOutputParser()

# Invoke the chain
topic_1 = "cats"
response_1 = simple_joke_chain.invoke({"topic": topic_1})
print(f"Topic: {topic_1}")
print(f"Joke: {response_1}\n")

# --- Example 2: Structured Joke with PydanticOutputParser ---

print("--- Example 2: Structured Joke (Pydantic Output) ---")

# Define a Pydantic model for our structured joke output
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke.")
    punchline: str = Field(description="The punchline of the joke.")
    category: str = Field(description="The category of the joke (e.g., animal, food, tech).")

# Create a PydanticOutputParser from our Joke model
parser = PydanticOutputParser(pydantic_object=Joke)

# Define the prompt for a structured joke, including parser's format instructions
structured_joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty comedian. Generate a joke based on the user's topic "
               "and output it in the specified JSON format."),
    ("user", "Tell me a joke about {topic}.\n{format_instructions}"),
])

# Combine the prompt with format instructions from the parser
# This is a key step: parser.get_format_instructions() must be included in the prompt
final_structured_joke_prompt = structured_joke_prompt.partial(
    format_instructions=parser.get_format_instructions()
)

# Build the structured joke chain
structured_joke_chain = final_structured_joke_prompt | llm | parser

# Invoke the chain
topic_2 = "programming"
response_2 = structured_joke_chain.invoke({"topic": topic_2})

print(f"Topic: {topic_2}")
print(f"Joke Setup: {response_2.setup}")
print(f"Joke Punchline: {response_2.punchline}")
print(f"Joke Category: {response_2.category}")
print(f"Parsed Object Type: {type(response_2)}\n")

# Another structured example
topic_3 = "dogs"
response_3 = structured_joke_chain.invoke({"topic": topic_3})
print(f"Topic: {topic_3}")
print(f"Joke Setup: {response_3.setup}")
print(f"Joke Punchline: {response_3.punchline}")
print(f"Joke Category: {response_3.category}")
