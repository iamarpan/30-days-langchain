**30DaysLangchain**

Demystifying the Modern LangChain Ecosystem – One Day at a Time
Welcome to the #30DaysOfLangChain challenge! This repository, 30DaysLangchain, is your go-to resource for diving deep into LangChain 0.3, LangGraph, and the latest best practices for building powerful Generative AI (GenAI) applications.

How to Navigate This Repository
Each day's content will be organized into its own dedicated folder, following the dayX-<topicName> format (e.g., day1-runnables-lcel, day2-prompts-parsers).

You can follow along by cloning this repository and navigating into the respective daily directories. Each day's folder will contain its specific code, explaining that day's concepts, and instructions on how to run the code.

Prerequisites
Python 3.9+
pip for package management
Access to an LLM provider (e.g., OpenAI API Key) or local LLM setup (e.g., Ollama).

General Setup Instructions
Clone the repository:

```bash
git clone https://github.com/your-username/30DaysLangchain.git
cd 30DaysLangchain
(Remember to replace your-username with your actual GitHub username!)
```

Create a virtual environment (recommended):

```bash

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install base dependencies (specific daily requirements might be in requirements.txt within each day's folder):
```

```bash
pip install langchain-community langchain-openai python-dotenv

# You might also install langchain-ollama, streamlit, fastapi etc. as needed for later days
Set up API Keys:
For security, create a file named .env in the root of your project (or in specific daily folders if mentioned) and add your API keys:

OPENAI_API_KEY="your_openai_api_key_here"
# Add other keys like HUGGINGFACEHUB_API_TOKEN, etc., as needed for later days
It's crucial to add .env to your project's root .gitignore file to prevent accidentally committing your API keys!
```
