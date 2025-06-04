import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Explicitly import Document for clarity

file_path = "sample_document.txt"
with open(file_path, "w") as f:
    f.write(document_content)

print(f"Created '{file_path}' for demonstration.\n")

# --- Step 1: Load the Document ---
print("--- Loading Document ---")
loader = TextLoader(file_path)
documents = loader.load() # This returns a list of Document objects (usually one per file)

# Check if any documents were loaded
if documents:
    original_document = documents[0]
    print(f"Original Document Page Content (first 200 chars):\n{original_document.page_content[:200]}...\n")
    print(f"Original Document Metadata: {original_document.metadata}\n")
else:
    print("No documents loaded. Exiting.")
    exit()

# --- Step 2: Split the Document into Chunks ---
print("--- Splitting Document ---")

# Initialize the text splitter
# chunk_size: max size of each chunk
# chunk_overlap: number of characters to overlap between chunks (helps maintain context)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len, # Uses character count for length. Can be a token counter.
    add_start_index=True # Adds a 'start_index' to metadata indicating where the chunk began in the original text
)

# Split the document
chunks = text_splitter.split_documents(documents)

print(f"Original document split into {len(chunks)} chunks.\n")

# --- Step 3: Print the Chunks and their Metadata ---
print("--- Reviewing Chunks ---")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Content (first 100 chars): {chunk.page_content[:100]}...")
    print(f"  Length: {len(chunk.page_content)} characters")
    print(f"  Metadata: {chunk.metadata}")
    print("-" * 30)

# Optional: Clean up the dummy file
# os.remove(file_path)
# print(f"\nCleaned up '{file_path}'.")
