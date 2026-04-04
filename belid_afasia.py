print("Step 1: Loading documents...")
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import os

# Load documents
docs = []

folder_path = "rag_docs"

for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        file_path = os.path.join(folder_path, file)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append(Document(page_content=text))

print("Total documents loaded:", len(docs))

print("Step 2: Splitting documents...")
# Split documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = text_splitter.split_documents(docs)

print("Step 3: Creating embeddings...")
# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("Step 4: Building FAISS index...")
# Create FAISS
db = FAISS.from_documents(docs_split, embeddings)

# Save index
db.save_local("faiss_index")

print("FAISS index created successfully")