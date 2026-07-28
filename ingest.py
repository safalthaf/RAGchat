import os

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCUMENT_FOLDER = "documents"
DB_DIRECTORY = "chroma_db"


def load_documents(folder):
    docs = []

    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(path)

                elif file.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(path)

                elif file.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")

                else:
                    continue

                docs.extend(loader.load())

            except Exception as e:
                print(f"Skipped {file}: {e}")

    return docs


documents = load_documents(DOCUMENT_FOLDER)

print(f"Loaded Documents : {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks : {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

if os.path.exists(DB_DIRECTORY):
    import shutil
    shutil.rmtree(DB_DIRECTORY)

db = Chroma(
    persist_directory=DB_DIRECTORY,
    embedding_function=embeddings
)

db.add_documents(chunks)

print("\nVector Database Created Successfully!")
print(f"Saved to : {DB_DIRECTORY}")