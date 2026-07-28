from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import (StateGraph,END)


#vector database:
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

#Graph State:
class RAGState(TypedDict):
    question:str
    context:str
    answer:str


#retrieve node:
def retrieve_node(state:RAGState):
    question = state["question"]
    docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    return {
        "context":context
    }


def generate_node(state:RAGState):
    template = """ You are a helpful AI assistant.Use the context below to answer.If the answer is not available in the context,use your own knowledge.
    Context: {context}
    Question:{question}
    Answer:
"""
    prompt = ChatPromptTemplate.from_template(
        template
    )
    chain = (prompt|llm|StrOutputParser())

    answer = chain.invoke(
        {
        "context":state["context"],
        "question":state["question"]
        }
    )
    return {
        "answer":answer
    }


graph = StateGraph(
    RAGState
)


# Nodes
graph.add_node(
    "retrieve",
    retrieve_node
)


graph.add_node(
    "generate",
    generate_node

)


# Flow
graph.set_entry_point(
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    END
)


# Compile
app = graph.compile()


# CHAT LOOP
print(
    "\n RAG READY"
)


while True:
    question = input(
        "\nAsk: "
    )
    if question.lower()=="exit":
        break

    result = app.invoke(
        {
        "question":question,
        "context":"",
        "answer":""
        }

    )

    print(
        "\nAnswer:\n"
    )
    print(
        result["answer"]
    )