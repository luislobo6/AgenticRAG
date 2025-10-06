#environment variables
from dotenv import load_dotenv
import os
# imports necesary for the VectoStore and Embeddings (OpenAI)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# Imports for the retriever tool
from langchain.tools.retriever import create_retriever_tool


# Load environment variables from .env file
load_dotenv()
# get the directory where to persist the database
chroma_persist_directory = os.getenv("PERSIST_DIRECTORY", "chromadb")
retriever_name = os.getenv("RETRIEVER_NAME", "retrieve_RAG")
retriever_description = os.getenv("RETRIEVER_DESCRIPTION", "Search and return information about EOG performance.")

def create_vectorstore(documents, collection_name) -> Chroma:
    """
    Create a Chroma vector store from the given documents.
    Args:
        documents (list): A list of documents to be added to the vector store.
    Returns:
        Chroma: The created Chroma vector store.
        retriever: The retriever created from the vector store.
    """
    # create the embeddings function from OpenAI
    # embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # check if the persist directory exists, if not create it
    if not os.path.exists(chroma_persist_directory):
        os.makedirs(chroma_persist_directory)
    print(f"Chroma persist directory: {chroma_persist_directory}")
    
    #check if vector store already exists if so load it
    if os.path.exists(os.path.join(chroma_persist_directory, "index")):
        print("Loading existing Chroma vector store...")
        vectorstore = Chroma(
            persist_directory=chroma_persist_directory,
            embedding=embeddings,
            collection_name=collection_name
        )

        return vectorstore
    
    print("Creating new Chroma vector store...")
    # create the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_persist_directory,
        collection_name=collection_name
    )
    
    print(f"Chroma vector store created and persisted - collection_name: {collection_name}.")

    return vectorstore
