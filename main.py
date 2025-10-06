#environment variables
from dotenv import load_dotenv
import os
#imports from the same project
from src.pdf_processor import load_pdf, split_documents
from src.chroma_db import create_vectorstore
from src.agent_tools import get_retriever_tool, simple_query, generate_query_or_respond


# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# Path to the documents specific for this demo
path=os.getenv("DOCUMENTS_PATH")
file_name=os.getenv("FILE_NAME")
file_path = os.path.join(path, file_name)
# get the directory where to persist the database
retriever_name = os.getenv("RETRIEVER_NAME", "retrieve_RAG")
retriever_description = os.getenv("RETRIEVER_DESCRIPTION", "Search and return information about EOG performance.")
simple_query_str = "what can you say about the culture?"



# run the code only if the script is executed directly (not imported as a module)
# for testing purposes
if __name__ == "__main__":
    
    print(file_path)
    # load the PDF document
    documents = load_pdf(file_path)

    # split the document into smaller chunks
    doc_splits = split_documents(documents)

    # use a vector store to create the embeddings and persist the database
    collection_name = "EOG_2024_Annual_Report"
    vectorstore = create_vectorstore(doc_splits, collection_name)
    
    # create the retriever tool
    retriever_tool = get_retriever_tool(vectorstore, retriever_name, retriever_description)

    # perform a simple query
    simple_response = simple_query(retriever_tool, simple_query_str)
    print(f"Response: {simple_response}")

    # message not related to RAG (hello!)
    input = {"messages": [{"role": "user", "content": "hello!"}]}
    generate_query_or_respond(input,retriever_tool)["messages"][-1].pretty_print()

    # query that use the RAG
    input = {
        "messages": [
            {
                "role": "user",
                "content": "What does Ezra Y. Yacob say about culture?",
            }
        ]
    }
    generate_query_or_respond(input, retriever_tool)["messages"][-1].pretty_print()