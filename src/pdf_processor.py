#environment variables
from dotenv import load_dotenv
import os
# import the loader for PDFs
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.document_loaders import PyPDFDirectoryLoader
# import the splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Path to the documents specific for this demo
path=os.getenv("DOCUMENTS_PATH")
file_name=os.getenv("FILE_NAME")
file_path = os.path.join(path, file_name)
#file_path = "../documents/2024-Annual-Report-Web-Ready.pdf"

def load_pdf(file_path) -> list[Document]:
    """
    Load a PDF file and return its documents.
    Args:
        file_path (str): The path to the PDF file.
    Returns:
        list: A list of documents loaded from the PDF.
    """

    # create the loader with page mode and extracting tables and images
    loader = PyMuPDFLoader(
        file_path = file_path,
        # headers = None
        # password = None,
        mode = "page",
        #pages_delimiter = "",
        extract_images = True, 
        # images_parser = TesseractBlobParser(),
        images_inner_format = "markdown-img", 
        extract_tables = "csv", 
        # extract_tables_settings = None,
    )

    # lazy load the document
    docs = []
    docs_lazy = loader.lazy_load()

    for doc in docs_lazy:
        docs.append(doc)

    # example of part of the loaded document with its metadata
    print(docs[0].page_content[:100])
    print(docs[0].metadata)

    return docs


def split_documents(docs) -> list:
    """Split the documents into smaller chunks and return them."""
    
    # list of documents in a form of list comprehensions for URLs
    # docs_list = [item for sublist in docs for item in sublist]
    docs_list = []
    for doc in docs:
        # Check if the item is a tuple and extract the first element if it is
        # PyMuPDFLoader returns a list of tuples (document,metadata)
        if isinstance(doc, tuple):
            docs_list.append(doc[0])
        else:
            docs_list.append(doc)

    # split the text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800, chunk_overlap=200
    )

    # different splits of the document
    doc_splits = text_splitter.split_documents(docs_list)

    # example of content of the splitted document with metadata
    print(doc_splits[0].page_content.strip())
    print(doc_splits[0].metadata)

    return doc_splits

# run the code only if the script is executed directly (not imported as a module)
# for testing purposes
if __name__ == "__main__":
    # load the PDF document
    documents = load_pdf(file_path)

    # split the document into smaller chunks
    doc_splits = split_documents(documents)