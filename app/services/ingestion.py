from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vector_store
from bs4 import BeautifulSoup as Soup

def ingest_website(url: str, max_depth: int = 2):
    # Load HTML
    loader = RecursiveUrlLoader(
        url=url, 
        max_depth=max_depth, 
        extractor=lambda x: Soup(x, "html.parser").get_text()
    )
    docs = loader.load()

    # Transform HTML to text
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs_transformed)

    # Add to vector store
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)
