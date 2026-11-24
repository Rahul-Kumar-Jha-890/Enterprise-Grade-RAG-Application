import json
import logging
import shutil
import config
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from llm_client import get_embedding_model
from src.chunking import split_documents

logger = logging.getLogger(__name__)

#converts langchain documents to dicts so they can be saved. 
def documents_to_dicts(documents: List[Document]) -> List[dict]:
    return [
        {"page-content" : doc.page_content, "metadata" :doc.metadata}
        for doc in documents
    ]

#converts dictionaries loaded from json back into langchain docs
def dicts_to_documents(items:List[dict]) -> List[Document]:
    return [
        Document(page_content=items["page_content"], metadata= item["metadata"])
        for item in items
    ]

def save_doc_store(parents:List[Document], children : List[Document]) -> None:
    pass

def create_vector_store(children: List[Document]) -> VectorStore:
    pass

def load_vector_store() -> VectorStore:
    pass

def fiass_class():
    pass

def pinecone_class():
    pass

def build_index(documents: List[Document]) -> Dict[str,int]:
    pass

def update_index(documents : List[Document]) -> Dict[str,int]:
    pass

def clear_index() ->None:
    pass

def clear_pinecone_namespace() ->None:
    pass

def index_exists() ->bool:
    pass

def index_summary() -> Dict[str,object]:
    pass