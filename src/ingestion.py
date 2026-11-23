import logging   #Records warnings and errors.
import re   #Regular expressions, useful for cleaning text.
from pathlib import Path
from typing import List

from langchain_community.document_loaders import(PyPDFLoader, Docx2txtLoader, TextLoader)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

def load_document(file_path:str | Path) -> List[Document] :

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {extension}."
                          f"Supported file types ; {', '.join(SUPPORTED_EXTENSIONS)}")

    if not path.exists() :
        raise FileNotFoundError(f"File not found here : {path}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
    else :
        loader = TextLoader(str(path), encoding= "utf-8")

    documents = loader.load()  #extracts document content. Returns a list of LangChain Document objects.
    cleaned_documents = []

    for doc in documents:
        doc.page_content= clean_text(doc.page_content)   #cleans the extracted text.
        if doc.page_content.strip():   #checks whether anything remains after removing whitespace.
            cleaned_documents.append(doc) 

    if not cleaned_documents:  #empty document is discarded 
        logger.warning("No readable text found in %s. Skipping it", path.name)
        return []

    return add_metadata(cleaned_documents,path)

def clean_text():
    pass

def add_metadata():
    pass