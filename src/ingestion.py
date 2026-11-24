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

#This function loads multiple files from a folder and combines their extracted Document objects into one list, ready for chunking.
def load_documents_from_folder(folder_path : str |Path) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("Folder does not exist : %s", folder)
        return []

    all_documents : List[Document] = []

#iterdir() lists the immediate contents of the folder.
#sorted() processes them in alphabetical order.

    for file_path in sorted(folder.iterdir()):  #returns path of files to file_path variable. One file/iteration
        if file_path.is_file() and file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            try:   #If one file fails to load, the application logs the error and continues processing the remaining files.
                all_documents.extend(load_document(file_path)) 
            except Exception as err:
                logger.warning("Could not load %s %s ", file_path.name, err )       

    logger.info("Loaded %d document pages from %s : ", len(all_documents), folder)
    return all_documents



def clean_text(text : str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text) #Eg. secu-\nrity -> security

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def add_metadata(documents : List[Document], path :Path) -> List[Document]:
    for document in documents:
        document.metadata["source"] = str(path)
        document.metadata["filename"] = str(path.name)
        document.metadata["doctype"] = path.suffix.lower().lstrip(".")

        if "page" in document.metadata:   #in case of txt or word file, page no is not added automatically.
            document.metadata["page_no"] = int(document.metadata["page"]) + 1

    return documents

#attempts to identify the section heading of a document by examining its text.
def guess_section(text: str) -> str:
      
    for raw_line in text.split("\n")[:15]:  #examine the 1st 15 lines to find headings.
        line = raw_line.strip()  #remove whitespaces

        if not (3 < len(line) <= 70) or line.endswith((".", ",", ";", ":")):  #eliminate ordinary sentences
            continue
        if len(line.split()) > 8:
            continue

        looks_numbered = re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line) is not None
        if line.isupper() or line.istitle() or looks_numbered:
           
            return re.sub(r"^\d+(\.\d+)*[.)]?\s+", "", line).strip()
        
    return ""