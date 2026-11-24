from typing import Dict, List, Tuple
from langchain_core.documents import Document
import config
from src.ingestion import guess_section
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except OSError as err:
    raise ImportError(f"Error while import in chunking module : {err}")

#Prepares the splitting tool.
def make_splitter(chunk_size:int, chunk_overlap:int) -> RecursiveCharacterTextSplitter:   #It tries to preserve meaningful text boundaries.
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,   #Maximum target chunk length
        chunk_overlap=chunk_overlap,  #Characters shared between consecutive chunks
        length_function = len,   #measures chunk length in characters
        separators = ["\n\n", "\n", ".", "!", "?", ",", ";", "", " "],
    )

#	Splits input documents into parent and child chunks and assigns metadata/IDs.
def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    parent_splitter = make_splitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
    child_splitter = make_splitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP)

    parents : List[Document] = []   #Stores all parent Document objects.
    children : List[Document] = []   #Stores all child Document objects.

    parents_per_file : Dict[str,int] = {}  #Tracks how many parent chunks have been generated for each file

    for document in documents:
        filename = document.metadata.get("filename", "unknown")  #retrieve filename of 1st pdf from langchain document object.

        for parent_text in parent_splitter.split_text(document.page_content):  #split_text() takes the document's actual text and divides it into larger chunks.
            parent_number = parents_per_file.get(filename,0)  
            parents_per_file[filename] = parent_number + 1
            parent_id = f"{filename}::{parent_number}"

            parent_metadata = dict(document.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = guess_section(parent_text)

            parents.append(Document(page_content=parent_text, metadata = parent_metadata))

            for child_index, child_text in enumerate(child_splitter.split_text(parent_text)):  #splitting parent_text, not the original document.
                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = f"{parent_id}::{child_index}"
                children.append(Document(page_content=child_text, metadata = child_metadata))

    return parents, children

#This function creates a dictionary that allows you to retrieve a parent document using its ID.
#Helps retrieve the original parent using a child’s parent_id.
def build_parent_lookup(parents : List[Document]) ->Dict[str, Document]:
    return {parent.metadata["parent_id"]: parent for parent in parents }
 