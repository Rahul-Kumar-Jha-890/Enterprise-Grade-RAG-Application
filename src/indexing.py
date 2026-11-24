import json
import logging
import shutil
import config
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from llm_client import get_embedding_model
from src.chunking import split_documents


logger = logging.getLogger(__name__)

#converts langchain documents to dicts so they can be saved.Because dictionaries can be converted into JSON and saved to a file.
def documents_to_dicts(documents: List[Document]) -> List[dict]:
    return [
        {"page_content" : doc.page_content, "metadata" :doc.metadata}
        for doc in documents
    ]

#converts dictionaries loaded from json back into langchain docs because your RAG pipeline works with LangChain Documents.
def dicts_to_documents(items:List[dict]) -> List[Document]:
    return [
        Document(page_content=items["page_content"], metadata= item["metadata"])
        for item in items
    ]

#Saves parent and child documents into a JSON file.
def save_doc_store(parents:List[Document], children : List[Document]) -> None:
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    payload={   #Stores both parent and children inside one dictionary called payload.
        "parents" : documents_to_dicts(parents),
        "children" : documents_to_dicts(children),
    }

    #json.dumps() converts the dictionary into JSON text. write_text() saves that JSON text into a file.
    config.DOC_STORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

#Records how many parent and child documents were saved.
    logger.info(
        "Saved %d parents and %d children to %s",
        len(parents),
        len(children),
        config.DOC_STORE_PATH
    )

#Loads previously saved documents from the JSON file.
def load_doc_store() -> Tuple[List[Document], List[Document]]:
    if not config.DOC_STORE_PATH.exists():
        return [], []

#Reads the JSON file and converts its contents back into a Python dictionary.
    payload = json.loads(config.DOC_STORE_PATH.read_text(encoding="utf-8"))
    parents = dicts_to_documents(payload.get("parents", []))
    children = dicts_to_documents(payload.get("children", []))

    return parents, children
    
#Creates the vector database by converting child chunks into embeddings and storing them.
def create_vector_store(children: List[Document]) -> VectorStore:   #The vector store will contain embeddings of those chunks.
    if not children:
        raise ValueError("There are no chunks to index. Upload a file before building index.")

    embedding = get_embedding_model()

    if config.VECTOR_DB == "faiss":
        store = faiss_class().from_documents(children,embedding)  #from_documents(): Generates embeddings and creates the vector store.
        config.FAISS_DIR.mkdir(parents=True, exist_ok=True)  #creates the directory where fiass will be stored
        store.save_local(str(config.FIASS_DIR))  #Saves the FAISS index locally on your computer.
        return store
    
    if config.VECTOR_DB == "pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = pinecone_class()
        return PineconeVectorStore.from_documents(
            children,
            embedding,
            index_name = config.PINECONE_INDEX_NAME,
            namespace = config.PINECONE_NAMESPACE   #namespace Separates records within a Pinecone index
        )

    raise ValueError(f"Unknown vector DB {config.VECTOR_DB}. Use either fiass or pinecone.")

#Load an already-created vector database so your RAG application can perform similarity searches without creating embeddings and indexing everything again.
def load_vector_store() -> VectorStore:
    embeddings = get_embedding_model()

    if config.VECTOR_DB=="faiss":
        if not (config.FIASS_DIR/"index.faiss").exists():
            raise FileNotFoundError("No file index found. Click 'Vector DB' in sidebar or run python scripts/build_index.py")

        return faiss_class().load_local(
            str(config.FIASS_DIR),
            embeddings,
            allow_dangerous_deserialization = True,
        )

    if config.VECTOR_DB=="pinecone":
        config.check_pinecone_config()
        PineconeVectorStore = pinecone_class()

        return PineconeVectorStore.from_existing_index(
            index_name = config.PINECONE_INDEX_NAME,
            embedding = embeddings,
            namespace = config.PINECONE_NAMESPACE
        )
    raise ValueError(f"Unknown Vector DB {config.VECTOR_DB}. Use faiss or pinecone.")

def faiss_class():
    from langchain_community.vectorstores import FAISS
    return FAISS

def pinecone_class():
    try:
        from langchain_pinecone import PineconeVectorStore
    except ImportError as err:
        ("VECTOR DB==pinecone needs the pinecone package. pip install it")

#build_index() coordinates the entire indexing process, from document splitting to vector database creation and document storage.
def build_index(documents: List[Document]) -> Dict[str,int]:
    if not documents:
        raise ValueError("No docs found.")
    parents,children = split_documents(documents)

    clear_index()
    create_vector_store(children)
    save_doc_store(parents ,children)

    return {
        "documents" : len(documents),
        "parents" : len(parents),
        "children" : len(children)
    }

#implements incremental indexing i
def update_index(documents : List[Document]) -> Dict[str,int]:
    existing_children, existing_parents = load_doc_store()

    if not existing_children:
        return build_index(documents)

    new_parents, new_children = split_documents(documents)

    known_chunk_ids = {doc.metadata["chunk_id"] for doc in existing_children}  #Python set comprehension.
    known_parent_ids = {doc.metadata["parent_id"] for doc in existing_parents}

    children_to_add = [
        doc for doc in new_children if doc.metadata["chunk_id"]  not in known_chunk_ids
    ]
    parent_to_add = [
            doc for doc in new_parents if doc.metadata["parent_id"]  not in known_parent_ids
        ]

    if not children_to_add:
        return{"added children" : 0, "skipped_children" : len(new_children)}

    store = load_vector_store()   #Loads the existing FAISS or Pinecone vector database.

    if config.VECTOR_DB=="pinecone":
        store.add_documents(children_to_add, namespace = config.PINECONE_NAMESPACE)
    else:
        store.add_documents(children_to_add)     #adds new documents and their embeddings.
        store.save_local(config.FIASS_DIR)

    save_doc_store(   #Combines old and new documents.
        existing_parents + parent_to_add,
        existing_children + children_to_add,
    )

    return {
        "added_children" : len(children_to_add),
        "skipped_children" : len(new_children) - len(children_to_add),
    }

#Removes the old index so a fresh one can be created.
def clear_index() ->None:
    if config.FIASS_DIR.exists():
        shutil.rmtree(config.FIASS_DIR)   #deletes the entire directory and everything inside it.

    if config.DOC_STORE_PATH.exists():  #Checks and deletes the JSON document store exists.
        config.DOC_STORE_PATH.unlink()

    if config.VECTOR_DB == "pinecone":   
        clear_pinecone_namespace()

    logger.info("index cleared.")


#Deletes all vectors belonging to a particular Pinecone namespace.
def clear_pinecone_namespace() ->None:
    try:
        from pinecone import Pinecone
        pinecone = Pinecone(api_key = config.PINECONE_API_KEY)
        index = Pinecone.Index(config.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)   #deletes all vectors in the specified namespace

    except Exception as err:
        logger.warning("Could not clear pinecone namespace %s: ", err)

def index_exists() ->bool:
    if not config.DOC_STORE_PATH.exists():
        return False

    if config.VECTOR_DB=="faiss":
        return(config.FIASS_DIR/"index.faiss").exists()

    return True

#Returns indexing statistics
def index_summary() -> Dict[str,object]:
    parents, children = load_doc_store()

    filenames = sorted({doc.metadata.get("filename","?") for doc in parents})
    return {"parents" : len(parents), "children" : len(children), "files" : filenames, }
