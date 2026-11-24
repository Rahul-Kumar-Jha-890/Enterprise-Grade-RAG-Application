#The purpose of the config file is to centralize all the configurable settings of your RAG application in one 
# place, instead of hardcoding them throughout the codebase.
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   #Loads environment variables

def get_int(name:str, default : int):
    raw_value = os.getenv(name)   #Reads the value associated with name.
    if raw_value is None or raw_value.strip() == "":  #if .env variable contains on;y whitespaces.
        return default
    try:
        return int(raw_value)   #converts str value from .env file to int
    except ValueError:
        print(f"[config] {name}={raw_value} is not a number. Using the value = {default}")
        return default

PROJECT_ROOT = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_ROOT/ "data" / "documents"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embedding"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"
FIASS_DIR = VECTORSTORE_DIR / "fiass"
DOC_STORE_PATH = VECTORSTORE_DIR/"doc_store.json"

#OPEN_AI MODELS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
OPENAI_EMBEDDING_MODEL= os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "gpt-4o-mini")
LLM_TEMP = 0.0   #Randomness in LLM 0

#faiss  # fast similarity search and clustering of dense vectors
#Pinecone is a full vector database(cloud), while Faiss is a vector search library.
VECTOR_DB= os.getenv("VECTOR_DB", "faiss")
PINECONE_API_KEY =  os.getenv("PINECONE_API_KEY", "")  #Managed cloud vector database
PINECONE_INDEX_NAME= os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

#CHUNKING

PARENT_CHUNK_SIZE = get_int("PARENT_CHUNK_SIZE", 1600)  #get_int returns a integer
PARENT_CHUNK_OVERLAP = get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = get_int("CHILD_CHUNK_SIZE", 600)
CHILD_CHUNK_OVERLAP = get_int("CHILD_CHUNK_OVERLAP", 56)

#RETRIEVAl pipleline 
NUM_QUERY_VARIANTS = get_int("NUM_QUERY_VARIANTS",3)  #Controls how many alternative versions of the user's query are generated.
VECTOR_TOP_K= get_int("VECTOR_TOP_K",6)  #Retrieve 6 documents using vector similarity search
BM25_TOP_K= get_int("BM25_TOP_K",6)      #Retrieve 6 documents using keyword search
HYBRID_TOP_K = get_int("HYBRID_TOP_K", 10)  #Keep 10 docs after combining both results.
RERANK_TOP_K= get_int("RERANK_TOP_K",3)  #After retrieving up to 10 documents, the reranker selects the top 3 most relevant documents.
MIN_RERANK_SCORE= float(os.getenv("MIN_RERANK_SCORE", "-4"))  #Defines a minimum relevance score threshold.
MAX_CONTEXT_CHARS= get_int("MAX_CONTEXT_CHARS",8000)   #Limits the total context supplied to the LLM
MEMORY_WINDOW = get_int("MEMORY_WINDOW",4)  #Controls how many previous conversation interactions are retained in conversational memory,

def check_openai_apikey() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("API KEY not set")

def check_pinecone_config() ->None:
    if not PINECONE_API_KEY:
        raise ValueError("VECTOR_DB = PINECONE but PINECONE_API_KEY is empty.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("VECTOR_DB = PINECONE but PINECONE_INDEX_NAME is empty.")
    
def describe_settings():
    return{
        "Chat Model" : OPENAI_CHAT_MODEL,
        "Embedding Model" : OPENAI_EMBEDDING_MODEL,
        "Vector DB" : VECTOR_DB,
        "Parent Chunk Size" : PARENT_CHUNK_SIZE,
        " Child Chunk Size" :  CHILD_CHUNK_SIZE
    }