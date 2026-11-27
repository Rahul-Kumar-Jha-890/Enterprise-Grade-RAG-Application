import logging
from typing import List, Dict,Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
import config
from llm_client import get_chat_model

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a document search engine.\n"
            "Given a user question, write {num_variants} alternative versions "
            "of it that use different wording or synonyms but keep the same "
            "meaning.\n"
            "If conversation history is given, make each version standalone by "
            "replacing words like 'it' or 'that policy' with the real subject.\n"
            "Return one query per line. No numbering, no extra text.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nUser question: {question}",
        ),
    ]
)

def generate_query_variants(question : str, history:str="") -> List[str]:
    queries = [question]   #Creates a list containing the original question.

    try:
        chain = MULTI_QUERY_PROMPT | get_chat_model() | StrOutputParser()  #StrOutputParser() :Converts model output into a string
        response = chain.invoke(   #invoke() executes the chain by supplying values for the prompt variables.
            {
                "question" : question,
                "history" : history or "(None)",
                "num_variants" : config.NUM_QUERY_VARIANTS,
            }
        )

        for line in response.split("\n"):
           cleaned  =  line.strip(" -*123456789.").strip()   #Removes unwanted characters from the beginning and end of each line.

           if cleaned and cleaned.lower() != question.lower():
               queries.append(cleaned)

    except Exception as err:
        logger.warning("Multi query generation failed (%s). Using original query only.", err)

    return queries[:config.NUM_QUERY_VARIANTS + 1]  #Returns the original question plus the configured number of alternative queries at most


def build_keyword_retriever(children : List[Document]) -> BM25Retriever:
    pass

def search_vector_store(store: VectorStore, query:str, k:int) -> List[Document]:
    pass

def search_keywords(retriever: BM25Retriever, query : str)-> List[Document]:
    pass

def fuse_results(ranked_list: List[List[Document]], top_k:int) -> List[Document]:
    pass

def hybrid_search(queries: List[str], store : VectorStore, keywpord_retriever : BM25Retriever) ->List[Document]:
    pass

def record_parents(parent_lookup : Dict[str, Document], children : List[Document])-> List[Document]:
    pass