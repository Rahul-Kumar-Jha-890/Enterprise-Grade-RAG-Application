from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
    
#chat model does 2 jobs. writing alternate versions of user's qs and output final answer form retrieved chunks.
@lru_cache(maxsize=1)  #Prevents unnecessary repeated client initialization.
def get_chat_model() -> ChatOpenAI:  #LangChain class used to interact with OpenAI chat models.
    config.check_openai_apikey()

    return ChatOpenAI(   #ChatOpenAI parameters client parametres
        model = config.OPENAI_CHAT_MODEL,
        temperature = config.LLM_TEMP,
        api_key = config.OPENAI_API_KEY
    )

@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    config.check_openai_apikey

    return OpenAIEmbeddings(   #LangChain class used to generate text embeddings.
        model = config.OPENAI_EMBEDDING_MODEL,
        api_key = config.OPENAI_API_KEY
    )