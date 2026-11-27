
#This is the main entry-point script for building your RAG pipeline's vector index.
import logging,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent)) 

import config
from src.indexing import build_index
from src.ingestion import load_documents_from_folder

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

def main() -> None:
    print(f"Reading documents from :  {config.DOCUMENTS_DIR}")   #Displays the folder from which documents will be loaded.

    documents = load_documents_from_folder(config)

    if not documents:
        print(f"No readable docs found. Add, pdf, docx file and try again.")
        return 

    try:
        stats = build_index(documents)
    except Exception as err:
        print(f"Build failed : {err}")
        return 

    print("\n Index built successfully. ")
    print(f"Documents loaded : {stats['documents']}")
    print(f"Parent Chunks : {stats['parents']}")
    print(f"Child chunks : {stats['children']}")
    print(f" Vector DB :  {config.VECTOR_DB}")
    print("\n Now run : Streamlit run app.py")

if __name__ == "__main__":
    main()

