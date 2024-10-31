import nest_asyncio

nest_asyncio.apply()
from llama_parse import LlamaParse
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


Settings.llm = Ollama(model="mistral", request_timeout=60.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

parser = LlamaParse(
    api_key="llx-",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# sync

DATA_PATH = "./manual-data"


def load_document_in_index():
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        "./pdf", file_extractor=file_extractor
    ).load_data()

    # documents = SimpleDirectoryReader("pdf").load_data()

    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("manual-data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    return index.as_query_engine()


def load_index():
    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("manual-data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index.as_query_engine()


load_document_in_index()

# index=load_index()
# # retriever = index.as_retriever(similarity_top_k=50)
# query=" What do you know about 'TH-OR27'?"
# messages = [
#         {
#             "role": "system",
#             "content": (
#                 "Provide a thorough answer to the user's query. If the precise answer is unavailable, "
#                 "supply closely related information. The response must be detailed enough to explain well "
#                 "Specifically, your response should:"
#                 "\n1. Contain all the information which he get from query results"

#                 "\n1) Abstract number it unique number each report separately <<TH-? | SH-?>>"
#                 "\n2) Title"
#                 "\n3) Background"
#                 "\n4) Methods"
#                 "\n5) Results"
#                 "\n6) Conclusion"
#                 "\n7) Funding"

#             ),
#         },
#         {"role": "user", "query": query },
#     ]
# retrieval_results = index.query(str(query))
# print(retrieval_results)
