# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
from langchain.embeddings import GPT4AllEmbeddings
from llama_index.llms.ollama import Ollama
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext

DATA_PATH="./new_index"
embedding_function = GPT4AllEmbeddings()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)


# create client and a new collection


def load_document_in_index():
    documents = SimpleDirectoryReader("pdf").load_data()

    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("formcon")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    return index.as_query_engine()

def load_index():
    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("formcon")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index.as_query_engine()

load_document_in_index();
# index=load_index()
# # retriever = index.as_retriever(similarity_top_k=50)
# query=" What insights can be gained from comparing Formycon AG's asset and liability composition at the two reporting dates mentioned in the financial statements of their Half-Year Report 2023?"
# content="The query is asking for an analysis or insights that can be gained by comparing the composition of Formycon AG's assets and liabilities at the two reporting dates mentioned in their Half-Year Report 2023."
# messages = [
#         {
#             "role": "system",
#             "content": (
#                 "Provide a thorough answer to the user's query. If the precise answer is unavailable, "
#                 "supply closely related information. The response must be detailed enough to explain well "
#                 "Specifically, your response should:"
#                 "\n1. Contain all the information which he get from query results"
#                 """
#                     \n- **Headings**: Use '#' before a heading to denote it. For emphasis, headings can be bolded by enclosing them in '**',headings should not longer then 9 words .
#                     \n- **Bullet Points**: Use '-' to introduce each item in a list.
#                     \n- **Plain Text**: Enclose simple text in double quotes (\").
#                 """
#             ),
#         },
#         {"role": "user", "content": query + " " + content},
#     ]
# retrieval_results = index.query(str(messages))
# print(retrieval_results)