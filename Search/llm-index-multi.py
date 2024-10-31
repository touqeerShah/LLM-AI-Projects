from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
from langchain_community.embeddings import GPT4AllEmbeddings

# from langchain_community.llms import Ollama
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from langchain.chains import RetrievalQA
from llama_index.core import DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)
DATA_PATH = "./new_index"
embedding_function = GPT4AllEmbeddings()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter


# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
# embed_model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
# ollama
from langchain_ollama import OllamaLLM

Settings.llm = OllamaLLM(model="mistral")

# create client and a new collection


def load_document_in_index():
    import pymupdf4llm

    llama_reader = pymupdf4llm.LlamaMarkdownReader()
    llama_docs = llama_reader.load_data("./demo.pdf")
    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("demo")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    index = VectorStoreIndex.from_documents(
        llama_docs, storage_context=storage_context, embed_model=embed_model
    )
    return index.as_query_engine()


def load_index():
    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("demo")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index


# def summary_load_index():
#     from llama_index.core.indices.document_summary import (
#     DocumentSummaryIndexLLMRetriever,
# )
#     db = chromadb.PersistentClient(path=DATA_PATH)
#     chroma_collection = db.get_or_create_collection("demo")
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#     index = DocumentSummaryIndexLLMRetriever.from_vector_store(
#         vector_store,
#         embed_model=embed_model,
#     )
#     return index


# load_document_in_index()

# # build retriever
index = load_index()
retriever = VectorIndexRetriever(
    index=load_index(),
    similarity_top_k=50,
    vector_store_query_mode="mmr",
    alpha=None,
    doc_ids=None,
)
# https://samisabiridrissi.medium.com/understanding-retrieverqueryengine-and-similaritypostprocessor-in-llamaindex-722923caca49
# https://freedium.cfd/https://samisabiridrissi.medium.com/groq-meta-ollama-llama-index-vectorstoreindex-and-query-engine-data-access-explained-dc9c1a09b82c
# build query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=get_response_synthesizer(),
)

# index=load_index()
retriever = index.as_retriever(similarity_top_k=50)
messages = [
    {"role": "user", "content": "Give me top 10 sessions related to IgA? with company names"},
]
index_query_engine = index.as_query_engine()


retrieval_results = query_engine.query(str(messages))
print(retrieval_results)
