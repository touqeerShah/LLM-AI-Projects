from __future__ import annotations

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import SentenceTransformer
from langchain.embeddings import GPT4AllEmbeddings

from langchain_community.vectorstores import Chroma
from langchain import hub

from typing import Dict, Optional, Sequence
from langchain.schema import Document
from langchain.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from sentence_transformers import CrossEncoder


from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

from langchain.chains import RetrievalQA
import json


# Set up RetrievelQA model
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")


# loader = WebBaseLoader(
#     "https://medium.aiplanet.com/implementing-rag-using-langchain-ollama-and-chainlit-on-windows-using-wsl-92d14472f15d"
# )

# documents = loader.load()
# print(documents[0])

input_filename = './../mongo-backup/scrapingdata.json'
with open(input_filename, 'r') as file:
    data = json.load(file)

embeddings=GPT4AllEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1250,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex = False
)
#


# split_docs = text_splitter.split_documents(documents)
# print(split_docs[1])
# print(len(split_docs))
DATA_PATH="pdf/"
DB_PATH = "data/"


loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()
print(f"Processed {len(documents)} pdf files")
texts=text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=texts, collection_name="full_documents",embedding=GPT4AllEmbeddings(),persist_directory=DB_PATH)     
vectorstore.persist()


bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k=10