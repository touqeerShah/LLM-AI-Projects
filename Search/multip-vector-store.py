# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from langchain_chroma import Chroma

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import (
    ContextualCompressionRetriever,
    # DocumentCompressorPipeline,
    MergerRetriever,
)
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain_community.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)
from typing import Dict, Optional, Sequence
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever


DATA_PATH = "./multimodal_collection"
embedding_function = GPT4AllEmbeddings()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = None

# create client and a new collection


class BgeRerank(BaseDocumentCompressor):
    model_name: str = "BAAI/bge-reranker-large"
    """Model name to use for reranking."""
    top_n: int = 3
    """Number of documents to return."""
    model: CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""

    def bge_rerank(self, query, docs):
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results


def load_document_in_index():
    documents = SimpleDirectoryReader("pdf").load_data()

    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("f22")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embedding_function
    )
    return index.as_query_engine()


def load_index():
    retrievers=[]
    for collection in ["f22","f23"]:
        db = Chroma(
            collection_name=collection,
            persist_directory=DATA_PATH,
            # client_settings=client_settings,
            embedding_function=embedding_function,
        )
        retrievers.append(db.as_retriever(search_kwargs={"k": 10}))

    print("retrievers = > ",retrievers)
    # f23 = Chroma(
    #     collection_name="f23",
    #     persist_directory=DATA_PATH,
    #     # client_settings=client_settings,
    #     embedding_function=embedding_function,
    # )
    # f22_retriever = f22.as_retriever(search_kwargs={"k": 10})
    # f23_retriever = f23.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(retrievers=retrievers)

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_function)

    reordering = LongContextReorder()
    reranker = BgeRerank()
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[ redundant_filter,reordering, reranker]
    )
    compression_pipeline = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=ensemble_retriever
    )

    return compression_pipeline


# load_document_in_index()
compression_pipeline = load_index()

    # compression_pipeline = get_compression_pipeline(docs)
llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

qa_advanced = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_pipeline,
        return_source_documents=False,
    )

qa_adv_response = qa_advanced.invoke("what you know about annual report of Formycon 2022")
print(qa_adv_response["result"])


