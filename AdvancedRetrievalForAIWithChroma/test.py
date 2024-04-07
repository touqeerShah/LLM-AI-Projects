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

from langchain.chains import RetrievalQA
import json
from pypdf import PdfReader



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



vectorstore = Chroma(embedding_function=GPT4AllEmbeddings(),
                     persist_directory="./data",
                     collection_name="full_documents")
all_split_docs = []  # To collect all split documents for the BM25Retriever

class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {'source': 'https://medium.aiplanet.com/implementing-rag-using-langchain-ollama-and-chainlit-on-windows-using-wsl-92d14472f15d'}

count=0
reader = PdfReader('./FormyconAG_EN_H1-2023_online.pdf')

pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]
for text in pdf_texts:

        # Create an instance of Document with the text
        doc_obj = Document(text)
        # Now pass a list of these Document instances
        count =count+1
        if count==1000:
            break;
        split_docs = text_splitter.split_documents([doc_obj])
        # print(split_docs)
        vectorstore.add_documents(split_docs)
        all_split_docs.extend(split_docs)


# After adding all documents, persist the vector store
# vectorstore.persist()
vectorstore.add_documents(split_docs)
vectorstore.persist()


bm25_retriever = BM25Retriever.from_documents(all_split_docs)
bm25_retriever.k=10



class BgeRerank(BaseDocumentCompressor):
    model_name:str = 'BAAI/bge-reranker-large'
    """Model name to use for reranking."""
    top_n: int = 3
    """Number of documents to return."""
    model:CrossEncoder = CrossEncoder(model_name)
    """CrossEncoder instance to use for reranking."""

    def bge_rerank(self,query,docs):
        model_inputs =  [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]


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
    
vs_retriever = vectorstore.as_retriever(search_kwargs={"k":10})
#

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,vs_retriever],
                                       weight=[0.5,0.5])
#

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
#
reordering = LongContextReorder()
#
reranker = BgeRerank()
#
pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter,reordering,reranker])
#
compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                      base_retriever=ensemble_retriever)

def pretty_print_docs(docs):
  print(
      f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n + {d.page_content}" for i,d in enumerate(docs)])
  )

pretty_print_docs(vs_retriever.get_relevant_documents("what is Ollama and why we need it?"))

llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever(search_kwargs={"k":5}),
                                 return_source_documents=True)

naive_response = qa("what product of ophthalmology and immunology in formycon?")
print("\n\n")
qa_advanced = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=compression_pipeline,
                                 return_source_documents=True)

qa_adv_response = qa_advanced("what product of ophthalmology and immunology in formycon?")  
qa_adv_response["result"]