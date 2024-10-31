from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.callbacks import AsyncIteratorCallbackHandler

from typing import Dict, Optional, Sequence
from langchain.schema import Document
from langchain.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from sentence_transformers import CrossEncoder

from langchain_core.callbacks.base import BaseCallbackHandler

from langchain_community.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain_community.llms.ollama import Ollama

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import RetrievalQA
import json
import asyncio

DB_PATH = "./multimodal_collection"

# Set up RetrievelQA model
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")


class Document:
    def __init__(self, text, source):
        self.page_content = text
        self.metadata = {"source": source}


bm25_retriever = BM25Retriever.from_documents([Document("demo", "demo")])
embeddings = GPT4AllEmbeddings()

vectorstore = Chroma(
    embedding_function=GPT4AllEmbeddings(),
    persist_directory=DB_PATH,
    collection_name="f23",
)


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


vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vs_retriever], weight=[0.5, 0.5]
)
#

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
#
reordering = LongContextReorder()
#
reranker = BgeRerank()
#
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering, reranker]
)
#
compression_pipeline = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=ensemble_retriever
)


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n + {d.page_content}" for i, d in enumerate(docs)]
        )
    )


pretty_print_docs(
    vs_retriever.get_relevant_documents("what is Ollama and why we need it?")
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, initial_text=""):
        # self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + ""
        print("on_llm_new_token", token)
        # self.container.markdown(self.text)


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, error, **kwargs):
        logging.error(f"Error in LLM: {error}")
        self.queue.put(None)


stream_handler = StreamHandler("message_placeholder")
callback = AsyncIteratorCallbackHandler()

llm = Ollama(
    model="mistral",
    verbose=True,
    # callbacks=[StreamingHandler]
)

messages = [
    {
        "role": "system",
        "content": (
            """find any record with title exact 'Change in GFR and UPC (Urinary Protein:Creatinine Ratio) Before and After Eculizumab in C3 Glomerulopathy'
          """
        ),
    },
    {
        "role": "user",
        "title": "Change in GFR and UPC (Urinary Protein:Creatinine Ratio) Before and After Eculizumab in C3 Glomerulopathy",
    },
]

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

qa_advanced = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_pipeline,
    return_source_documents=True,
)

# qa_adv_response = qa_advanced("get funding of abstract number:TH-OR36 from vectore store")
# qa_adv_response["result"]


# async for chunks in qa.astream({"query": "get funding of abstract number:TH-OR36 from vectore store"}):
#     print(chunks)

import logging
from queue import Queue
from threading import Thread
class StreamingChain:
    def __init__(self, llm, qa_advanced):
        self.qa_advanced = qa_advanced
        self.llm = llm
        self.thread = None

    def stream(self, input):
        queue = Queue()
        self.llm.callbacks = [StreamingHandler(queue)]

        def task():
            print(len(self.qa_advanced(str(input))["source_documents"]))

        self.thread = Thread(target=task)
        self.thread.start()

        try:
            while True:
                token = queue.get()
                if token is None:
                    self.cleanup()
                    break
                yield token
        finally:
            self.cleanup()

    def cleanup(self):
        if self.thread and self.thread.is_alive():
            self.thread.join()

async def main():
    chain = StreamingChain(llm=llm, qa_advanced=qa_advanced)
    token=""
    for output in chain.stream(input={"content": "report about formycon 2023 annual report? also provide of source of which you used to generate this information from metadata"}):
        token+=output
        print(output)

    print(token)
if __name__ == "__main__":
    asyncio.run(main())


# async def main():
#     task = asyncio.create_task(
#         qa_advanced.stream("get funding of abstract number:TH-OR36 from vectore store")
#     )

#     async def process_stream():
#         try:
#             async for token in callback.aiter():
#                 print(token)  # Handle the token as needed
#         except Exception as e:
#             print(f"Caught exception: {e}")
#         finally:
#             callback.done.set()

#     await process_stream()
#     await task


# if __name__ == "__main__":
#     asyncio.run(main())
