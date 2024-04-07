import chromadb

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import numpy as np
from pypdf import PdfReader
from tqdm import tqdm
from langchain.llms import Ollama
import ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

from sentence_transformers import CrossEncoder
from langchain.embeddings import GPT4AllEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.pydantic_v1 import Extra, root_validator
from typing import Dict, Optional, Sequence
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from sentence_transformers import CrossEncoder
class Document:
    def __init__(self, text, source):
        self.page_content = text
        self.metadata = {"source": source}
bm25_retriever = BM25Retriever.from_documents([Document("demo","demo")])
### Multiple Query
from typing import List
import logging

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


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
    


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


###

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
DB_PATH = "vectorstores/db/"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1250, chunk_overlap=100, length_function=len, is_separator_regex=False
)


#
def _read_pdf(filename):
    reader = PdfReader(filename)

    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=256
    )

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def get_vector_store(COLLECTION_NAME):
    # get/create a chroma client
    vectorstore = Chroma(
        embedding_function=GPT4AllEmbeddings(),
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def get_or_create_client_and_collection(COLLECTION_NAME):
    # get/create a chroma client
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # get or create collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    return collection


def load_chroma(filename, collection_name, embedding_function):
    texts = _read_pdf(filename)
    chunks = _chunk_texts(texts)

    chroma_cliet = chromadb.PersistentClient(path=DB_PATH)

    chroma_collection = chroma_cliet.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    ids = [str(i) for i in range(len(chunks))]

    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection


def load_chroma_gpt(filename, collection_name):
    texts = _read_pdf(filename)
    print("texts", (texts[0]))
    # chunks = _chunk_texts(texts)

    vectorstore = Chroma(
        embedding_function=GPT4AllEmbeddings(),
        persist_directory=DB_PATH,
        collection_name=collection_name,
    )
    all_split_docs = []  # To collect all split documents for the BM25Retriever

    for text in texts:
        # Create an instance of Document with the text
        doc_obj = Document(text, collection_name)
        # Now pass a list of these Document instances

        split_docs = text_splitter.split_documents([doc_obj])
        # print(split_docs)
        vectorstore.add_documents(split_docs)
        all_split_docs.extend(split_docs)


    # ids = [str(i) for i in range(len(chunks))]

    # chroma_collection.add(ids=ids, documents=chunks)
    # vectorstore.add_documents(split_docs)
    vectorstore.persist()
    bm25_retriever = BM25Retriever.from_documents(all_split_docs)
    bm25_retriever.k=10
    return vectorstore


def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return (
            string[:n_chars].rsplit(" ", 1)[0]
            + "\n"
            + word_wrap(string[len(string[:n_chars].rsplit(" ", 1)[0]) + 1 :], n_chars)
        )


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])


def get_best_three_document(original_query, queries, COLLECTION_NAME):
    # Assuming get_or_create_client_and_collection is a predefined function that initializes
    # the collection client
    chroma_collection = get_or_create_client_and_collection(COLLECTION_NAME)
    # Querying the collection
    results = chroma_collection.query(
        query_texts=queries, n_results=10, include=["documents", "embeddings"]
    )
    retrieved_documents = results["documents"]

    # Extracting unique documents
    unique_documents = set()
    for documents in retrieved_documents:
        for document in documents:
            unique_documents.add(
                document["content"]
            )  # Assuming document has 'content' key

    unique_documents = list(unique_documents)

    # Scoring documents
    pairs = [[original_query, doc] for doc in unique_documents]
    scores = cross_encoder.predict(pairs)

    # Pairing each document with its score
    scored_documents = list(zip(unique_documents, scores))

    # Sorting documents by their scores in descending order
    scored_documents.sort(key=lambda x: x[1], reverse=True)

    # Selecting the top 40% of the documents
    top_40_percent_index = max(
        1, len(scored_documents) * 40 // 100
    )  # Ensure at least one document is returned
    top_documents = scored_documents[:top_40_percent_index]

    return top_documents

def get_llm_object(COLLECTION_NAME):

    # compression_pipeline = get_compression_pipeline(docs)
    llm = Ollama(
        model=model,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    qa_advanced = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=get_compression_pipeline(COLLECTION_NAME),
        return_source_documents=False,
    )
    return qa_advanced
def generate_content_for_query(query, model,COLLECTION_NAME):
    messages = [
        {
            "role": "system",
            "content": "explain the query contents what it query mean? used vector source to answer it.",
        },
        {"role": "user", "content": query},
    ]

    # compression_pipeline = get_compression_pipeline(docs)
   
    qa_advanced =get_llm_object(COLLECTION_NAME)
    qa_adv_response = qa_advanced(str(messages))
    return qa_adv_response["result"]


def augment_multiple_query(question, content, model, COLLECTION_NAME, no_question):
    messages = [
        {
            "role": "system",
            "content ": (content)
            + " "
            + "Suggest up to "+str(no_question)+" additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Suggest used vector storage which provided to create questions"
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions.",
        },
        {"role": "user", "content": question},
    ]
    qa_advanced =get_llm_object(COLLECTION_NAME)

    qa_adv_response = qa_advanced(str(messages))
    content = qa_adv_response["result"]
    content = content.split("\n")
    return  content


def augment_query_generated(query, content, docs, model, COLLECTION_NAME):
    content = generate_content_for_query(query,model,COLLECTION_NAME)
    messages = [
        {"role": "system", "content": content +" Suggest if you don't know answer of the query just come up with similar information most related to it"},
        {"role": "user", "content": query},
    ]
    qa_advanced =get_llm_object(COLLECTION_NAME)

    qa_adv_response = qa_advanced(str(messages))

    return qa_adv_response["result"]


def get_compression_pipeline(COLLECTION_NAME):
    embeddings = GPT4AllEmbeddings()

    vectorstore = get_vector_store(COLLECTION_NAME)
   
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
    return compression_pipeline
