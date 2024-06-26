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
import re
import json

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

from sentence_transformers import CrossEncoder
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import  EnsembleRetriever
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


bm25_retriever = BM25Retriever.from_documents([Document("demo", "demo")])
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
    bm25_retriever.k = 10
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
                document["context"]
            )  # Assuming document has 'context' key

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


def get_llm_object(COLLECTION_NAME, model):

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


def generate_presentation_title(query, model, COLLECTION_NAME):
    messages = [
        {
            "role": "system",
            "context": (
                "Analyze the provided query and generate a concise, engaging title for a presentation that captures the query's essence. it length must no more then 8 words and more precise  "
                "Format the output as JSON, including a 'title' key for the presentation title. "
                "If there are additional relevant details that do not fit into the title but might be useful for understanding or elaborating on the presentation's topic, "
                "include these under an 'extra' key all details in one line. Ensure the 'title' is succinct and ignore other details ."
            ),
        },
        {
            "role": "user",
            "context": query,
        },  # Replace 'query' with the actual query string
    ]

    qa_advanced = get_llm_object(COLLECTION_NAME, model)
    qa_adv_response = qa_advanced(str(messages))
    title = qa_adv_response[
        "result"
    ].strip()  # Stripping to remove any leading/trailing whitespace

    # Now, use 'title' as part of your file name, ensuring it's concise enough to avoid the OSError
    return title


def generate_content_for_query(query, model, COLLECTION_NAME):
    
    messages = [
        {
            "role": "system",
            "context": "explain the query contents what it query mean? used vector source to answer it.",
        },
        {"role": "user", "context": query},
    ]
    #     messages = [
    #     {
    #         "role": "system",
    #         "context": (
    #             "Interpret the provided query, explaining its contents and the implications of what is being asked. "
    #             "Leverage the data available in the vector database to thoroughly understand and explicate the query's meaning. "
    #             "Ensure your explanation includes: "
    #             "\n- A clear interpretation of the query's main themes and questions. "
    #             "\n- Insight into the context or background information that informs the query, as derived from the vector DB. "
    #             "\n- Any relevant concepts or data from the vector DB that can help clarify the query's meaning or intentions. "
    #             "\n- Suggestions for further exploration or questions that could arise from the initial query, based on vector DB insights. "
    #             "\nYour goal is to provide a comprehensive understanding of the query, enhancing the user's grasp of the topic using the vector database's insights."
    #         )
    #     },
    #     {"role": "user", "context": query},
    # ]

    # compression_pipeline = get_compression_pipeline(docs)

    qa_advanced = get_llm_object(COLLECTION_NAME, model)
    qa_adv_response = qa_advanced(str(messages))
    return qa_adv_response["result"]


def augment_multiple_query(question, context, model, COLLECTION_NAME, no_question):
    # messages = [
    #     {
    #         "role": "system",
    #         "context ": (context)
    #         + " "
    #         + "Suggest up to "+str(no_question)+" additional related questions to help them find the information they need, for the provided question. "
    #         "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
    #         "Suggest used vector storage which provided to create questions"
    #         "Make sure they are complete questions, and that they are related to the original question."
    #         "Output one question per line. Do not number the questions.",
    #     },
    #     {"role": "user", "context": question},
    # ]
    messages = [
        {
            "role": "system",
            "context": (
                f"{context} "
                f"Given the topic provided, generate only {no_question} related questions by utilizing the available data in the vector store. "
                "These questions should aim to deepen understanding of the topic, offering a wide range of perspectives. "
                "Follow these guidelines: "
                "\n- Questions should be simple and direct, avoiding complex or compound sentence structures. "
                "\n- Ensure a broad coverage by varying the aspects of the topic addressed in each question. "
                "\n- Use the data in the vector store as a foundation for crafting relevant and insightful questions. "
                "\n- Each question must be a complete sentence and clearly related to the initial topic. "
                "\nsuggest only List each question on a new line without any numbering no extra comment needed."
            ),
        },
        {"role": "user", "context": question},
    ]

    qa_advanced = get_llm_object(COLLECTION_NAME, model)

    qa_adv_response = qa_advanced(str(messages))
    context = qa_adv_response["result"]
    # Split the context by newline and filter out any empty strings
    context = [line for line in context.split("\n") if line.strip()]
    return context


def augment_query_generated(query, context, docs, model, COLLECTION_NAME):
    context = generate_content_for_query(query, model, COLLECTION_NAME)
    print("\n\n\n\n\n\n\n\n query: ", query + "\n context: ", context)
    # messages = [
    #     {"role": "system", "context": context
    #      +" Suggest if you don't know answer of the query just come up with similar information most related to it"
    #      "Output should contain enought information which used to create slide and  on answer give title for the slide, based on answer what style summary , chart , or bulit points best to make slide, number of slides to display result, what will be front size  4:3(1280 x 720) so contents fit nicely , Final answer will be title ,slide style,number of slide and front size in json"
    #      },
    #     {"role": "user", "context": query},
    # ]
    messages = [
        {
            "role": "system",
            "context": (
                "Provide a thorough answer to the user's query. If the precise answer is unavailable, "
                "supply closely related information. The response must be detailed enough to explain well "
                "Specifically, your response should:"
                "\n1. Contain all the information which he get from query results"
                """
                    \n- **Headings**: Use '#' before a heading to denote it. heading length no more then three or four words.
                    \n- **Bullet Points**: Use '-' to introduce each item in a list.
                    \n- **Plain Text**: Enclose simple text in double quotes (\").
                    Heading examples: something small like this given examples

                    <<
                    "Formycon 2023 Financials"
                    "Asset vs. Liability Insights"
                    "Formycon Asset-Liability Review"
                    "2023 Financial Overview"
                    >>
                """
            ),
        },
        {"role": "user", "context": query + " " + context},
    ]

    
    qa_advanced = get_llm_object(COLLECTION_NAME, model)

    qa_adv_response = qa_advanced(str(messages))
    return qa_adv_response["result"], context


def generate_data_for_ppt(query, context, answer, model, COLLECTION_NAME):

    messages = [
        {
            "role": "system",
            "context": (
                "Based on the provided question, context, and answer, generate detailed specifications for a slide presentation."
                "Specifically, your output must include: "
                "\n1. A concise slide title reflecting the main question's theme. it length must no more then 9 words "
                "\n2. A recommended style for the presentation slides (options include: summary, chart, bullet points) that aligns with the context's nature. "
                "\n3. The estimated total number of slides needed for a thorough yet succinct presentation. "
                "\n4. A suitable font size for legibility on slides designed in a 4:3 aspect ratio (1280 x 720 pixels)."
                "Structure your response as a JSON object with 'title', 'slideStyle', 'numberOfSlides', and 'fontSize' keys for the core details. "
                "No Need to, compile any supplementary insights or suggestions no need to include anything else just title,slideStyle,numberOfSlides,fontSize "
            ),
        },
        {"role": "user", "question": query, "context": context, "answer": answer},
    ]
    qa_advanced = get_llm_object(COLLECTION_NAME, model)

    qa_adv_response = qa_advanced(str(messages))
    return qa_adv_response["result"]


def formate_data_for_ppt( answer, model, COLLECTION_NAME):

    messages = [
        {
            "role": "system",
            "context": (
                f"""based on answer make generate json object ,
                No need to add extra details just used what every you get in answer"""
            ),
        },
        {
            "role": "user",
            "context": {
                "answer": answer,
            },
        },
    ]

    qa_advanced = get_llm_object(COLLECTION_NAME, model)

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


def extract_json(text):
    # Regular expression pattern to match a JSON object
    # This is a simplified pattern and might need adjustments for complex cases
    pattern = r"\{(?:[^{}]|(?:\{.*?\}))*\}"

    # Search for JSON object in the text
    text = text.replace("\n", " ")
    match = re.search(pattern, text)

    if match:
        # Extract the matched JSON text
        json_text = match.group(0)

        try:
            # Parse the JSON text to a Python dictionary
            json_obj = json.loads(json_text)
            return json_obj
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
    else:
        print("No valid JSON object found in the text.")

    return None
