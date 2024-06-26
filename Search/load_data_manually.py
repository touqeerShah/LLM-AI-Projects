import re
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import hashlib
import json
from llama_index.core.schema import TextNode

from langchain_community.vectorstores import Chroma

DATA_PATH = "./manual-data"
file_path = "./KW23Abstracts.txt"

vectorstore = Chroma(
    embedding_function=GPT4AllEmbeddings(),
    persist_directory=DATA_PATH,
    collection_name="manual-data",
)


text_splitter = RecursiveCharacterTextSplitter()


class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {"source": "./KW23Abstracts.txt"}
        # Optionally, initialize a unique document ID. This could be done in various ways:
        # For simplicity, here we're generating a hash based on the document's content.


all_split_docs = []  # To collect all split documents for the BM25Retriever


def extract_sections_from_file(file_path):
    # Open and read the file
    with open(file_path, "r") as file:
        data = file.read()

    # Regex pattern to match each block starting with "TH-" until the next "TH-" or end of text
    pattern = re.compile(r"(TH-.*?)(?=\nTH-|\Z)", re.DOTALL)
    # Phrase to remove
    phrase_to_remove = "Oral Abstract  Thursday"

    # Finding all matches
    matches = pattern.findall(data)

    # Finding all matches
    matches = pattern.findall(data)

    # Display each matched block
    print(len(matches))
    print(matches[79])
    for i, match in enumerate(matches, 1):
        if len(matches) - 1 == i:
            break
        if "Funding" in match:
            match = match.replace(phrase_to_remove, "")
        # print(f"Match {i}:")
        # print(match.strip())
        # print("------\n")
        doc_obj = Document(("abstract number :" + match.strip()))
        # split_docs = text_splitter.split_documents([doc_obj])
        # print("split_docs", split_docs)
        vectorstore.add_documents([doc_obj])
        vectorstore.persist()

    return all_split_docs


# Replace 'path_to_your_file.txt' with the actual file path


def load_document_in_index():
    extract_sections_from_file(file_path)


def load_index():
    db = chromadb.PersistentClient(path=DATA_PATH)
    chroma_collection = db.get_or_create_collection("manual-data")
    print(chroma_collection.count())


# load_document_in_index()

# load_index()
llm = Ollama(
    model="mistral",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

messages = [
    {
        "role": "system",
        "content": (
            """find record with abstract number 'TH-OR01' return it background and funding'
          """
        ),
    },
    {"role": "user", "abstract number": "TH-OR01"},
]

naive_response = qa("what you know about Na/K ATPase ?")
print("\n\n")
