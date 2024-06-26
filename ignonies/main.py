# I works!! travere pdf with questions succesfully generates slides with llm rag content
import streamlit as st
from ppt_data_gen import slide_data_gen
from ppt_gen_withoutline import ppt_gen
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
import io
import re

import streamlit as st
import PyPDF2
from ppt_data_gen import slide_data_gen
from ppt_gen_withoutline import ppt_gen

import streamlit as st
from ppt_data_gen import slide_data_gen
from ppt_gen_withoutline import ppt_gen
import PyPDF2

import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
import streamlit as st
import PyPDF2
from ppt_data_gen import slide_data_gen
from ppt_gen_withoutline import ppt_gen

import streamlit as st
import PyPDF2
from ppt_data_gen import slide_data_gen
from ppt_gen_withoutline import ppt_gen

st.title("PPT Generator")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


# Define a function to process the PDF content
def process_pdf(pdf_content):
    # Split the text into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=0)
    # # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
    ########This gave ppt 38 long chunk size of 35000 no overlap
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size = 25000, chunk_overlap=1000, separator='', strip_whitespace=False)

    texts = text_splitter.split_text(pdf_content)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    docsearch = Chroma.from_texts(texts, embedding= embeddings, metadatas=metadatas)

    return docsearch

# Process the PDF content and prompt user for title
if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    docsearch = process_pdf(pdf_text)

    # Prompt user for title and questions
    topic = st.text_input("Enter a topic:")
    question1 = st.text_input("Question for Section 1:")
    question2 = st.text_input("Question for Section 2:")
    question3 = st.text_input("Question for Section 3:")
    question4 = st.text_input("Question for Section 4:")
    question5 = st.text_input("Question for Section 5:")

    # Generate slide data using the extracted text, topic, and questions
    if st.button("Generate"):
        questions = [question1, question2, question3, question4, question5]
        data = slide_data_gen(topic, pdf_text, docsearch, questions)
        ppt_file = ppt_gen(data)

        file_name = f"Presentation.pptx"

        st.download_button(
            label="Download Presentation",
            data=ppt_file,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
