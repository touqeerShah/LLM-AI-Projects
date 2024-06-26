from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_nomic.embeddings import NomicEmbeddings
import re
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
from langchain.llms import Ollama

def extract_items(input_string):
    # Function to extract items from a string
    items = re.findall(r'"(.*?)"', input_string)
    return items





def slide_data_gen(topic, pdf_text, docsearch, questions):
    slide_data = []

    # Retrieve embeddings of the PDF content from Chroma DB
    retriever = docsearch.as_retriever()

    # Define RAG chain prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize RAG chain with retriever
    rag_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model="qwen:14b" , temperature="0.12"),  
        retriever=retriever,
        memory=None,  # No memory needed for RAG
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )

    # Iterate over each question
    for question in questions:
        # Use the RAG chain to generate slide content for the current question
        slide_content = rag_chain.invoke(f"""
        For the topic "{topic}" and the provided text from the PDF file, summarize the key points related to "{question}".
        """)

        # Print the generated text
        print("Generated Text:")
        print(slide_content['result'])

        # Extract the generated content from the slide_content
        generated_content = extract_items(slide_content['result'])
        
        # # Append the question and generated content to slide_data
        # slide_data.append([question] + generated_content)

        # slide_data.append([question, topic, generated_content])
        slide_data.append([question, topic, slide_content['result']])


    return slide_data