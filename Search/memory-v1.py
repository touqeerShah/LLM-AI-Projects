import logging

from __future__ import annotations

from langchain.memory import (
    ConversationBufferMemory,
    ChatMessageHistory,
    ConversationSummaryBufferMemory,
)


# history = ChatMessageHistory()
# history.add_user_message("hi! This is human")
# history.add_ai_message("whats up human?")

# print(history)

# history.messages

# memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=history,return_messages=True)
# memory.chat_memory.add_user_message("Hello! My name is Sam")
# memory.chat_memory.add_ai_message("Hey Sam. How can can I help you")

# memory.load_memory_variables({})

# history.add_messages([
#         {
#             "Human" : "hello\n",
#             "AI" : " Here is one way to have a contextual conversation based on the provided inputs:\n\n1. Create context based on the question \"hello\":\nGreetings, it seems the user is simply saying hello to begin our conversation. \n\n2. Provide an answer based on the context: \nHello! Thank you for starting our conversation. How can I assist you today? Please feel free to ask me a question - I'm here to help provide precise and comprehensive answers."
#         },
#         {
#             "Human" : "hello\n",
#             "AI" : " Hello!"
#         },
#         {
#             "Human" : "how are you?\n",
#             "AI" : " Hello! I'm doing well, thanks for asking."
#         },
#         {
#             "Human" : "what your name\n",
#             "AI" : " My name is Claude."
#         },
#         {
#             "Human" : "what you know about formycon?\n",
#             "AI" : [
#                 " I do not have enough information from the provided context to answer the question \"what you know about formycon?\". The context provides details about Formycon Group and its operations but does not include any facts about what the question asker already knows. Please provide some additional context about what information you are looking to obtain about Formycon in order to give a more helpful response."
#             ]
#         },
#         {
#             "Human" : "what you know about formycon annual report\n?\n\n",
#             "AI" : [
#                 " I don't have enough information from the provided context to answer the question about Formycon's annual report. The context does not mention an annual report or provide any details about Formycon that would help explain what an annual report may contain."
#             ]
#         },
#         {
#             "Human" : "what you know about formycon annual report ?\n",
#             "AI" : [
#                 " Based on the context provided, I do not have enough information to answer your question about Formycon's annual report. The passages given provide details about Formycon's practices around employee training, satisfaction surveys, health and safety, recruiting and diversity initiatives. However, there is no explicit mention of Formycon's annual report. Without more relevant context around their annual reporting processes, I do not have enough data to discuss their annual report. I apologize I cannot provide a more substantive response given the limited information here regarding Formycon's annual report specifically. Please let me know if you have any other context you can provide."
#             ]
#         },
#         {
#             "Human" : "formycon\n",
#             "AI" : [
#                 " 1. **Answer:** The context provides details about Formycon AG's Supervisory Board and Audit Committee activities during fiscal year 2022.",
#                 "2. **Supporting Information:** Based on the details given in the annual report excerpt, the Supervisory Board and Audit Committee of Formycon AG appropriately carried out their governance and oversight responsibilities during the fiscal year 2022. The Supervisory Board met quarterly as required and discussed a range of important matters relating to the company's finances, strategy, risk management, legal compliance and human resources. Likewise, the Audit Committee composed of three members reviewed the company's financial reports and audit processes on a regular basis, as their role demands. Meeting attendance is documented, also indicating due process was followed. No issues seem to have arisen based on the information provided."
#             ]
#         }]
#     )

# print("1.memory",memory)


from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    PromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

llm = Ollama(
    model="mistral",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

connection_string = ("mongodb://stada:stada@localhost:27017",)
database_name = "admin"
collection_name = "chats"
chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string=connection_string,
    database_name="admin",
    collection_name="chats",
)

# chat_message_history.add_user_message(
#     "Pharmatrace is Blockchain based ecosystem where the pharmaceutical industry works together to do better business based on a common distributed ledger."
# )
# chat_message_history.add_ai_message("Hi")


chat_memory = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string=connection_string,
    database_name=database_name,
    collection_name=collection_name,
)

# memory = ConversationSummaryBufferMemory(
#  llm=llm,  chat_memory=chat_memory, max_token_limit=10
# )

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=chat_memory,
    return_messages=True,
    # output_key="answer",
)

# print("2.memory", memory2.load_memory_variables({}))

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """role : system
                "context": 
                You are an expert assistant with a strong ability to provide precise and comprehensive answers. Here is a question. Please provide a direct and concise answer in no more than two lines. Utilize the available data in the vector store to ensure accuracy and comprehensiveness in your response.

                - Follow the answer with a detailed paragraph that includes supporting arguments, explanations, or related information. This paragraph should be between 50 to 200 words based on the complexity of the question.
                
                Utilize the available data in the vector store to ensure accuracy and comprehensiveness in your response.
                """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
# print("prompt",prompt)
# conversation_chain = LLMChain(llm=llm, prompt=prompt, memory=memory2)

# # conversation_chain({"question": "Hello, Many name is touqeer"})
# conversation_chain({"question": "What do you know about Pharmatrace?"})


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

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import json
import asyncio

DB_PATH = "./multimodal_collection"

# Set up RetrievelQA model
# QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")


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



template2 = """
    Your name is Bot.
    You are a chatbot specialized in human resources. 
    Use the following context (delimited by <ctx></ctx>) to answer the questions.
    If there is any history of previous conversations, use it to answer (delimited by <hs></hs>)
    If you don't know the answer just answer that you don't know. 
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    Question:
    {question} 
    """


prompt2 = PromptTemplate(
    template=template2, input_variables=["context", "chat_history", "question"]
)


qa_advanced = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_pipeline,
    return_source_documents=True,
)


crc = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_pipeline,
    memory=memory,
    chain_type="stuff",
    # condense_question_llm = llm,
    # combine_docs_chain_kwargs={"prompt": prompt2},
    verbose=True,
    output_key="answer",
    # return_source_documents=True,
    # get_chat_history=lambda h: h,
)
# qa_adv_response = qa_advanced("get funding of abstract number:TH-OR36 from vectore store")
# qa_adv_response["result"]
result = crc(
    {
        "question": "what you know about FYB202?",
    }
)


# async for chunks in qa.astream({"query": "get funding of abstract number:TH-OR36 from vectore store"}):
#     print(chunks)

# import logging
# from queue import Queue
# from threading import Thread


# class StreamingChain:
#     def __init__(self, llm, qa_advanced):
#         self.qa_advanced = qa_advanced
#         self.llm = llm
#         self.thread = None

#     def stream(self, input):
#         queue = Queue()
#         self.llm.callbacks = [StreamingHandler(queue)]

#         def task():
#             print(self.qa_advanced(input))

#         self.thread = Thread(target=task)
#         self.thread.start()

#         try:
#             while True:
#                 token = queue.get()
#                 if token is None:
#                     self.cleanup()
#                     break
#                 yield token
#         finally:
#             self.cleanup()

#     def cleanup(self):
#         if self.thread and self.thread.is_alive():
#             self.thread.join()


# async def main():
#     chain = StreamingChain(llm=llm, qa_advanced=crc)
#     token = ""
#     for output in chain.stream(
#         {
#             "question": "what you know about FYB202?",
#             "chat_history": memory,
#         }
#     ):
#         token += output
#         print(output)

#     print(token)


# if __name__ == "__main__":
#     asyncio.run(main())
