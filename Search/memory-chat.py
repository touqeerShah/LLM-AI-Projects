from __future__ import annotations

# from langchain.memory import (
#     ConversationBufferMemory,
#     ChatMessageHistory,
#     ConversationSummaryBufferMemory,
# )

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
#             "Human" : "how are you?\n",q
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
    HumanMessagePromptTemplate,
)

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
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

DB_PATH = "./multimodal_collection"

vectorstore = Chroma(
    embedding_function=GPT4AllEmbeddings(),
    persist_directory=DB_PATH,
    collection_name="f23",
)

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


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

from langchain.retrievers import (
    MergerRetriever,
)

ensemble_retriever = MergerRetriever(
    retrievers=[vs_retriever, vs_retriever], weight=[0.5, 0.5]
)

history_aware_retriever = create_history_aware_retriever(
    llm, ensemble_retriever, contextualize_q_prompt
)
from langchain_core.messages import HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

chat_history = []

# question = "hello?"
# # ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# chat_history.extend([HumanMessage(content=question),"my name is touqeer ?" ])
# chat_history.extend([HumanMessage(content="where are you from"),"I am from pakistan" ])
# chat_history.extend([HumanMessage(content="where from pakistan"),"karachi" ])
# chat_history.extend([HumanMessage(content="nice place tell me more about it"),"It have Nice Beach , Good Food , It Is largest city in terms of area of pakistan" ])

# # print((chat_history))

qa_system_prompt = """
You are an AI assistant name as Alpha Bot , you task is question-answering 
Use the following pieces of retrieved context to answer the question \
answer always based on need of question not to long not to short.\

{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# ai_msg_1 = rag_chain.invoke(
#     {"input": "what you know about FYB202?", "chat_history": chat_memory.messages}
# )

import asyncio

async def process_chunks():

    async for event in rag_chain.astream_events(
        {
            "input": "what you know about FYB202?",
            "chat_history": chat_memory.messages,
        },
        version="v1",
    ):
        # Process events here
        if event["name"] == "Retriever":
            print(event)
            print()
            # if (
            #     event["event"] == "on_chat_model_stream"
            #     and "contextualize_q_llm" in event["tags"]
            # ):
            #     ai_message_chunk = event["data"]["chunk"]
            #     print(f"{ai_message_chunk.content}|", end="")
# Run the asynchronous function
asyncio.run(process_chunks())

# print("= = = = > > > > " ,ai_msg_1["answer"])

# print(history_aware_retriever.invoke({"input": "what you know about FYB202?", "chat_history": chat_memory.messages}))

# chat_message_history.add_user_message(
#     "Hello my name is touqeer?, how about you?"
# )

# chat_message_history.add_ai_message(ai_msg_1["answer"])

# print("prompt",prompt)
# conversation_chain = LLMChain(llm=llm, prompt=prompt, memory=memory2)

# # conversation_chain({"question": "Hello, Many name is touqeer"})
# conversation_chain({"question": "What do you know about Pharmatrace?"})

# Version: 0.1.16