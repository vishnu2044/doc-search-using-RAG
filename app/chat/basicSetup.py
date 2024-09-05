import os
from decouple import config
from langchain.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# from langchain.vectorstores import Chroma
# from langchain.llms import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain, create_stuff_documents_chain

# from langchain.prompts import ChatPromptTemplate
# from langchain.llms import ChatGroq
# from langchain.chains.base import BaseChain

def process_document(file_path: str):
    # Create an instance of UnstructuredFileLoader with the file path
    loader = UnstructuredFileLoader(file_path)

    # Load the document from the file
    document = loader.load()
    text_split_to_chunks(document)

    return document


def text_split_to_chunks(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 400
    )
    texts = text_splitter.split_documents(documents)
    move_into_chroma_db(texts)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("type of the text ::", type(texts))
    print("length of the texts ::", len(texts))
    print("first chunk :::", texts[0])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

def move_into_chroma_db(texts):

    embeddings = HuggingFaceEmbeddings()
    persistant_directory = 'doc_db'
    print("into chroma db is working fine. !!")

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding= embeddings,
        persist_directory= persistant_directory
    )
    print("its completed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Vector db ::", vectordb)
    doc_retriver(vectordb)








def doc_retriver(vectordb):
    # Convert vectordb into a retriever
    retriever = vectordb.as_retriever()

    # Initialize the language model
    llm = ChatGroq(
        model='llama-3.1-70b-versatile',
        temperature=0,
        groq_api_key=config('GROQ_API_KEY')
    )

    # Define the system prompt for the LLM
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{query}")
        ]
    )

    # Use RetrievalQA chain  mmr mar
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Define the query
    query = "What is the model architecture discussed in this paper?"

    # Invoke the chain with the query
    response = qa_chain({"query": query})

    print("QA chaining is completed !!!!!!!!")
    print("Response:", response)













# def doc_retriver(vectordb):
#     retriver = vectordb.as_retriever()
#     query = "what is the model architecture discussed in this paper ? "

#     llm = ChatGroq(
#         model='llama-3.1-70b-versatile',
#         temperature=0,
#         groq_api_key=config('GROQ_API_KEY')
#     )

#     system_prompt = (
#         "Use the given context to answer the question. "
#         "If you don't know the answer, say you don't know. "
#         "Use three sentence maximum and keep the answer concise. "
#         "Context: {context}"
#     )
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             ("human", "{query}"),
#         ]
#     )
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)

#     chain = create_retrieval_chain(retriever, question_answer_chain)

#     chain.invoke({"input": query})

#     # Create a qa chaing
#     qa_chain = create_retrieval_chain(
#         llm=llm,
#         chain_type = 'stuff',
#         return_source_documents=True
#     )
#     print("Qa chaining is completed !!!!!!!!")
#     print("QA chain :::", qa_chain)



#     response = qa_chain.invoke(
#         {'query': query}
#     )

#     print("the response ::::::::::::::::::::::::::::::::::::::::::", response)





# def doc_retriver(vectordb):
#     # Convert vectordb into a retriever
#     retriever = vectordb.as_retriever()

#     # Initialize the language model
#     llm = ChatGroq(
#         model='llama-3.1-70b-versatile',
#         temperature=0,
#         groq_api_key=config('GROQ_API_KEY')  # Ensure you have imported `config`
#     )

#     # Define the system prompt for the LLM
#     system_prompt = (
#         "Use the given context to answer the question. "
#         "If you don't know the answer, say you don't know. "
#         "Use three sentences maximum and keep the answer concise. "
#         "Context: {context}"
#     )

#     # Create a prompt template
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             ("human", "{query}"),
#         ]
#     )


#     question_answer_chain = create_stuff_documents_chain(llm, prompt)


#     chain = create_retrieval_chain(retriever, question_answer_chain)


#     query = "What is the model architecture discussed in this paper?"

#     response = chain.invoke({"input": query})

#     print("QA chaining is completed !!!!!!!!")
#     print("Response:", response)