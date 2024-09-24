import os
from decouple import config
from langchain.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from doc_model import get_doc_details_from_qdrant_db

from fastapi import  HTTPException



# Here is the section for process the document from api get the file and send to text split into chunks  { Process 01 } { api 1 }
def process_document(file_path: str, user_email: str, file_name:str):
    try:
        # Create an instance of UnstructuredFileLoader with the file path
        loader = UnstructuredFileLoader(file_path)

        # Load the document from the file
        document = loader.load()
        print("User email :::", user_email)
        print("file name :::", file_name)
        print("docs moves into text spliting ::::::::::::::::::::::")
        texts = text_split_to_chunks(document)
        print("Texts length :::", len(texts))

        # move_into_qdrant_db(texts, user_email, file_name)
        return {"Message": "document processed Successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

  
# Here is the process of the document to split into chunks { process 02 }  { api 1 }
def text_split_to_chunks(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 400
    )
    texts = text_splitter.split_documents(documents)
    # move_into_chroma_db(texts)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("From :: Text split to chunks")
    print("type of the text ::", type(texts))
    print("length of the texts ::", len(texts))
    print("chunked texts ::::", texts)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    move_into_chroma_db(texts)

    return texts






def move_into_chroma_db(texts):

    embeddings = HuggingFaceEmbeddings()
    persistant_directory = 'doc_db'
    print("into chroma db is working fine. !!")

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding= embeddings,
        persist_directory= persistant_directory
    )

    print("data stored in the chroma DB !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



def load_and_query_chroma_db(query, persist_directory='doc_db'):
    print("Load and query started to working fine :::::::::::::::::::::::::::::::::::::::")
    print("Entered query ::::::::", query)
    

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("Embedding model loaded !!")
    except Exception as e:
        print(f"error:::::::: {e} ")
    
    try:
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        response  = doc_retriver(vectordb, query)
        return response
    except Exception as e:
        print(f"error:::::::: {e} ")
    


def doc_retriver(vectordb, query):
    retriever = vectordb.as_retriever()

    # language model
    llm = ChatGroq(
        model='llama-3.1-70b-versatile',
        temperature=0,
        groq_api_key=config('GROQ_API_KEY')
    )


    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )

    # prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{query}")
        ]
    )

    # Use RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain({"query": query})

    extracted_query = response.get("query")
    extracted_result = response.get("result")


    print(f"Query:::: {extracted_query}")
    print(f"Result:::: {extracted_result}")

    return ({
        "Query": extracted_query,
        "Result": extracted_result
        })











#  here will be get the stored data according to the user email and file name
def get_chunks_from_qdrant(user_email, file_name):
    print("User email::::::", user_email)
    print("file_name::::::", file_name)
    ar = get_doc_details_from_qdrant_db(user_email, file_name)
    print("retrieved data in initial setup file :::::::", ar)












# # Define a function to load and query a Chroma vector database
# def load_and_query_chroma_db(query, persist_directory='doc_db'):
#     # Print a message indicating the start of the load and query process
#     print("Load and query started to working fine :::::::::::::::::::::::::::::::::::::::")
#     # Print the query being used
#     print("Entered query ::::::::", query)
    
#     # Try to load the HuggingFaceEmbeddings model
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         # Print a message indicating that the embedding model has been loaded successfully
#         print("Embedding model loaded !!")
#     except Exception as e:
#         # Print any errors encountered while loading the embedding model
#         print(f"error:::::::: {e} ")
    
#     # Try to load the Chroma vector database and query it
#     try:
#         # Initialize Chroma with the embedding function and persist directory
#         vectordb = Chroma(
#             embedding_function=embeddings,
#             persist_directory=persist_directory
#         )
#         # Retrieve and process documents based on the query
#         response = doc_retriver(vectordb, query)
#         # Return the response obtained from the document retriever
#         return response
#     except Exception as e:
#         # Print any errors encountered while loading the vector database or querying it
#         print(f"error:::::::: {e} ")



# # Define a function to retrieve documents based on a query
# def doc_retriver(vectordb, query):
#     # Convert the vector database into a retriever object
#     retriever = vectordb.as_retriever()

#     # Initialize the language model with specified parameters
#     llm = ChatGroq(
#         model='llama-3.1-70b-versatile',  # Model to use for language processing
#         temperature=0,  # Temperature setting for model responses
#         groq_api_key=config('GROQ_API_KEY')  # API key for accessing the Groq service
#     )

#     # Define a system prompt template for generating responses
#     system_prompt = (
#         "Use the given context to answer the question. "
#         "If you don't know the answer, say you don't know. "
#         "Use three sentences maximum and keep the answer concise. "
#         "Context: {context}"
#     )

#     # Create a prompt template for the ChatGroq model
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),  # System message defining the prompt behavior
#             ("human", "{query}")  # Human message placeholder for the query
#         ]
#     )

#     # Create a RetrievalQA chain object to handle the querying process
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,  # Language model to use for answering
#         chain_type="stuff",  # Chain type for processing the query
#         retriever=retriever,  # Document retriever to get relevant documents
#         return_source_documents=True  # Return source documents along with the answer
#     )

#     # Execute the QA chain with the provided query
#     response = qa_chain({"query": query})

#     # Extract the query and result from the response
#     extracted_query = response.get("query")
#     extracted_result = response.get("result")

#     # Print the extracted query and result
#     print(f"Query:::: {extracted_query}")
#     print(f"Result:::: {extracted_result}")

#     # Return a dictionary containing the query and result
#     return ({
#         "Query": extracted_query,
#         "Result": extracted_result
#     })



