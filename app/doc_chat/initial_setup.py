import os
from decouple import config
from langchain.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Ensure you're using the updated import
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from doc_model import move_into_qdrant_db, get_doc_details_from_qdrant_db
from dependancies import get_current_user
from fastapi import APIRouter, Depends, HTTPException



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
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    move_into_chroma_db(texts)

    return texts

#  here will be get the stored data according to the user email and file name
def get_chunks_from_qdrant(user_email, file_name):
    print("User email::::::", user_email)
    print("file_name::::::", file_name)
    ar = get_doc_details_from_qdrant_db(user_email, file_name)
    print("retrieved data in initial setup file :::::::", ar)




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

    # print("Vector db ::", vectordb)
    # query = "tell me about vishnu narayanan"
    # # load_and_query_chroma_db(query)

    # doc_retriver(vectordb, query)


def load_and_query_chroma_db(query, persist_directory='doc_db'):
    print("Load and query started to working fine :::::::::::::::::::::::::::::::::::::::")
    print("Entered query ::::::::", query)
    
    # Explicitly pass the model name to avoid the deprecation warning
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("Embedding model loaded !!")
    except Exception as e:
        print(f"error:::::::: {e} ")
    
    try:

        # Load the existing Chroma database
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        response  = doc_retriver(vectordb, query)
        return response
    except Exception as e:
        print(f"error:::::::: {e} ")
    

    
    # Convert the query into an embedding
    # try:
    #     query_embedding = embeddings.embed_query(query)
    #     print("Query embedding created: ", query_embedding)
    # except Exception as e:
    #     print(f"Error during query embedding creation: {e}")
    
    # try:
    #     print(f"Query embedding type: {type(query_embedding)}, embedding content: {query_embedding}")
    #     # Search the vector database for the most relevant document(s) based on the query embedding
    #     results = vectordb.similarity_search(query_embedding, k=1)  # k=1 returns the top match
    #     print(f"Results: {results}")
    # except Exception as e:
    #     print(f"Error during similarity search: {e}")      
    
    # # Print the most relevant document or the answer
    # try:
    #     if results:
    #         print("Relevant document found:")
    #         for result in results:
    #             print(result.page_content)  # This is where your document content is stored
    #     else:
    #         print("No relevant documents found.")
    # except Exception as e:
    #     print(f"error from print results:::::::: {e} ")











def doc_retriver(vectordb, query):
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
    # query = "talk me about vishnu's work experience?"

    # Invoke the chain with the query
    response = qa_chain({"query": query})

    extracted_query = response.get("query")
    extracted_result = response.get("result")

    # Print only the query and result
    print(f"Query: {extracted_query}")
    print(f"Result: {extracted_result}")
    return ({
        "Query": extracted_query,
        "Result": extracted_result
        })
