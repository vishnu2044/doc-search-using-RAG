from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import Depends
from decouple import config
from langchain.vectorstores import Qdrant as QdrantVectorStore
from langchain_core.documents import Document


# Rename your custom Document class to avoid conflict
class MyDocument:
    def __init__(self, content):
        self.content = content  # This might be the actual attribute



def inspect_document(doc):
    if isinstance(doc, Document):
        print("Document class detected.")
        print("Attributes and methods of Document:", dir(doc))
        if hasattr(doc, 'content'):
            print("Content attribute found:", doc.content)
        elif hasattr(doc, 'get_text'):
            print("get_text method found:", doc.get_text())
        else:
            print("No recognizable text attribute or method found.")
    else:
        print("Not a Document instance.")



# Define Qdrant settings
class QdrantDB:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(url = config("QDRANT_URL"), api_key=config("QDRANT_API_KEY"))  # Replace with actual Qdrant URL if remote

        # Define collection name
        self.collection_name = 'document_vectors'

        # Define embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("HuggingFaceEmbeddings initialized successfully.")
        except Exception as e:
            print(f"Error initializing HuggingFaceEmbeddings: {e}")
            raise e
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("HuggingFaceEmbedding is working :::::", self.embeddings)

        # Ensure collection exists in Qdrant
        self.create_collection()

    def create_collection(self):
        # Retrieve embedding dimension
        try:
            # Generate a sample embedding to determine the dimension
            sample_text = "test sentence"
            embedding_vector = self.embeddings.embed_query(sample_text)
            embedding_dim = len(embedding_vector)
            print(f"Embedding dimension dynamically determined: {embedding_dim}")
        except AttributeError:
            # Fallback if embedding_dim is not available
            embedding_dim = 768  # Default dimension
        except Exception as e:
            print(f"Error retrieving embedding dimension: {e}")
            raise e

        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("Embedding dime collection creation done successfully")
        print(embedding_dim)
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

        # Create collection if it doesn't exist
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            print(f"Qdrant collection '{self.collection_name}' created/recreated.")
        except Exception as e:
            print(f"Error creating collection in Qdrant: {e}")
            raise e
        

    def verify_storage(self, user_email, file_name):
        print("Verifying stored documents...")

        # You can also manually inspect the data in the collection
        stored_data = self.client.scroll(
            collection_name=self.collection_name,
            limit=10,
            # with_vector=True  # Ensure vectors are included
        )
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Stored data in Qdrant after upsert:", stored_data['vector'])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        documents = self.retrieve_documents(user_email, file_name)
        
        if documents:
            print(f"Documents retrieved successfully for {user_email} and {file_name}.")
            return documents
        else:
            print(f"No documents found for {user_email} and {file_name}.")
            return None

    def store_documents(self, texts, user_email, file_name):
        try:
            documents_with_metadata = []
            print("Stored documents is working")
            print("User email ::", user_email)

            for doc in texts:
                print("Type of doc:", type(doc))

                # Inspect the document
                inspect_document(doc)

                if isinstance(doc, Document):
                    print("Document class detected.")
                    if hasattr(doc, 'content'):
                        text = doc.content
                        print("Content attribute found:", text)
                    elif hasattr(doc, 'get_text'):
                        text = doc.get_text()
                        print("get_text method found:", text)
                    else:
                        print("No recognizable text attribute or method found.")
                        text = str(doc)  
                elif isinstance(doc, str):
                    text = doc
                    print("String detected:", type(text))
                else:
                    raise TypeError(f"Expected text to be a string or Document, got {type(doc)}")

                try:
                    embedding_vector = self.embeddings.embed_query(text)
                except Exception as e:
                    print(f"Issue on setting embedding query: {e}")
                    raise e

                # print("Embedding vector ::::::", embedding_vector)
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                if not isinstance(embedding_vector, (list, tuple)):
                    raise TypeError("Embedding vector is not of type list or tuple")
                if not all(isinstance(x, (int, float)) for x in embedding_vector):
                    raise ValueError("Embedding vector contains non-numeric values")

                documents_with_metadata.append({
                    'embedding': embedding_vector,
                    'metadata': {
                        'email': user_email,
                        'file_name': file_name
                        }
                })

            print("Adding document with metadata works fine >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            ids = [i for i in range(len(documents_with_metadata))]
            vectors = [doc['embedding'] for doc in documents_with_metadata]
            payloads = [doc['metadata'] for doc in documents_with_metadata]
            print("Vector and payload and ids setup correct here :::,  >>>>>>><<<<<<<<<<<<<<<<<", vectors)

            try:
                upsert_response =  self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        {
                        'id': id_,
                        'vector': vector,
                        'payload': payload
                    } for id_, vector, payload in zip(ids, vectors, payloads)]
                )
                # self.verify_storage(user_email,file_name)
                print("::::::::::::::::::::::::::::::::::::::::::::::::")
                print("::::::::::::::::::::::::::::::::::::::::::::::::")
                print("Upsert Response :::::::", upsert_response)
                print(f"Documents successfully stored in collection '{self.collection_name}'")
                # print("vector :: ", vectors)
                print("::::::::::::::::::::::::::::::::::::::::::::::::")
            except Exception as e:
                print(f"Error storing documents in Qdrant: {e}")
                raise e
        except Exception as e:
            print(f"Error in store_documents function: {e}")
            raise e
        

    def retrieve_documents(self, user_email, file_name):
        try:
            print("Query is working ::::::::::::")
            query_text = f"{user_email} {file_name}"
            query_vector = self.embeddings.embed_query("vishnu")

            # Define the query filter for matching metadata fields
            query_filter = {
                "must": [
                    {"key": "email", "match": {"value": user_email}},
                    {"key": "file_name", "match": {"value": file_name}}
                ]
            }

            print("Query data :::::::", query_filter)
            print("Colllection name ::::::", self.collection_name)

            stored_data = self.client.scroll(
                collection_name=self.collection_name,
                limit=10  # Fetch 10 stored points
            )
            print("Stored data in Qdrant:", stored_data)


            # Perform the query with a filter
            try:
                response = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=10  # Adjust this as needed
                )
                print("response get successfully !!!!!!!")

            except Exception as e:
                print("Exception error, ::::::", e)
                raise e

            print("Query response ::::", response)

            # Extract and return the results
            results = response['result']['hits']
            documents = [hit['payload'] for hit in results]
            return documents

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            raise e



def move_into_qdrant_db(texts, user_email, file_name):
    print("Move into quarant db is working ?????????????????????????????????????????????????")
    qdrant_db = QdrantDB()
    print("Qdrant db is works fine !!!!!!")
    qdrant_db.store_documents(texts, user_email, file_name)

    print("Documents stored in Qdrant DB successfully!")

# Dependency to inject QdrantDB into routes
def get_qdrant_db():
    return QdrantDB()


def get_doc_details_from_qdrant_db(user_email, file_name):
    print("get function is working :::::")
    qdrant_db = QdrantDB()
    x = qdrant_db.retrieve_documents(user_email, file_name)
    print("retrieved data :::::::", x)
    return x
