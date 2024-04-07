##########################################################################################################
'''
Includes some functions to create a new vector store collection, fill it and query it
'''
##########################################################################################################
import chromadb
from chromadb.config import Settings
import pandas as pd
import ast  # Module to safely evaluate strings containing Python expressions
import json

# vector store settings
VECTOR_STORE_PATH = r'./02_Data/00_Vector_Store'
COLLECTION_NAME = 'my_collection'

# Load embeddings_df.csv into data frame
embeddings_df = pd.read_csv('./mongoDB_embeddings.csv')

def get_or_create_client_and_collection(VECTOR_STORE_PATH, COLLECTION_NAME):
    # get/create a chroma client
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # get or create collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    return collection

# get or create collection
collection = get_or_create_client_and_collection(VECTOR_STORE_PATH, COLLECTION_NAME)

def add_to_collection(embeddings_df):
    # add a sample entry to collection
    # collection.add(
    #     documents=["This is a document", "This is another document"],
    #     metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    #     ids=["id1", "id2"]
    # )

    # combine all dimensions of the vector embeddings to one array
    embeddings_df['embeddings_array'] = embeddings_df['embeddings_array'].apply(ast.literal_eval)

    # Now, prepare the data for adding to the collection
    embeddings = embeddings_df['embeddings_array'].tolist()
    sentences = embeddings_df['text_chunk'].tolist()
    ids = embeddings_df['index'].apply(str).tolist()  # Convert indices to strings to use as IDs
    # print("embeddings",embeddings)
    # Define a common source for all documents
        
        # If you have unique metadata for each document, prepare a list of dictionaries here
        # For demonstration, using the same source for all entries
    metadatas = []

    for text_chunk in embeddings_df['text_chunk']:
        try:
            # Load the JSON content from the text_chunk
            content = json.loads(text_chunk)
            # Extract the URL, assuming 'url' is the key in the JSON object
            url = content.get("url", "No URL Found")  # Provide a default in case 'url' key doesn't exist
        except json.JSONDecodeError:
            url = "Invalid JSON"  # Handle cases where text_chunk is not valid JSON

        # Create metadata dictionary for this row and append to metadatas list
        metadatas.append({"source": url})

    # Assuming 'collection' is already defined and ready to add documents to
    collection.add(
        embeddings=embeddings,
        documents=sentences,
        metadatas=metadatas,  # Include the metadatas in the add call
        ids=ids
    )
    # print(collection.get(include=['embeddings', 'documents', 'metadatas']))

    # print(collection.get())
# add the embeddings_df to our vector store collection
add_to_collection(embeddings_df)

# def get_all_entries(collection):
#     # query collection
#     existing_docs = pd.DataFrame(collection.get()).rename(columns={0: "ids", 1:"embeddings", 2:"documents", 3:"metadatas"})
#     existing_docs.to_excel(r"./02_Data//01_vector_stores_export.xlsx")
#     return existing_docs

# # extract all entries in vector store collection
# existing_docs = get_all_entries(collection)

def query_vector_database(VECTOR_STORE_PATH, COLLECTION_NAME, query, n=1):
    # query collection
    results = collection.query(
        query_texts=query,
        # n_results=n,
        # include=["embeddings"]
    )

    print(f"Similarity Search: {n} most similar entries:")
    print(len(results))
    return results

# similarity search
similar_vector_entries = query_vector_database(VECTOR_STORE_PATH, COLLECTION_NAME, query=["lupus-alpha-2024"])