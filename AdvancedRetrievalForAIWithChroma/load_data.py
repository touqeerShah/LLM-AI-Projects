from helper_utils import load_chroma, load_chroma_gpt
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction()

chroma_collection = load_chroma_gpt(filename='./FormyconAG_EN_H1-2023_online.pdf', collection_name='FormyconAG_EN_H1')
# chroma_collection.count()