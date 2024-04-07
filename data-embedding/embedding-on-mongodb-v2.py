from langchain.embeddings import GPT4AllEmbeddings
import csv
import json

# Load your data
input_filename = './../mongo-backup/scrapingdata.json'
with open(input_filename, 'r') as file:
    data = json.load(file)

# Initialize your chosen embedding model
embedding_model = GPT4AllEmbeddings()

# Convert each JSON item to a string
sentences = [json.dumps(item) for item in data]

# Assuming the method to get embeddings might be different, e.g., `get_embeddings`
# Note: This is speculative and should be replaced with the correct method name
embeddings = [embedding_model.embed_documents(sentence) for sentence in sentences]

print("embeddings")
# Prepare data for CSV, associating each original JSON item's string representation with its embedding
data_to_write = [(i, sentences[i], embedding.tolist()) for i, embedding in enumerate(embeddings)]

# Specify the CSV file name
csv_file_name = 'mongoDB_embeddings.csv'

# Write to CSV
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["index", "json_string", "embeddings_array"])  # Writing the header
    for row in data_to_write:
        writer.writerow(row)

print(f"Embeddings and indices have been saved to {csv_file_name}.")
