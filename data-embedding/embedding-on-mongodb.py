from sentence_transformers import SentenceTransformer
import csv
import json

input_filename = './../mongo-backup/scrapingdata.json'
with open(input_filename, 'r') as file:
    data = json.load(file)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(data)
# print(embeddings[0])


# Assuming 'embeddings' contains the embeddings for your data
# And 'data' contains your original data

# Prepare data to be written into CSV
# Here, we convert each embedding numpy array to a list for easier handling
data_to_write = [(i, sentence, embedding.tolist()) for i, (sentence, embedding) in enumerate(zip(data, embeddings))]

# Specify the CSV file name
csv_file_name = 'mongoDB_embeddings.csv'

# Writing to CSV
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["index", "text_chunk", "embeddings_array"])  # Writing the header
    for row in data_to_write:
        writer.writerow(row)

print(f"Embeddings and indices have been saved to {csv_file_name}.")