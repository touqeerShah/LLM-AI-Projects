from sentence_transformers import SentenceTransformer
import csv

sentences = ["This is an example sentence", "Each sentence is converted", "Hello To the Moon", "Data Science", "Phrama Blockchain", "got it , next time not happen"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
# print(embeddings[0])


# Assuming 'embeddings' contains the embeddings for your sentences
# And 'sentences' contains your original sentences

# Prepare data to be written into CSV
# Here, we convert each embedding numpy array to a list for easier handling
data_to_write = [(i, sentence, embedding.tolist()) for i, (sentence, embedding) in enumerate(zip(sentences, embeddings))]

# Specify the CSV file name
csv_file_name = 'sentence_embeddings.csv'

# Writing to CSV
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["index", "text_chunk", "embeddings_array"])  # Writing the header
    for row in data_to_write:
        writer.writerow(row)

print(f"Embeddings and indices have been saved to {csv_file_name}.")