from transformers import AutoTokenizer, AutoModel

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("microsoft/all-MiniLM-L6-v2")

# Example text to encode
text = "The all-MiniLM-L6-v2 model is capable of encoding text into high-dimensional vectors."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Generate the embeddings
with torch.no_grad(): # Disable gradient calculation for inference
    outputs = model(**inputs)

# The last hidden states are the embeddings
embeddings = outputs.last_hidden_state

# To get a single vector representation for the entire sentence, you can take the mean of the embeddings
sentence_embedding = embeddings.mean(dim=1)

print(sentence_embedding)
