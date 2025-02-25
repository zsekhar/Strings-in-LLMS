import tiktoken
import torch
import torch.nn as nn

# Initialize tokenizer
encoder = tiktoken.get_encoding("cl100k_base")

# Sample text
text = "Hello, how are you?"

# Tokenize and get token IDs
token_ids = encoder.encode(text)
token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

# Define embedding layer (vocab size assumed large, embedding size is 128)
vocab_size = 100000  # This should ideally match the model's vocab size
embedding_dim = 128  # Change as needed

# Define embedding model
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Get embeddings for token IDs
token_embeddings = embedding_layer(token_ids_tensor)

# Print results
print("Text", text)
print("Token IDs:", token_ids)
print("Token Embeddings Shape:", token_embeddings.shape)
print("First Token's Embedding:\n", token_embeddings[0])
