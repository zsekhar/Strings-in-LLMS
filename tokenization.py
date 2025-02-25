import tiktoken
import numpy as np

# Initialize tokenizer
encoder = tiktoken.get_encoding("cl100k_base")

# Sample text
text = "Hello, how are you?"

# Tokenize the text and get token IDs
token_ids = encoder.encode(text)

# Define embedding dimensions
embedding_dim = 128  # You can change this based on the model

# Create random embeddings (simulating learned embeddings)
np.random.seed(42)  # For reproducibility
embeddings = {tid: np.random.randn(embedding_dim) for tid in token_ids}

# Convert token IDs into their embeddings
token_embeddings = np.array([embeddings[tid] for tid in token_ids])

# Print results
print("Text :", text)
print("Token IDs:", token_ids)
print("Token Embeddings Shape:", token_embeddings.shape)
print("First Token's Embedding:\n", token_embeddings[0])
