import tiktoken
import mlx.core as mx
import mlx.nn as nn

# Initialize tokenizer
encoder = tiktoken.get_encoding("cl100k_base")

# Sample text
text = "Hello, how are you?"

# Tokenize and get token IDs
token_ids = encoder.encode(text)
token_ids_array = mx.array(token_ids, dtype=mx.int32)  # Convert to MLX tensor

# Define embedding layer (assume large vocab, embedding size 128)
vocab_size = 100000  # Adjust based on your tokenizer
embedding_dim = 128

# Initialize MLX embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Get embeddings for token IDs
token_embeddings = embedding_layer(token_ids_array)

# Print results
print("Token IDs:", token_ids)
print("Token Embeddings Shape:", token_embeddings.shape)
print("First Token's Embedding:\n", token_embeddings[0])  # Convert back to NumPy for display

