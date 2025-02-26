# The Anatomy of strings in Large Language Models

Large Language Models (LLMs) process text by breaking it down into tokens, a fundamental step for understanding and generating language. This repository explores the anatomy of strings in LLMs, focusing on tokenization and embeddings. Using Python libraries like tiktoken, numpy, and mlx, we analyze how text is represented numerically and processed efficiently on Apple Silicon. Understanding these concepts helps optimize model performance and memory usage. Dive in to explore the mechanics of strings in LLMs with hands-on code examples!

## Overview

This section explores how text is processed in LLMs, focusing on tokenization and embeddings. We first use the tiktoken library to convert text into tokens and token IDs. Then, we generate token embeddings using numpy. For Apple Silicon Macs, we leverage mlx to accelerate embedding computations for faster processing. Below is a step-by-step demonstration:

  ### Running In Apple Silicon Macs

  #### Initialize tokenizer
  encoder = tiktoken.get_encoding("cl100k_base")

  #### Sample text
  text = "Understanding strings in LLMs starts with tokenization."

  #### Tokenize and get token IDs
  token_ids = encoder.encode(text)
  token_ids_array = mx.array(token_ids, dtype=mx.int32)  # Convert to MLX tensor

  #### Define embedding layer (assume large vocab, embedding size 128)
  vocab_size = 100000  # Adjust based on your tokenizer
  embedding_dim = 128

  #### Initialize MLX embedding layer
  embedding_layer = nn.Embedding(vocab_size, embedding_dim)

  #### Get embeddings for token IDs
  token_embeddings = embedding_layer(token_ids_array)

  ### Running in other machines

  #### Initialize tokenizer
  encoder = tiktoken.get_encoding("cl100k_base")

  #### Sample text
  text = "Understanding strings in LLMs starts with tokenization."

  #### Tokenize and get token IDs
  token_ids = encoder.encode(text)
  token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

  #### Define embedding layer (vocab size assumed large, embedding size is 128)
  vocab_size = 100000  # This should ideally match the model's vocab size
  embedding_dim = 128  # Change as needed

  #### Define embedding model
  embedding_layer = nn.Embedding(vocab_size, embedding_dim)

  #### Get embeddings for token IDs
  token_embeddings = embedding_layer(token_ids_tensor)

## Installation

  pip3 install torch numpy
  
  pip3 install mlx

## Usage

  Use tokenization_mlx file on Apple Silcon Macs

  



     
