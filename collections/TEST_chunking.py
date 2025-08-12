import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunking_evaluation import BaseChunker, GeneralEvaluation
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

from sentence_transformers import SentenceTransformer
import torch

from chunking_evaluation.chunking import FixedTokenChunker, RecursiveTokenChunker, ClusterSemanticChunker, LLMSemanticChunker, KamradtModifiedChunker
from chunking_evaluation import GeneralEvaluation, SyntheticEvaluation, BaseChunker
from chunking_evaluation.utils import openai_token_count
from chromadb.utils import embedding_functions
import pandas as pd
import http.client



from embeddings import QwenEmbeddingFunction


ef = QwenEmbeddingFunction(
    model_name="./models/embedding/qwen-emb",
    device="mps",
    model_kwargs={"torch_dtype": torch.float16}
)

chunkers = [
    RecursiveTokenChunker(chunk_size=800, chunk_overlap=400, length_function=openai_token_count),
    FixedTokenChunker(chunk_size=800, chunk_overlap=400, encoding_name="cl100k_base"),
    RecursiveTokenChunker(chunk_size=400, chunk_overlap=200, length_function=openai_token_count),
    FixedTokenChunker(chunk_size=400, chunk_overlap=200, encoding_name="cl100k_base"),
    RecursiveTokenChunker(chunk_size=400, chunk_overlap=0, length_function=openai_token_count),
    FixedTokenChunker(chunk_size=400, chunk_overlap=0, encoding_name="cl100k_base"),
    RecursiveTokenChunker(chunk_size=200, chunk_overlap=0, length_function=openai_token_count),
    FixedTokenChunker(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
]

chunkers.extend(
    [
        ClusterSemanticChunker(embedding_function=ef, max_chunk_size=400, length_function=openai_token_count),
        ClusterSemanticChunker(embedding_function=ef, max_chunk_size=200, length_function=openai_token_count)
    ]
)

# Initialize evaluation
evaluation = GeneralEvaluation()

results = []

# Initialize an empty DataFrame
df = pd.DataFrame()

# Display the DataFrame


for chunker in chunkers:
    result = evaluation.run(chunker, ef, retrieve=5)
    del result['corpora_scores']  # Remove detailed scores for brevity
    chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else 0
    chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else 0
    result['chunker'] = chunker.__class__.__name__ + f"_{chunk_size}_{chunk_overlap}"
    results.append(result)

    # Update the DataFrame
    df = pd.DataFrame(results)

df

'''

def download_text(book_id, file_name, directory):
    conn = http.client.HTTPSConnection("www.gutenberg.org")
    url = f"/files/{book_id}/{book_id}-0.txt"

    conn.request("GET", url)
    response = conn.getresponse()

    if response.status == 200:
        text = response.read().decode('utf-8')

        # Create directory if it does not exist
        os.makedirs(directory, exist_ok=True)

        # Save the text to the specified file within the directory
        file_path = os.path.join(directory, file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Book '{file_name}' downloaded and saved successfully in '{directory}'.")
    else:
        print(f"Failed to download the book. Status code: {response.status}")

# Define the books to download with their IDs and file names
books = {
    1661: "the_adventures_of_sherlock_holmes.txt",
    1342: "pride_and_prejudice.txt",
    174: "the_picture_of_dorian_gray.txt"
}

# Define the directory to save the books
directory = "corpora"

# Download each book
for book_id, file_name in books.items():
    download_text(book_id, file_name, directory)

'''