import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.config import Settings

from embeddings import QwenEmbeddingFunction

from langchain_experimental.text_splitter import SemanticChunker

import torch

import http.client

with open("/Users/aaronrassiq/Desktop/LocalDex/chunking/text_samples/the_adventures_of_sherlock_holmes.txt") as f:
    sherlockHolmes = f.read()

ef = QwenEmbeddingFunction(
    model_name="./models/embedding/qwen-emb",
    device="mps",
    model_kwargs={"torch_dtype": torch.float16}
)

text_splitter = SemanticChunker(ef)

sherlockHolmes = sherlockHolmes[:3000]

docs = text_splitter.create_documents([sherlockHolmes])
print(docs[0].page_content)


'''
for doc in docs[0].page_content:
    print(doc, "\n\n\n\n\n")

print(len(docs))
'''