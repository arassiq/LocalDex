import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob, hashlib
from typing import Any, Dict, List, Sequence, Union, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import re

import hashlib

from embeddings import QwenEmbeddingFunction

from langchain_experimental.text_splitter import SemanticChunker

import torch

#import http.client

with open("/Users/aaronrassiq/Desktop/LocalDex/chunking/text_samples/the_adventures_of_sherlock_holmes.txt") as f:
    sherlockHolmes = f.read()

text_splitter = SemanticChunker(ef)

class noteChunker:
    def __init__(
        self,
        ef: Union[object, None] = None,               # "balanced" | "high_recall" | "high_precision"
        max_tokens: int = 450,
        overlap: int = 60,
                 ) -> None:
        
        if ef is None:
            ef = QwenEmbeddingFunction(
                model_name="./models/embedding/qwen-emb",
                device="mps",
                model_kwargs={"torch_dtype": torch.float16}
            )
        
        min_tokens = max(150, max_tokens // 2)
        
        '''
        self.bptype = "standard_deviation"
        self.ef = ef
        self.max_tokens = max_tokens
        self.min_tokens = max(150, max_tokens // 2)
        self.overlap = overlap
        '''
        
        self.splitter = SemanticChunker( #good for now
            embedding=ef,
            min_chunk_size=min_tokens,
            max_chunk_size=max_tokens,
            overlap=overlap,
            breakpoint_threshold_type=self.bptype,
        )

    def id_maker(self, text_content):
        return hashlib.md5(text_content.encode('utf-8')).hexdigest()

    def __call__(self, text):
        txt = re.sub(r'[ \t]+', ' ', txt)
        txt = re.sub(r'\n{3,}', '\n\n', txt)
        txt = txt.strip()

        chunks = self.splitter.split_text(text)
        ids, docs, metas = [], [], []
        for i, ch in enumerate(chunk=s):
            ids.append(self.id_maker(ch))  # md5(path:i:chunk)
            docs.append(ch)
            metas.append({"source_path": path, "chunk_id": i})

        #TODO: needed, can move into the actual collections class?    
        for i in range(0, len(ids), batch):
            collection.upsert(ids=ids[i:i+batch],
                            documents=docs[i:i+batch],
                            metadatas=metas[i:i+batch])

class codeChunker:
    '''
    
    structure-aware splitter (functions/classes) -> semantic merge.
    
    '''
    def __init__(
            self,
            asdfasdfasdfasdfasdfasdgasdfasdfasdf
            ) -> None:
