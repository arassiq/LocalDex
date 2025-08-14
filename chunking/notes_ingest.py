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

from tree_sitter_language_pack import get_parser

from embeddings import QwenEmbeddingFunction

from langchain_experimental.text_splitter import SemanticChunker

import torch

#import http.client

with open("/Users/aaronrassiq/Desktop/LocalDex/chunking/text_samples/the_adventures_of_sherlock_holmes.txt") as f:
    sherlockHolmes = f.read()


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

    def __call__(self, path: str):
        txt = re.sub(r'[ \t]+', ' ', txt)
        txt = re.sub(r'\n{3,}', '\n\n', txt)
        txt = txt.strip()

        chunks = self.splitter.split_text(txt)
        recs = []

        for i, ch in enumerate(chunks):

            text = ch
            cid = hashlib.md5(f"{path}:{i}:{text}".encode()).hexdigest()
            recs.append((cid, text, {
                "source_path": os.path.abspath(path),
                "chunk_id": i,
                "span": None,                          # inclusive line range
                "chunk_type": "semantic-text"
            }))

        return recs

class codeChunker:
    EXT_TO_LANG = {
        ".py": "python", ".rs": "rust", ".ts": "typescript", ".js": "javascript",
        ".java": "java", ".go": "go", ".cpp": "cpp", ".c": "c"
    }
    '''
    
    structure-aware splitter (functions/classes) -> semantic merge.
    
    '''
    def __init__(self, max_tokens: int = 500) -> None:
        self.max_tokens = max_tokens
        

    def approx_tokens(s: str) -> int: return max(1, len(s)//4)

    def __call__(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        lang = self.EXT_TO_LANG.get(ext)
        if not lang:
            return []
        parser = get_parser(lang)
        code = open(path, "rb").read()
        tree = parser.parse(code)
        root = tree.root_node
        lines = code.splitlines()

        def blocks(node):
            for ch in node.children:
                if ch.type in {"function_definition","function_declaration","class_definition",
                            "method_definition","struct_item","impl_item"}:
                    yield ch
                yield from blocks(ch)

        # pack adjacent small blocks
        out, cur, cur_tok = [], [], 0
        for n in blocks(root):
            s, e = n.start_point[0], n.end_point[0] + 1
            txt = b"\n".join(lines[s:e]).decode("utf-8","ignore")
            t = self.approx_tokens(txt)
            if cur_tok + t <= self.max_tokens:
                cur.append((s,e,txt)); cur_tok += t
            else:
                if cur: out.append(cur); cur=[(s,e,txt)]; cur_tok = t
        if cur: out.append(cur)

        recs = []
        for i, group in enumerate(out):
            s, e = group[0][0], group[-1][1]            # merged span lines
            text = "\n\n".join(x[2] for x in group)
            cid = hashlib.md5(f"{path}:{i}:{text}".encode()).hexdigest()
            recs.append((cid, text, {
                "source_path": os.path.abspath(path),
                "chunk_id": i,
                "span": (s, e),                          # inclusive line range
                "chunk_type": "semantic-code"
            }))
        return recs