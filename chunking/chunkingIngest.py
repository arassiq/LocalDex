import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob, hashlib
from typing import Any, Dict, List, Sequence, Union, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import re, unicodedata

from tree_sitter_language_pack import get_parser

from embeddings import QwenEmbeddingFunction_600M

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter

import torch

class noteChunker:
    def __init__(
        self,
        ef: Union[object, None] = None,               # "balanced" | "high_recall" | "high_precision"
        breakpoint_threshold_amount: float = 0.5,
        buffer_size: int = 256,
        add_start_index: bool = True,
        bptype: str = "standard_deviation",
        number_of_chunks: Union[int, None] = None,
        sentence_split_regex: str = r"(?<=[.!?])\s+|\n{2,}|\n[-*]\s+"
                 ) -> None:
        
        self.ef = ef
        self.buffer_size = buffer_size
        self.add_start_index = add_start_index
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.bptype = bptype
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex

        if self.ef is None:
            self.ef = QwenEmbeddingFunction_600M(
                model_name="./models/embedding/qwen-emb-600M",
                device="mps",
                model_kwargs={"torch_dtype": torch.float32}, #32 preffered by mps, add option to change for cuda 
            )

        #min_tokens = max(150, self.max_tokens // 2)

        self.splitter = SemanticChunker(
            embeddings=self.ef,
            buffer_size=self.buffer_size,
            add_start_index=self.add_start_index,
            breakpoint_threshold_type=self.bptype,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            number_of_chunks=self.number_of_chunks,
            sentence_split_regex=self.sentence_split_regex
        )

    def clean_text_for_chunking(self, txt: str) -> str:
        txt = unicodedata.normalize("NFKC", txt).replace("\r\n", "\n").replace("\r", "\n")
        txt = re.sub(r"[ \t]+", " ", txt)               # collapse spaces/tabs
        txt = re.sub(r"\s+([.!?])", r"\1", txt)         # remove spaces BEFORE .?!  ("text ."=> "text.")
        txt = re.sub(r"([.!?])([^\s\W])", r"\1 \2", txt)# ensure space AFTER .?! if next is word
        txt = re.sub(r"\.{3,}", "...", txt)             # normalize ellipses
        txt = re.sub(r"\n{3,}", "\n\n", txt)            # normalize big gaps to blank line
        return txt.strip()

    @staticmethod
    def id_maker(text_content: str) -> str:
        return hashlib.md5(text_content.encode("utf-8")).hexdigest()

    def __call__(self, path: str):
        # Read and normalize text from disk
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        txt = self.clean_text_for_chunking(txt)

        # Split into semantic chunks
        chunks = self.splitter.split_text(txt)
        recs: List[Tuple[str, str, Dict[str, Any]]] = []
        abs_path = os.path.abspath(path)
        for i, ch in enumerate(chunks):
            text = ch
            cid = hashlib.md5(f"{abs_path}:{i}:{text}".encode()).hexdigest()
            recs.append(
                (
                    cid,
                    text,
                    {
                        "source_path": abs_path,
                        "chunk_id": i,
                        "span": None,  # inclusive line range (N/A for plain text)
                        "chunk_type": "semantic-text",
                    },
                )
            )
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
        

    @staticmethod
    def approx_tokens(s: str) -> int:
        return max(1, len(s) // 4)

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