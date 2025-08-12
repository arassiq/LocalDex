import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.config import Settings
from embeddings import QwenEmbeddingFunction

from langchain_text_splitters import SemanticChunker