from __future__ import annotations
from typing import Any, Dict, List, Sequence, Union

from chromadb import Documents, EmbeddingFunction, Embeddings

import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from e


class QwenEmbeddingFunction_8B(EmbeddingFunction[Documents]):

    def __init__(
        self,
        model_name: str = "./models/embedding/qwen-emb-8b",
        device: Union[str, None] = None,
        batch_size: int = None, #TODO: might create 16 * 32 error in MPS, lower to 8 if memory spikes
        normalize: bool = True,
        model_kwargs: Union[Dict[str, Any], None] = None #using float16 for MPS optimization
    ) -> None:
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if device == "cuda":
            attn_impl = "sdpa"           # fast & stable on CUDA
            torch_dtype = torch.float16  # bf16/amp also fine if available
            default_batch = 8
            max_seq_len = 128
        elif device == "mps":
            attn_impl = "eager"          # SDPA on MPS is memory-spiky
            torch_dtype = torch.float32
            default_batch = 1            # start tiny; bump when stable
            max_seq_len = 96
        else:  # cpu
            attn_impl = "eager"
            torch_dtype = torch.float32
            default_batch = 8
            max_seq_len = 128

        if model_kwargs is None:
            model_kwargs = {"torch_dtype": torch.float16}

        self.model_name = model_name
        self.device = device
        self.normalize = bool(normalize)
        self.batch_size = int(batch_size) if batch_size is not None else default_batch
        self.model_kwargs = model_kwargs

        ''' 
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.device = device
        self.model_kwargs = model_kwargs
        '''

        self.model = SentenceTransformer( #load once
            self.model_name, 
            device= self.device, 
            model_kwargs=self.model_kwargs 
        )
        try:
            self.model[0].auto_model.config.attn_implementation = attn_impl
        except Exception:
            pass

        self.model.max_seq_length = max_seq_len

    @staticmethod
    def name() -> str:
        return "Qwen3-Embedding-8B"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embed input documents.

        Parameters
        ----------
        input : Documents
            Either a single string or a sequence of strings.
        Returns
        -------
        Embeddings
            A list of embeddings (List[List[float]]), one per input document.
        """
        if isinstance(input, str):
            docs: List[str] = [input] #normalize to list input 
        else:
            docs = list(input)  # type: ignore[arg-type]

        # Compute embeddings
        with torch.no_grad():
            vectors = self.model.encode(
                docs,
                batch_size=self.batch_size,
                convert_to_numpy=True, #already moves the vector to cpu 
                normalize_embeddings=self.normalize, #QWEN expects normalized embeddings
                show_progress_bar=False,
            )

        return vectors.tolist()  # type: ignore[return-value], expects a python list of lists (float)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "QwenEmbeddingFunction_8B":
        
        kw = dict(config.get("model_kwargs") or {}) #convert dtype arg to torch dtype from str
        if isinstance(kw.get("torch_dtype"), str):
            kw["torch_dtype"] = getattr(torch, kw["torch_dtype"])

        return QwenEmbeddingFunction_8B(
            model_name=config.get("model_name", "./models/embedding/qwen-emb-8b"),
            device=config.get("device"),
            batch_size=config.get("batch_size", 16),
            normalize=config.get("normalize", True),
            model_kwargs=kw #already converted config model kwargs to torch dtype, stored as kw
        )

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable config so Chroma can persist this EF."""

        safe_kwargs = dict(self.model_kwargs) #use safekwards so the saving of kwargs doesnt cause multi downstream errors trying to save a torch.dtype
        if isinstance(safe_kwargs.get("torch_dtype"), torch.dtype):
            safe_kwargs["torch_dtype"] = str(safe_kwargs["torch_dtype"]).replace("torch.", "")

        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "model_kwargs": safe_kwargs
        }

class QwenEmbeddingFunction_600M(EmbeddingFunction[Documents]):

    def __init__(
        self,
        model_name: str = "./models/embedding/qwen-emb-600M",
        device: Union[str, None] = None,
        batch_size: int = None, #TODO: might create 16 * 32 error in MPS, lower to 8 if memory spikes
        normalize: bool = True,
        model_kwargs: Union[Dict[str, Any], None] = None #using float16 for MPS optimization
    ) -> None:
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if device == "cuda":
            attn_impl = "sdpa"           # fast & stable on CUDA
            torch_dtype = torch.float16  # bf16/amp also fine if available
            default_batch = 8
            max_seq_len = 128
        elif device == "mps":
            attn_impl = "eager"          # SDPA on MPS is memory-spiky
            torch_dtype = torch.float32
            default_batch = 16          # start tiny; bump when stable
            max_seq_len = 96
        else:  # cpu
            attn_impl = "eager"
            torch_dtype = torch.float32
            default_batch = 8
            max_seq_len = 128

        if model_kwargs is None:
            model_kwargs = {"torch_dtype": torch.float16}

        self.model_name = model_name
        self.device = device
        self.normalize = bool(normalize)
        self.batch_size = int(batch_size) if batch_size is not None else default_batch
        self.model_kwargs = model_kwargs

        self.model = SentenceTransformer( #load once
            self.model_name, 
            device= self.device, 
            model_kwargs=self.model_kwargs 
        )

                # after self.model = SentenceTransformer(...)
        print("EF device (arg):", self.device)
        print("EF target_device (ST):", getattr(self.model, "_target_device", None))

        # Check the underlying HF model params live on MPS
        try:
            pdev = next(self.model[0].auto_model.parameters()).device
            print("EF HF param device:", pdev)
        except Exception as e:
            print("EF HF param device: <unavailable>", e)

        # Attention implementation chosen
        try:
            print("EF attn_impl:", self.model[0].auto_model.config.attn_implementation)
        except Exception:
            print("EF attn_impl: <unavailable>")

        # PyTorch MPS status
        print("torch.mps built:", torch.backends.mps.is_built())
        print("torch.mps available:", torch.backends.mps.is_available())

        try:
            self.model[0].auto_model.config.attn_implementation = attn_impl
        except Exception:
            pass

        self.model.max_seq_length = max_seq_len


    @staticmethod
    def name() -> str:
        return "Qwen3-Embedding-600M"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embed input documents.

        Parameters
        ----------
        input : Documents
            Either a single string or a sequence of strings.
        Returns
        -------
        Embeddings
            A list of embeddings (List[List[float]]), one per input document.
        """

        if isinstance(input, str):
            docs: List[str] = [input]
        # Treat raw bytes as one doc too (decode)
        elif isinstance(input, (bytes, bytearray, memoryview)):
            docs = [bytes(input).decode("utf-8", errors="ignore")]
        # If it's a proper sequence (list/tuple) of strings, accept it
        elif isinstance(input, (list, tuple)) and all(isinstance(x, str) for x in input):
            docs = list(input)
        else:
            # Last-resort: avoid iterating unknown string-likes into chars
            # If 'input' is an arbitrary iterable, DO NOT 'list(input)' blindly.
            # Convert to string (single doc) instead of exploding into characters.
            docs = [str(input)]

        print(
            "EF sanity — num_docs:", len(docs),
            "top5_lens_preclamp:", sorted((len(d) for d in docs), reverse=True)[:5]
        )

        MAX_CHARS = 2000  # or 1500
        docs = [d if len(d) <= MAX_CHARS else d[:MAX_CHARS] for d in docs]

        print(
            "EF sanity — top5_lens_postclamp:",
            sorted((len(d) for d in docs), reverse=True)[:5]
        )
        print("EF sanity — batch_size:", self.batch_size)

        print("EF num_docs:", len(docs),
        "top5_lens:", sorted((len(d) for d in docs), reverse=True)[:5])

        # Compute embeddings
        with torch.no_grad():
            vectors = self.model.encode(
                docs,
                batch_size=self.batch_size,
                convert_to_numpy=True, #already moves the vector to cpu 
                normalize_embeddings=self.normalize, #QWEN expects normalized embeddings
                show_progress_bar=False,
            )

        return vectors.tolist()  # type: ignore[return-value], expects a python list of lists (float)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "QwenEmbeddingFunction_600M":
        
        kw = dict(config.get("model_kwargs") or {}) #convert dtype arg to torch dtype from str
        if isinstance(kw.get("torch_dtype"), str):
            kw["torch_dtype"] = getattr(torch, kw["torch_dtype"])

        return QwenEmbeddingFunction_600M(
            model_name=config.get("model_name", "./models/embedding/qwen-emb-600M"),
            device=config.get("device"),
            batch_size=config.get("batch_size", 16),
            normalize=config.get("normalize", True),
            model_kwargs=kw #already converted config model kwargs to torch dtype, stored as kw
        )

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable config so Chroma can persist this EF."""

        safe_kwargs = dict(self.model_kwargs) #use safekwards so the saving of kwargs doesnt cause multi downstream errors trying to save a torch.dtype
        if isinstance(safe_kwargs.get("torch_dtype"), torch.dtype):
            safe_kwargs["torch_dtype"] = str(safe_kwargs["torch_dtype"]).replace("torch.", "")

        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "model_kwargs": safe_kwargs
        }


'''
def main():
    ef = QwenEmbeddingFunction()
    sample = [
        "Qwen models are strong, multilingual embedding models.",
        "Chroma uses EmbeddingFunction interfaces to compute vectors.",
    ]
    embs = ef(sample)
    print(len(embs), len(embs[0]))

if __name__ == "__main__":
    main()
'''