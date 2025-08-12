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


class QwenEmbeddingFunction(EmbeddingFunction[Documents]):

    def __init__(
        self,
        model_name: str = "./models/embedding/qwen-emb",
        device: Union[str, None] = None,
        batch_size: int = 16, #TODO: might create 16 * 32 error in MPS, lower to 8 if memory spikes
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

        if model_kwargs is None:
            model_kwargs = {"torch_dtype": torch.float16}

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        self.device = device
        self.model_kwargs = model_kwargs

        self.model = SentenceTransformer( #load once
            self.model_name, 
            device= self.device, 
            model_kwargs=self.model_kwargs 
        )

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
    def build_from_config(config: Dict[str, Any]) -> "QwenEmbeddingFunction":
        
        kw = dict(config.get("model_kwargs") or {}) #convert dtype arg to torch dtype from str
        if isinstance(kw.get("torch_dtype"), str):
            kw["torch_dtype"] = getattr(torch, kw["torch_dtype"])

        return QwenEmbeddingFunction(
            model_name=config.get("model_name", "./models/embedding/qwen-emb"),
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