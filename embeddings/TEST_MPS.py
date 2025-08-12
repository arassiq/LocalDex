from sentence_transformers import SentenceTransformer
import torch
import time

torch.device("mps")

model = SentenceTransformer(
    "/Users/aaronrassiq/Desktop/LocalLLM-RAG/qwen-emb",
    device= "mps",
    model_kwargs={"torch_dtype": torch.float16} #TODO: GOOD 
)
#    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"}, #NO FLASH ATTENTION ON MPS 
#    tokenizer_kwargs={"padding_side": "left"},

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

start = time.time()
# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity, similarity.size(), similarity.dtype())
print(f"Time Taken: {time.time() - start}")
# tensor([[0.7493, 0.0751], 
#         [0.0880, 0.6318]])
