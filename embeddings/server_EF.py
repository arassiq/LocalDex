import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flask
import pydantic
from embeddings import QwenEmbeddingFunction
import uvicorn

from fastapi import FastAPI, Request, Header, Body
from fastapi.middleware.cors import CORSMiddleware

import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ef = QwenEmbeddingFunction(
    model_name="./models/embedding/qwen-emb",
    device="mps",
    model_kwargs={"torch_dtype": torch.float16}
)

@app.post("/embed")
def embedText(text):
    result = ef.__call__(text)
    array_bytes = result.tobytes()

    return array_bytes

def run():
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
    print(f"Starting server at http://0.0.0.0:{port}")

if __name__ == "__main__":
    run()

