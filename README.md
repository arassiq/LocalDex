# RAG System Notes

#### Embeddings
* Qwen-emb 8B: too large for practical use.
* Qwen-emb 0.6B: current choice for both embedding + chunking.
* Integrated into a custom Chroma EmbeddingFunction (/models/embedding/inProcess_EF.py).
* Model weights live under /embedding/.
* Scripts for running embeddings are in /embedding/.
* Status: in-process EF ✅, remote EF ❌ (to do).

#### Chunking
* Using ClusterSemanticChunker from langchain_experimental.
* Evaluating chunking algorithms with Qwen embeddings; best-fit logic will move into /models/chunking/.
* Test scripts: /collections/chunkingtest.py.
* External reference repo: chunking_evaluation (not used directly).
* Reminder: install langchain-experimental via pip (to be added to pyproject).

#### Algorithms
* Default: SemanticChunker (O(n·w) complexity).
* Structure-aware splitters (functions/classes) + semantic merge in development.
* TreeSplitter: applied specifically for code chunking.
* Key call: langchain_experimental.text_splitter.py, line 203 → embed func comes from our custom Chroma EF.

#### Search
* Planned searchmg: aggregator layer (potentially Docker-based).

#### Memory / State
* Redis DB: stores the last 10 chats for session context.




## Personal Notes (messy)

* ~~Qwen-emb 8b~~ Way too big -> going to 0.6B qwen-emb for embedding and chunking
-> for embedding, written into a special class in chroma 
/models/embedding/ -> model download
/embedding/ -> embedding scripts

-> SDPA is blowing up embedding model in MPS, switching to eager for predictable memory. S

inProcess EF - DONE

server, remote EF - TO DO 

* Chunking
  
-> use of clustersemanticchunker:
  
testing chunking algorithms with qwen model then using best fit 
/models/chunking -> referenced chunking repo, empty, using langchain_experemental for chunking strategy
/collections/chunkingtest.py -> chunking test scripts
Not using ->https://github.com/brandonstarxel/chunking_evaluation -- DONT FORGET TO INSTALL VIA PIP IN COMMAND LINE, WILL BE ADDED TO pyproject

O(n·w) ->semanticChunker (default chunker)

code -> structure-aware splitter (functions/classes) -> semantic merge.

langchain_experemental.text_splitter.py -> line 203 > embed func to __call__ (native to our chroma class)

searchmg -> search engine aggregator -> docker based?