## models used

* ~~Qwen-emb 8b~~ Way too big -> going to 0.6B qwen-emb for embedding and chunking
-> for embedding, written into a special class in chroma 
/models/embedding/ -> model download
/embedding/ -> embedding scripts

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