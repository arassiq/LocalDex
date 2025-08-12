import chromadb
from chromadb.config import Settings
from embeddings.textEmbedding import QwenEmbeddingFunction

PERSISTENT_DIR = "path/to/dir"

class collectionsManagement:
    def __init__(self, collectionName):
        self.collection = "hai"

        #collection = client.get_or_create_collection(
        #                                        name="my-collection",
        #                                        metadata={"description": "..."}
        #                                    )


'''
1. init - collectionName
2. createcollection - name, embeddingmodel, metaData
3. 
'''