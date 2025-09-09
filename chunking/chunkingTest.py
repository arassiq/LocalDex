import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chunking import noteChunker, codeChunker


def testCode():
    codeChunk = codeChunker()

    path = "/Users/aaronrassiq/Desktop/LocalDex/embeddings/inProcess_EF.py"
    recs = codeChunk.__call__(path)
    for rec in recs:
        print(rec,"\n\n\n\n\n")

def testText():
    noteChunk = noteChunker()

    path = "/Users/aaronrassiq/Desktop/LocalDex/chunking/text_samples/the_adventures_of_sherlock_holmes.txt" 
    recs = noteChunk.__call__(path)
    for rec in recs:
        print(rec,"\n\n\n\n\n")

def main():
    testText()

if __name__ == "__main__":
    main()