import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()


def query_index(index_dir='rag/index', query:str='', k:int=5):
    embeddings = OpenAIEmbeddings()
    store = FAISS.load_local(index_dir, embeddings)
    results = store.similarity_search(query, k=k)
    return results


if __name__ == '__main__':
    res = query_index(query='Which tech stocks look bullish?', k=5)
    for r in res:
        print(r.page_content[:400])
