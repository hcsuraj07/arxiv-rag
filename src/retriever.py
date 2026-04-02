import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="arxiv_papers",
    metadata={"hnsw:space": "cosine"}
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def embed_query(query: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",  
        input=query
    )
    return response.data[0].embedding   


def vector_search(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_embedding],  
        n_results=top_k,                     
        include=["documents", "metadatas", "distances"]
    )

    # zip the results together into a clean list of dicts
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],   
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,           
            "source": meta["source"],      
            "title": meta["title"],       
            "chunk_index": meta["chunk_index"],  
            "distance": dist       
        })
    return chunks



def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    # build pairs of [query, chunk_text] for the cross-encoder
    pairs = [[query, chunk["text"]] for chunk in chunks]

    # score every pair — returns a numpy array of floats
    scores = reranker.predict(pairs)

    # attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # sort by score descending and return top_n
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


def retrieve(query: str) -> list[dict]:
    print(f"\nQuery: {query}")
    query_embedding = embed_query(query)

    candidates = vector_search(query_embedding, top_k=20)
    print(f"  → Retrieved {len(candidates)} candidates from ChromaDB")

    top_chunks = rerank(query, candidates, top_n=5)
    print(f"  → Reranked to top {len(top_chunks)} chunks")

    
    for i, chunk in enumerate(top_chunks):
        print(f"     {i+1}. [{chunk['rerank_score']:.2f}] {chunk['title']} "
              f"(chunk {chunk['chunk_index']})")

    return top_chunks


if __name__ == "__main__":
    test_query = "What methods are used to evaluate RAG systems?"
    results = retrieve(test_query)

    print("\n--- Top chunks ---")
    for i, chunk in enumerate(results):
        print(f"\n[{i+1}] {chunk['title']} (score: {chunk['rerank_score']:.2f})")
        print(chunk['text'][:300] + "...")  # print first 300 chars of each chunk
