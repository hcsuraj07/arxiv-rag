from retriever import retrieve
from generator import generate

def run_query(question: str) -> dict:
    chunks = retrieve(question)
    result = generate(question, chunks)
    return result

if __name__ == "__main__":
    questions = [
        "What methods are used to evaluate RAG systems?",
        "What are the main challenges in retrieval augmented generation?",
    ]
    for question in questions:
        print("\n" + "="*60)
        result = run_query(question)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:400]}...")
        print(f"Cost: ${result['usage']['estimated_cost_usd']}")