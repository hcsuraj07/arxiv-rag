from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    # --- SYSTEM MESSAGE ---
    # Sets the model's role and citation rules
    system_message = """You are a research assistant that answers questions 
about academic papers.

RULES you must follow:
1. Answer ONLY using the provided source chunks below. 
2. If the chunks don't contain enough information, say "I don't have enough 
   information in the retrieved chunks to answer this fully."
3. Always cite your sources using [SOURCE N] notation inline in your answer.
4. At the end of your answer, list the papers you cited under "Sources:".
5. Never make up facts, authors, or results not present in the chunks.
6. Be concise and precise — this is a technical research context."""
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"""
[SOURCE {i+1}]
Paper: {chunk['title']}
Relevance score: {chunk['rerank_score']:.2f}
Content:
{chunk['text']}
---"""

    user_message = f"""Here are the most relevant chunks retrieved from the paper database:

{context_block}

Based ONLY on the sources above, answer this question:
{question}"""

    # Return in OpenAI's expected message format
    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message}
    ]


def generate(question: str, chunks: list[dict]) -> dict:
    # build the structured prompt
    messages = build_prompt(question, chunks)

    print(f"\n  Sending to GPT-4o with {len(chunks)} chunks as context...")

    # call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,        
        max_tokens=1000       # cap response length
    )

    # extract the answer text from the response object
    answer = response.choices[0].message.content

    # extract token usage — useful for tracking API costs
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        # rough cost estimate: $5 per 1M input tokens, $15 per 1M output tokens
        "estimated_cost_usd": round(
            (response.usage.prompt_tokens * 0.000005) +
            (response.usage.completion_tokens * 0.000015), 5
        )
    }

    return {
        "question": question,
        "answer": answer,
        "sources": [{"title": c["title"], "score": c["rerank_score"]}
                    for c in chunks],
        "usage": usage
    }

if __name__ == "__main__":
    from retriever import retrieve

    question = "What methods are used to evaluate RAG systems?"

    print("Retrieving relevant chunks...")
    chunks = retrieve(question)

    print("Generating answer...")
    result = generate(question, chunks)

    print("\n" + "="*60)
    print("QUESTION:", result["question"])
    print("="*60)
    print("\nANSWER:\n", result["answer"])
    print("\n" + "="*60)
    print("SOURCES USED:")
    for i, source in enumerate(result["sources"]):
        print(f"  {i+1}. {source['title']} (score: {source['score']:.2f})")
    print("\nTOKEN USAGE:", result["usage"])