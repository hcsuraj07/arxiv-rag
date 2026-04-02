import os
import json
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retriever import retrieve
from generator import generate

load_dotenv()
client = OpenAI()

def generate_test_questions(n: int = 10) -> list[dict]:
    import chromadb

    print(f"Generating {n} test questions from corpus...")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="arxiv_papers",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = collection.get(include=["documents", "metadatas"])
    total = len(all_chunks["documents"])
    indices = random.sample(range(total), min(n, total))

    questions = []
    for i, idx in enumerate(indices):
        chunk_text = all_chunks["documents"][idx]
        paper_title = all_chunks["metadatas"][idx]["title"]

        # skip very short or very long chunks — they make bad questions
        word_count = len(chunk_text.split())
        if word_count < 50 or word_count > 400:
            chunk_text = " ".join(chunk_text.split()[:400])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate one specific, answerable question from the text. Return ONLY the question, nothing else."
                },
                {
                    "role": "user",
                    "content": f"Text:\n{chunk_text[:1500]}"  # cap at 1500 chars
                }
            ],
            temperature=0.7,
            max_tokens=80
        )

        question = response.choices[0].message.content.strip()
        questions.append({
            "question": question,
            "source_chunk": chunk_text[:1500],
            "source_paper": paper_title
        })
        print(f"  ✓ [{i+1}/{n}] {question[:80]}...")

    return questions

def score_single_result(question: str, answer: str,
                         contexts: list[str], ground_truth: str) -> dict:

    # truncate contexts to avoid token limits
    context_text = "\n---\n".join([c[:500] for c in contexts[:3]])
    ground_truth_short = ground_truth[:500]

    prompt = f"""You are evaluating a RAG system. Score each metric from 0.0 to 1.0.

QUESTION: {question}

RETRIEVED CONTEXTS:
{context_text}

GROUND TRUTH (source chunk the question came from):
{ground_truth_short}

ANSWER GENERATED:
{answer[:800]}

Score each metric and return ONLY valid JSON, no explanation:
{{
  "faithfulness": <0.0-1.0, is every claim in the answer supported by the contexts?>,
  "answer_relevancy": <0.0-1.0, does the answer actually address the question?>,
  "context_precision": <0.0-1.0, are the retrieved contexts relevant to the question?>,
  "context_recall": <0.0-1.0, do the contexts contain enough info to answer the question?>
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        scores = json.loads(raw)
        # ensure all values are floats between 0 and 1
        return {k: max(0.0, min(1.0, float(v))) for k, v in scores.items()}
    except Exception:
        print(f"    Warning: could not parse scores, using 0.5 defaults")
        return {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5
        }

def run_pipeline_on_questions(questions: list[dict]) -> list[dict]:
    print(f"\nRunning {len(questions)} questions through RAG pipeline...")

    results = []
    for i, q in enumerate(questions):
        print(f"\n  [{i+1}/{len(questions)}] {q['question'][:70]}...")

        chunks = retrieve(q["question"])
        result = generate(q["question"], chunks)

        retrieved_contexts = [chunk["text"] for chunk in chunks]

        results.append({
            "question": q["question"],
            "answer": result["answer"],
            "contexts": retrieved_contexts,
            "ground_truth": q["source_chunk"],
        })

    return results


def run_evaluation(results: list[dict]) -> list[dict]:
    print(f"\nScoring {len(results)} results with GPT-4o-mini judge...")

    scored = []
    for i, r in enumerate(results):
        print(f"  Scoring [{i+1}/{len(results)}]...")
        scores = score_single_result(
            question=r["question"],
            answer=r["answer"],
            contexts=r["contexts"],
            ground_truth=r["ground_truth"]
        )
        scored.append({**r, **scores})

    return scored

def save_results(scored: list[dict]):
    os.makedirs("notebooks", exist_ok=True)

    df = pd.DataFrame(scored)
    df.to_csv("notebooks/eval_results.csv", index=False)
    print("\nSaved → notebooks/eval_results.csv")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {m: round(float(df[m].mean()), 3) for m in metrics}

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("notebooks/eval_summary.csv", index=False)

    print("\n" + "="*50)
    print("RAGAS-STYLE EVALUATION RESULTS")
    print("="*50)
    print(f"  Faithfulness:      {summary['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {summary['answer_relevancy']:.3f}")
    print(f"  Context Precision: {summary['context_precision']:.3f}")
    print(f"  Context Recall:    {summary['context_recall']:.3f}")
    print(f"  Overall Average:   {round(sum(summary.values())/4, 3):.3f}")
    print("="*50)
    print("\nSaved summary → notebooks/eval_summary.csv")

    return summary


if __name__ == "__main__":
    # generate test questions
    questions = generate_test_questions(n=10)

    with open("notebooks/test_questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    print(f"\nSaved {len(questions)} questions → notebooks/test_questions.json")

    # run pipeline
    results = run_pipeline_on_questions(questions)

    # score results
    scored = run_evaluation(results)

    # save and print
    summary = save_results(scored)

    print("\nAdd this table to your README:")
    print("\n| Metric | Score |")
    print("|--------|-------|")
    for k, v in summary.items():
        print(f"| {k.replace('_', ' ').title()} | {v} |")