import os
import fitz                          # PyMuPDF
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI()

# --- connect to ChromaDB (persists to disk) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="arxiv_papers",
    metadata={"hnsw:space": "cosine"}   # cosine similarity for retrieval
)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull raw text from a PDF, page by page."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    We use words (not tokens) as a proxy — 512 words ≈ 600-700 tokens,
    which safely fits in the embedding model's 8191 token limit.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # step forward but keep overlap
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API — returns a vector per text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def ingest_pdf(pdf_path: str, paper_title: str = ""):
    """Full pipeline: PDF → chunks → embeddings → ChromaDB."""
    print(f"Processing: {pdf_path}")

    # Step 1: extract text
    text = extract_text_from_pdf(pdf_path)

    # Step 2: chunk it
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks created")

    # Step 3: embed in batches (API has a limit per request)
    batch_size = 100
    all_embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="  Embedding"):
        batch = chunks[i : i + batch_size]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)

    # Step 4: store in ChromaDB
    ids = [f"{os.path.basename(pdf_path)}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "title": paper_title, "chunk_index": i}
                 for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"  → Stored {len(chunks)} chunks in ChromaDB")

def ingest_all_pdfs(pdf_dir: str = "./data/pdfs"):
    """Ingest every PDF in the folder."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in data/pdfs/ — download some first.")
        return
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        ingest_pdf(pdf_path, paper_title=pdf_file.replace(".pdf", ""))

if __name__ == "__main__":
    ingest_all_pdfs()