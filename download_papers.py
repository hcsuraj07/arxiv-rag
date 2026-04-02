import arxiv
import os

search = arxiv.Search(
    query="large language models retrieval augmented generation",
    max_results=10,
    sort_by=arxiv.SortCriterion.Relevance
)

os.makedirs("data/pdfs", exist_ok=True)

for paper in search.results():
    print(f"Downloading: {paper.title}")
    paper.download_pdf(dirpath="data/pdfs", filename=f"{paper.get_short_id()}.pdf")

print("Done.")