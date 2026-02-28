import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1. ADD ALL YOUR SOURCES HERE
# ----------------------------

sources = [
    # Daily Nexus
    {
        "title": "Swan 2025 - Avoiding AI",
        "url": "https://dailynexus.com/2025-07-03/could-avoiding-ai-prepare-you-for-an-ai-integrated-world/"
    },
    {
        "title": "Giant 2025 - Replace Students with AI",
        "url": "https://dailynexus.com/2025-07-29/ucsb-to-replace-one-third-of-students-with-ai/"
    },

    # DOAJ (Reinke et al.)
    {
        "title": "Reinke et al. 2025",
        "url": "https://doaj.org/article/7178f7cf0518403da9c1cb04ccce7a0d"
    },

    # Springer Open (Bond et al.)
    {
        "title": "Bond et al. 2024",
        "url": "https://link.springer.com/article/10.1186/s41239-023-00436-z"
    },

    # arXiv (Kosmyna et al.)
    {
        "title": "Kosmyna et al. 2025",
        "url": "https://arxiv.org/abs/2506.08872"
    },

    # UCSB Policy Pages
    {
        "title": "UCSB AI Guidelines",
        "url": "https://cio.ucsb.edu/artificial-intelligence/ai-use-guidelines"
    },
    {
        "title": "UCSB AI in Classes",
        "url": "https://otl.ucsb.edu/resources/assessing-learning/ai-in-classes"
    }
]

# ----------------------------
# 2. SCRAPER FUNCTION
# ----------------------------

def scrape_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts/styles
    for script in soup(["script", "style", "nav", "footer"]):
        script.extract()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# 3. SCRAPE ALL SOURCES
# ----------------------------

documents = []

print("Scraping sources...\n")

for source in sources:
    try:
        print(f"Scraping: {source['title']}")
        text = scrape_page(source["url"])

        documents.append({
            "title": source["title"],
            "url": source["url"],
            "text": text
        })

    except Exception as e:
        print(f"Failed to scrape {source['url']} -> {e}")

print("\nFinished scraping.")

# ----------------------------
# 4. CHUNK TEXT
# ----------------------------

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
metadata = []

for doc in documents:
    chunks = chunk_text(doc["text"])
    for chunk in chunks:
        all_chunks.append(chunk)
        metadata.append({
            "title": doc["title"],
            "url": doc["url"]
        })

print(f"Total chunks created: {len(all_chunks)}")

# ----------------------------
# 5. EMBEDDINGS
# ----------------------------

print("Generating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# ----------------------------
# 6. BUILD FAISS INDEX
# ----------------------------

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "ethics_index.faiss")

# Save chunks + metadata
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\nKnowledge base built successfully!")