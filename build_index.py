from sentence_transformers import SentenceTransformer
import faiss, pickle

# 1) Load and chunk your data
docs = []
with open("my_faq.txt", "r", encoding="utf-8") as f:
    for para in f.read().split("\n\n"):
        text = para.strip()
        if text:
            docs.append(text)

# 2) Embed all chunks
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, convert_to_numpy=True)

# 3) Build the FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 4) Save for reuse
with open("faiss_index.pkl", "wb") as f:
    pickle.dump((index, docs), f)

print(f"Built index for {len(docs)} chunks ✓")
