import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/chunked_documents.json"
OUTPUT_EMBEDDINGS = "data/processed/embeddings.npy"
OUTPUT_METADATA = "data/processed/embedding_metadata.json"

# Small and good model for starting
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Total chunks loaded: {len(documents)}")

# Get only chunk text
texts = [doc["text"] for doc in documents]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Save embeddings
np.save(OUTPUT_EMBEDDINGS, embeddings)

# Save metadata separately
metadata = []

for doc in documents:
    metadata.append(
        {
            "source": doc["source"],
            "page": doc["page"],
            "chunk_id": doc["chunk_id"],
            "text": doc["text"],
            "extraction_type": doc["extraction_type"]
        }
    )

with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nEmbedding generation completed.")
print(f"Number of embeddings : {len(embeddings)}")
print(f"Vector dimensions    : {embeddings.shape[1]}")
print(f"Embeddings saved to  : {OUTPUT_EMBEDDINGS}")
print(f"Metadata saved to    : {OUTPUT_METADATA}")