import os
import numpy as np
import faiss

EMBEDDINGS_FILE = "data/processed/embeddings.npy"

VECTOR_STORE_FOLDER = "data/vector_store"
INDEX_FILE = "data/vector_store/nhai_policy.index"


# 1. Create vector store folder if it doesn't exist
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)


# 2. Load our saved embeddings
print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

# FAISS expects float32
embeddings = embeddings.astype("float32")

print(f"Embeddings shape: {embeddings.shape}")


# 3. Get vector dimension
dimension = embeddings.shape[1]

print(f"Vector dimension: {dimension}")


# 4. Create FAISS index
# Inner Product works as cosine similarity
# because our embeddings were normalized earlier
index = faiss.IndexFlatIP(dimension)


# 5. Add embeddings to FAISS
index.add(embeddings)

print(f"Vectors stored in FAISS: {index.ntotal}")


# 6. Save FAISS index
faiss.write_index(index, INDEX_FILE)

print(f"\nFAISS index saved to:")
print(INDEX_FILE)

print("\nVector database created successfully!")