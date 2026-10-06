import json
import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "data/vector_store/nhai_policy.index"
METADATA_FILE = "data/processed/embedding_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 8
MIN_SCORE = 0.45


print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print(f"Vectors in FAISS: {index.ntotal}")


with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Metadata records: {len(metadata)}")


def retrieve(query, top_k=8, min_score=0.45):

    # Convert user question into embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # FAISS expects float32
    query_embedding = query_embedding.astype("float32")

    # Retrieve more candidates so duplicates can be removed
    candidate_k = min(top_k * 3, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    results = []
    seen_texts = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        # Ignore weak results
        if float(score) < min_score:
            continue

        text = metadata[idx]["text"]

        # Normalize text for duplicate detection
        normalized_text = " ".join(
            text.lower().split()
        )

        # Simple duplicate fingerprint
        fingerprint = normalized_text[:300]

        if fingerprint in seen_texts:
            continue

        seen_texts.add(fingerprint)

        results.append(
            {
                "score": float(score),
                "source": metadata[idx]["source"],
                "page": metadata[idx]["page"],
                "chunk_id": metadata[idx]["chunk_id"],
                "text": text
            }
        )

        # Stop after collecting enough good results
        if len(results) >= top_k:
            break

    return results


# Test retriever directly
if __name__ == "__main__":

    query = input("\nAsk a question about NHAI policies: ")

    results = retrieve(
        query,
        top_k=TOP_K,
        min_score=MIN_SCORE
    )

    print("\n" + "=" * 70)
    print(f"TOP {len(results)} RESULTS FROM FAISS")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Score  : {result['score']:.4f}")
        print(f"Source : {result['source']}")
        print(f"Page   : {result['page']}")
        print(f"Chunk  : {result['chunk_id']}")

        print("\nText:")
        print(result["text"][:700])

        print("\n" + "-" * 70)