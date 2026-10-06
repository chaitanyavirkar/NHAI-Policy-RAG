import json

INPUT_FILE = "data/processed/cleaned_documents.json"
OUTPUT_FILE = "data/processed/chunked_documents.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def create_chunks(text, chunk_size, overlap):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# Load cleaned documents
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


chunked_documents = []


for document in documents:

    text = document["text"]

    chunks = create_chunks(
        text,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    for chunk_number, chunk in enumerate(chunks, start=1):

        chunked_documents.append(
            {
                "source": document["source"],
                "page": document["page"],
                "chunk_id": chunk_number,
                "text": chunk,
                "extraction_type": document["extraction_type"]
            }
        )


# Save chunks
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        chunked_documents,
        f,
        ensure_ascii=False,
        indent=4
    )


print("Chunking completed.")

print(f"Original pages : {len(documents)}")
print(f"Total chunks   : {len(chunked_documents)}")

print(f"Chunk size     : {CHUNK_SIZE}")
print(f"Chunk overlap  : {CHUNK_OVERLAP}")

print(f"Saved to       : {OUTPUT_FILE}")
