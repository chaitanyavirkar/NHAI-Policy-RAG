import json
import re

INPUT_FILE = "data/processed/extracted_documents.json"
OUTPUT_FILE = "data/processed/cleaned_documents.json"


def clean_text(text):

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


cleaned_documents = []

for document in documents:

    cleaned_text = clean_text(document["text"])

    if cleaned_text:

        cleaned_documents.append(
            {
                "source": document["source"],
                "page": document["page"],
                "text": cleaned_text,
                "extraction_type": document["extraction_type"]
            }
        )


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        cleaned_documents,
        f,
        ensure_ascii=False,
        indent=4
    )


print("Cleaning completed.")
print(f"Original pages: {len(documents)}")
print(f"Cleaned pages : {len(cleaned_documents)}")
print(f"Saved to      : {OUTPUT_FILE}")