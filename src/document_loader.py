import fitz
import pytesseract
from PIL import Image
import io
from pathlib import Path

# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

PDF_FOLDER = Path("data/raw")

documents = []

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF files.\n")


for pdf_path in pdf_files:

    print("=" * 60)
    print(f"Processing: {pdf_path.name}")

    doc = fitz.open(pdf_path)

    normal_pages = 0
    ocr_pages = 0
    failed_pages = 0

    for page_number, page in enumerate(doc, start=1):

        # Try normal extraction first
        normal_text = page.get_text("text").strip()

        # Use normal extraction if meaningful text exists
        if len(normal_text) >= 10:

            text = normal_text
            extraction_type = "normal"
            normal_pages += 1

            print(
                f"Page {page_number}: "
                f"Normal extraction ({len(text)} characters)"
            )

        else:

            print(
                f"Page {page_number}: "
                f"Running OCR..."
            )

            # Render PDF page at higher resolution
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            image_bytes = pix.tobytes("png")

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            # OCR
            ocr_text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

            # If OCR found useful text, use it
            if ocr_text:

                text = ocr_text
                extraction_type = "ocr"
                ocr_pages += 1

                print(
                    f"             OCR extracted "
                    f"{len(text)} characters"
                )

            # If OCR failed but normal extraction had something,
            # keep the original text
            elif normal_text:

                text = normal_text
                extraction_type = "normal"
                normal_pages += 1

            else:

                text = ""
                extraction_type = "failed"
                failed_pages += 1

                print("             No text extracted")

        # Save page
        if text:

            documents.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                    "extraction_type": extraction_type
                }
            )

    doc.close()

    # Summary for this PDF
    print("\nPDF SUMMARY")
    print(f"Total pages  : {normal_pages + ocr_pages + failed_pages}")
    print(f"Normal pages : {normal_pages}")
    print(f"OCR pages    : {ocr_pages}")
    print(f"Failed pages : {failed_pages}")
    print()


print("=" * 60)
print("ALL PDFs COMPLETED")
print(f"Total pages containing text: {len(documents)}")

import json

output_path = "data/processed/extracted_documents.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=4
    )

print(f"\nSaved extracted documents to: {output_path}")