import json
from src.ingest import DocumentIngester

def run_sanity_checks():
    ingester = DocumentIngester()
    
    # Define your test files here
    test_files = [
        "Assets/Text_pdf.pdf",  # Tests pypdf and pagination
        "Assets/OCR_pdf.pdf",    # Tests the Tesseract OCR fallback
        "Assets/Doc.docx",    # Tests sequential DOCX parsing
        "Assets/Text.txt"      # Tests encoding fallbacks
    ]

    for file_path in test_files:
        print(f"\n{'='*50}")
        print(f"TESTING FILE: {file_path}")
        print(f"{'='*50}")
        
        try:
            # Run the ingester
            result = ingester.read_document(file_path)
            
            # Print the result nicely formatted as JSON
            # We slice the content to 200 chars so it doesn't flood your terminal
            for page in result:
                snippet = page['content'][:200].replace('\n', ' \\n ')
                print(f"Page {page['page']} | Length: {len(page['content'])} chars")
                print(f"Preview: {snippet}...\n")
                
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")

if __name__ == "__main__":
    run_sanity_checks()