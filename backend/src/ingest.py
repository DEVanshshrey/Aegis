import pypdf
from docx import Document
from pathlib import Path
import re
import logging
import pytesseract
from pdf2image import convert_from_path

# Set up basic logging so failed documents don't crash the whole pipeline silently
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIngester:
    """Enhanced document ingester for legal pipelines."""

    def read_document(self, file_path: str) -> list[dict]:
        """
        Read document content and return structured data with pagination.
        Returns: list of dicts -> [{'page': int, 'content': str}]
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == '.pdf':
            return self._read_pdf(file_path)
        elif extension == '.docx':
            return self._read_docx(file_path)
        elif extension == '.txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _read_pdf(self, file_path: str) -> list[dict]:
        """Extract text from PDF with pagination tracking and OCR fallback."""
        pages_data = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    # Attempt native extraction first
                    page_text = page.extract_text() or ""
                    
                    if page_text.strip():
                        page_text = self._clean_pdf_text(page_text)
                    
                    # 1. OCR FALLBACK TRIGGER
                    # If pypdf extracts nothing, it's likely an image-based scanned page
                    if not page_text.strip():
                        logger.info(f"Page {page_num + 1} appears empty. Attempting OCR...")
                        page_text = self._perform_ocr(file_path, page_num)
                    
                    # 2. AUDIT TRAIL TRACKING
                    # Append the text alongside its specific page number
                    if page_text.strip():
                        pages_data.append({
                            "page": page_num + 1,
                            "content": page_text.strip()
                        })

                return pages_data
                
        except Exception as e:
            logger.error(f"Failed to process PDF: {file_path}", exc_info=True)
            raise Exception(f"Error reading PDF: {file_path}") from e

    def _perform_ocr(self, file_path: str, page_num: int) -> str:
        """Helper method to isolate and OCR a specific PDF page."""
        try:
            # pdf2image uses 1-based indexing for page numbers
            images = convert_from_path(
                file_path, 
                first_page=page_num + 1, 
                last_page=page_num + 1
            )
            
            if images:
                # Run tesseract on the converted image
                ocr_text = pytesseract.image_to_string(images[0])
                return self._clean_pdf_text(ocr_text)
            return ""
            
        except Exception as e:
            logger.warning(f"OCR failed on page {page_num + 1} of {file_path}: {str(e)}")
            return ""

    def _read_docx(self, file_path: str) -> list[dict]:
        """Extract text sequentially from DOCX."""
        try:
            doc = Document(file_path)
            text = ""
            
            # Iterate through elements in their actual document order
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Paragraph
                    from docx.text.paragraph import Paragraph
                    text += Paragraph(element, doc).text + "\n"
                elif element.tag.endswith('tbl'):  # Table
                    from docx.table import Table
                    table = Table(element, doc)
                    for row in table.rows:
                        row_text = " | ".join(cell.text for cell in row.cells)
                        text += row_text + "\n"
                        
            # DOCX lacks strict page bounds, so we return it as a single block
            return [{"page": 1, "content": text.strip()}]
            
        except Exception as e:
            logger.error(f"Failed to process DOCX: {file_path}", exc_info=True)
            raise Exception(f"Error reading DOCX: {file_path}") from e
        
    def _read_txt(self, file_path: str) -> list[dict]:
        """Read plain text file with encoding detection."""
        try:
            content = ""
            # Try UTF-8 first
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
            except UnicodeDecodeError:
                # Fallback to other encodings
                for encoding in ['latin1', 'cp1252', 'ascii']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read().strip()
                            break
                    except UnicodeDecodeError:
                        continue
                
                if not content:
                    # If all encodings fail, read as binary and decode with errors ignored
                    with open(file_path, 'rb') as file:
                        content = file.read().decode('utf-8', errors='ignore').strip()

            # TXT files don't have pages, so we return it as a single block
            return [{"page": 1, "content": content}]

        except Exception as e:
            logger.error(f"Failed to process TXT: {file_path}", exc_info=True)
            raise Exception(f"Error reading TXT: {file_path}") from e

    def _clean_pdf_text(self, text: str) -> str:
        """Clean common PDF text extraction artifacts safely for legal documents."""
        # Remove excessive horizontal whitespace, preserving newlines
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove standalone page numbers on their own lines
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) 
        
        return text.strip()