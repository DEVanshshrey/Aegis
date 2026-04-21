import PyPDF2
from docx import Document
from pathlib import Path
import re


class DocumentIngester:
    """Enhanced document ingester for various file formats."""

    def read_document(self, file_path: str) -> str:
        """Read document content based on file extension."""
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

    def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file with enhanced processing."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()

                    # Clean up common PDF extraction issues
                    page_text = self._clean_pdf_text(page_text)
                    text += page_text + "\n"

                return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def _read_docx(self, file_path: str) -> str:
        """Extract text from DOCX file including tables and formatting."""
        try:
            doc = Document(file_path)
            text = ""

            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    text += row_text + "\n"

            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")

    def _read_txt(self, file_path: str) -> str:
        """Read plain text file with encoding detection."""
        try:
            # Try UTF-8 first
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read().strip()
            except UnicodeDecodeError:
                # Fallback to other encodings
                for encoding in ['latin1', 'cp1252', 'ascii']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            return file.read().strip()
                    except UnicodeDecodeError:
                        continue

                # If all encodings fail, read as binary and decode with errors ignored
                with open(file_path, 'rb') as file:
                    return file.read().decode('utf-8', errors='ignore').strip()

        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")

    def _clean_pdf_text(self, text: str) -> str:
        """Clean common PDF text extraction artifacts."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)  # Add space after period and capital letter

        # Remove page headers/footers patterns
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'\d+\s*', '', text, flags=re.MULTILINE)

        return text.strip()