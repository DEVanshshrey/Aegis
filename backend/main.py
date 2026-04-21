from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import uuid
import tempfile
import uvicorn
from dotenv import load_dotenv

from src.pipeline import ClarityCounselPipeline
from src.models import DocumentAnalysisRequest, DocumentAnalysisResponse

load_dotenv()  # Load environment variables from .env file

app = FastAPI(
    title="Legal Lens - Universal Legal Analysis",
    description="AI-powered legal document analysis for all document types using Vertex AI",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = ClarityCounselPipeline()


@app.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        preserve_entities: bool = True,
        analysis_depth: str = "comprehensive"
):
    """
    Analyze any legal document with AI-powered clause analysis.

    Args:
        file: Legal document (PDF, DOCX, TXT)
        preserve_entities: Whether to anonymize sensitive entities during analysis
        analysis_depth: 'basic', 'comprehensive', or 'detailed'
    """

    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Generate unique session ID for this analysis
        session_id = str(uuid.uuid4())

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Process document through AI pipeline
        result = await pipeline.process_document(
            file_path=tmp_file_path,
            document_name=file.filename,
            session_id=session_id,
            preserve_entities=preserve_entities,
            analysis_depth=analysis_depth
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, tmp_file_path)

        return result

    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass

        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/analyze-text", response_model=DocumentAnalysisResponse)
async def analyze_text(request: DocumentAnalysisRequest):
    """Analyze raw text with AI-powered clause analysis."""

    try:
        session_id = str(uuid.uuid4())

        result = await pipeline.process_text(
            text=request.text,
            document_name=request.document_name or "Text Document",
            session_id=session_id,
            preserve_entities=request.preserve_entities,
            analysis_depth=request.analysis_depth
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Legal Lens v2.0",
        "ai_backend": "Google Vertex AI"
    }


@app.get("/supported-entities")
async def get_supported_entities():
    """Get list of entity types that can be anonymized."""
    return {
        "person_entities": ["PERSON", "ORGANIZATION", "WORK_OF_ART"],
        "location_entities": ["LOCATION", "ADDRESS"],
        "financial_entities": ["PRICE", "NUMBER", "PERCENT"],
        "date_entities": ["DATE", "TIME"],
        "contact_entities": ["PHONE_NUMBER", "EMAIL"],
        "legal_entities": ["CONTRACT_NUMBER", "CASE_NUMBER", "LICENSE"]
    }


async def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files."""
    try:
        os.unlink(file_path)
    except:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)