from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime

class DocumentAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Legal document text to analyze")
    document_name: Optional[str] = Field(None, description="Name of the document")
    preserve_entities: bool = Field(True, description="Whether to anonymize sensitive entities")
    analysis_depth: Literal["basic", "comprehensive", "detailed"] = Field("comprehensive")

class EntityMapping(BaseModel):
    """Model for entity anonymization mapping."""
    original: str
    anonymized: str
    entity_type: str
    confidence: float

class ClauseAnalysis(BaseModel):
    """Model for individual clause analysis."""
    clause_id: str
    original_clause: str
    anonymized_clause: Optional[str]
    severity: Literal["🔴", "🟡", "🟢"]
    risk_score: float = Field(..., ge=0, le=1, description="Risk score from 0 to 1")
    category: str = Field(..., description="Legal category of the clause")
    explanation: Optional[str]
    recommendations: List[str] = Field(default_factory=list)
    legal_implications: Optional[str]
    entities_found: List[EntityMapping] = Field(default_factory=list)

class DocumentSummary(BaseModel):
    """High-level document analysis summary."""
    document_type: str
    total_clauses: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    overall_risk_score: float
    key_concerns: List[str]
    document_category: str

class DocumentAnalysisResponse(BaseModel):
    """Complete document analysis response."""
    session_id: str
    document: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    summary: DocumentSummary
    clauses: List[ClauseAnalysis]
    entity_mappings: Dict[str, EntityMapping] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)