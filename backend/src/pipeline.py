from .ingest import DocumentIngester
from .segmenter import UniversalClauseSegmenter
from .entity_anonymizer import EntityAnonymizer
from .vertex_ai_analyzer import VertexAIAnalyzer
from .models import DocumentAnalysisResponse, DocumentSummary, ClauseAnalysis, EntityMapping
import uuid
from typing import Dict, Any, List
from datetime import datetime


class ClarityCounselPipeline:
    """Main pipeline orchestrating the complete AI-powered document analysis."""

    def __init__(self):
        self.ingester = DocumentIngester()
        self.segmenter = UniversalClauseSegmenter()
        self.anonymizer = EntityAnonymizer()
        self.ai_analyzer = VertexAIAnalyzer()

    async def process_document(
            self,
            file_path: str,
            document_name: str,
            session_id: str,
            preserve_entities: bool = True,
            analysis_depth: str = "comprehensive"
    ) -> DocumentAnalysisResponse:
        """Process a document file through the complete AI pipeline."""

        # Step 1: Extract text from document
        original_text = self.ingester.read_document(file_path)

        return await self.process_text(
            text=original_text,
            document_name=document_name,
            session_id=session_id,
            preserve_entities=preserve_entities,
            analysis_depth=analysis_depth
        )

    async def process_text(
            self,
            text: str,
            document_name: str,
            session_id: str,
            preserve_entities: bool = True,
            analysis_depth: str = "comprehensive"
    ) -> DocumentAnalysisResponse:
        """Process raw text through the complete AI analysis pipeline."""

        processing_metadata = {
            "original_text_length": len(text),
            "preserve_entities": preserve_entities,
            "analysis_depth": analysis_depth,
            "processing_steps": []
        }

        try:
            # Step 1: Entity anonymization (if enabled)
            entity_mappings = {}
            anonymized_text = text

            if preserve_entities:
                processing_metadata["processing_steps"].append("entity_anonymization")
                anonymized_text, entity_mappings = self.anonymizer.anonymize_document(text, session_id)
                processing_metadata["entities_anonymized"] = len(entity_mappings)

            # Step 2: Segment document into clauses
            processing_metadata["processing_steps"].append("clause_segmentation")
            clauses = self.segmenter.segment_document(anonymized_text)
            processing_metadata["clauses_extracted"] = len(clauses)

            if not clauses:
                return self._create_error_response(
                    session_id,
                    document_name,
                    "No analyzable clauses found in document",
                    processing_metadata
                )

            # Step 3: AI-powered analysis using Vertex AI
            processing_metadata["processing_steps"].append("vertex_ai_analysis")

            # Detect document type from content
            document_type = self._detect_document_type(text)

            # Analyze with Vertex AI
            ai_analysis = await self.ai_analyzer.analyze_document(
                clauses=clauses,
                document_type=document_type,
                analysis_depth=analysis_depth
            )

            # Step 4: Restore entities in analysis results
            if preserve_entities and entity_mappings:
                processing_metadata["processing_steps"].append("entity_restoration")
                ai_analysis = self._restore_entities_in_analysis(ai_analysis, entity_mappings, clauses)

            # Step 5: Build comprehensive response
            response = self._build_response(
                session_id=session_id,
                document_name=document_name,
                ai_analysis=ai_analysis,
                original_clauses=clauses if not preserve_entities else [
                    self.anonymizer.restore_entities(clause, entity_mappings) for clause in clauses
                ],
                entity_mappings=entity_mappings,
                processing_metadata=processing_metadata
            )

            processing_metadata["processing_steps"].append("response_compilation")
            processing_metadata["total_processing_time"] = "completed"

            return response

        except Exception as e:
            processing_metadata["error"] = str(e)
            processing_metadata["processing_steps"].append(f"error: {str(e)}")

            return self._create_error_response(
                session_id,
                document_name,
                f"Processing failed: {str(e)}",
                processing_metadata
            )

    def _detect_document_type(self, text: str) -> str:
        """Detect document type from content analysis."""
        text_lower = text.lower()

        # Document type indicators
        type_indicators = {
            'Employment Contract': ['employee', 'employer', 'salary', 'termination', 'job', 'work'],
            'Rental Agreement': ['tenant', 'landlord', 'rent', 'lease', 'property', 'premises'],
            'Service Agreement': ['service', 'provider', 'client', 'deliverables', 'scope', 'payment'],
            'Partnership Agreement': ['partner', 'partnership', 'profit', 'loss', 'equity', 'business'],
            'Purchase Agreement': ['buyer', 'seller', 'purchase', 'sale', 'goods', 'delivery'],
            'License Agreement': ['license', 'licensor', 'licensee', 'intellectual property', 'rights'],
            'Non-Disclosure Agreement': ['confidential', 'non-disclosure', 'nda', 'proprietary', 'secret'],
            'Terms of Service': ['terms', 'service', 'user', 'website', 'platform', 'account'],
            'Privacy Policy': ['privacy', 'data', 'information', 'collect', 'personal', 'cookies'],
            'Loan Agreement': ['loan', 'lender', 'borrower', 'interest', 'repayment', 'collateral']
        }

        # Score each document type
        type_scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[doc_type] = score

        # Return the highest scoring type, or "Legal Document" as fallback
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        else:
            return "Legal Document"

    def _restore_entities_in_analysis(
            self,
            ai_analysis: Dict[str, Any],
            entity_mappings: Dict[str, Any],
            original_clauses: List[str]
    ) -> Dict[str, Any]:
        """Restore original entities in the AI analysis results."""

        # Restore entities in clause analyses
        for i, clause_analysis in enumerate(ai_analysis.get('clause_analyses', [])):
            if i < len(original_clauses):
                # Restore entities in explanation and recommendations
                if clause_analysis.get('explanation'):
                    clause_analysis['explanation'] = self.anonymizer.restore_entities(
                        clause_analysis['explanation'], entity_mappings
                    )

                if clause_analysis.get('recommendations'):
                    clause_analysis['recommendations'] = [
                        self.anonymizer.restore_entities(rec, entity_mappings)
                        for rec in clause_analysis['recommendations']
                    ]

                if clause_analysis.get('legal_implications'):
                    clause_analysis['legal_implications'] = self.anonymizer.restore_entities(
                        clause_analysis['legal_implications'], entity_mappings
                    )

        return ai_analysis

    def _build_response(
            self,
            session_id: str,
            document_name: str,
            ai_analysis: Dict[str, Any],
            original_clauses: List[str],
            entity_mappings: Dict[str, Any],
            processing_metadata: Dict[str, Any]
    ) -> DocumentAnalysisResponse:
        """Build the complete response object."""

        # Build document summary
        summary_data = ai_analysis.get('document_summary', {})
        summary = DocumentSummary(
            document_type=summary_data.get('document_type', 'Legal Document'),
            total_clauses=len(original_clauses),
            high_risk_count=sum(1 for c in ai_analysis.get('clause_analyses', []) if c.get('severity') == '🔴'),
            medium_risk_count=sum(1 for c in ai_analysis.get('clause_analyses', []) if c.get('severity') == '🟡'),
            low_risk_count=sum(1 for c in ai_analysis.get('clause_analyses', []) if c.get('severity') == '🟢'),
            overall_risk_score=summary_data.get('overall_risk_score', 0.5),
            key_concerns=summary_data.get('key_concerns', []),
            document_category=summary_data.get('document_category', 'Contract')
        )

        # Build clause analyses
        clause_analyses = []
        for i, clause_data in enumerate(ai_analysis.get('clause_analyses', [])):
            if i < len(original_clauses):
                # Extract entities found in this clause
                entities_in_clause = []
                for placeholder, mapping in entity_mappings.items():
                    if placeholder in ai_analysis.get('clause_analyses', [{}])[i].get('clause_id', ''):
                        entities_in_clause.append(EntityMapping(**mapping))

                clause_analysis = ClauseAnalysis(
                    clause_id=f"CLAUSE_{i + 1}",
                    original_clause=original_clauses[i],
                    anonymized_clause=None,  # We restore entities, so no need for anonymized version
                    severity=clause_data.get('severity', '🟢'),
                    risk_score=clause_data.get('risk_score', 0.2),
                    category=clause_data.get('category', 'General'),
                    explanation=clause_data.get('explanation'),
                    recommendations=clause_data.get('recommendations', []),
                    legal_implications=clause_data.get('legal_implications'),
                    entities_found=entities_in_clause
                )
                clause_analyses.append(clause_analysis)

        # Convert entity mappings to proper format
        formatted_entity_mappings = {
            placeholder: EntityMapping(**mapping)
            for placeholder, mapping in entity_mappings.items()
        }

        return DocumentAnalysisResponse(
            session_id=session_id,
            document=document_name,
            summary=summary,
            clauses=clause_analyses,
            entity_mappings=formatted_entity_mappings,
            processing_metadata=processing_metadata
        )

    def _create_error_response(
            self,
            session_id: str,
            document_name: str,
            error_message: str,
            processing_metadata: Dict[str, Any]
    ) -> DocumentAnalysisResponse:
        """Create an error response."""

        summary = DocumentSummary(
            document_type="Unknown",
            total_clauses=0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            overall_risk_score=0.0,
            key_concerns=[error_message],
            document_category="Error"
        )

        processing_metadata["status"] = "error"

        return DocumentAnalysisResponse(
            session_id=session_id,
            document=document_name,
            summary=summary,
            clauses=[],
            entity_mappings={},
            processing_metadata=processing_metadata
        )