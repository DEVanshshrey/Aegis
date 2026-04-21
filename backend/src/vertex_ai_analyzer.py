import requests
import json
from typing import Dict, List, Any
import os
from datetime import datetime


class VertexAIAnalyzer:
    """Uses a hosted LLM to analyze legal clauses and provide intelligent insights."""

    def __init__(self):
        # Load Hugging Face configuration
        self.api_token = os.getenv('HF_API_TOKEN')
        model_id = os.getenv('HF_MODEL_ID', 'mistralai/Mistral-7B-Instruct-v0.2')
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

        if not self.api_token:
            raise ValueError("HF_API_TOKEN environment variable not set.")

        # Analysis prompts for different use cases
        self.analysis_prompts = {
            'comprehensive': self._get_comprehensive_prompt(),
            'basic': self._get_basic_prompt(),
            'detailed': self._get_detailed_prompt()
        }

    async def analyze_document(
            self,
            clauses: List[str],
            document_type: str,
            analysis_depth: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """Analyze document clauses using a hosted LLM."""
        prompt = self._build_analysis_prompt(clauses, document_type, analysis_depth)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_full_text": False,
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            
            generated_text = response.json()[0]['generated_text']
            analysis_result = self._parse_ai_response(generated_text, clauses)

            return analysis_result

        except Exception as e:
            print(f"LLM analysis failed: {e}")
            # Fallback to basic rule-based analysis
            return self._fallback_analysis(clauses, document_type)

    def _build_analysis_prompt(self, clauses: List[str], document_type: str, analysis_depth: str) -> str:
        """Build the prompt for LLM analysis."""
        base_prompt = self.analysis_prompts[analysis_depth]
        clauses_text = "\n".join([f"CLAUSE_{i + 1}: {clause}" for i, clause in enumerate(clauses)])

        # Using instruction-following format for models like Mistral
        prompt = f"""
[INST]
{base_prompt}

DOCUMENT TYPE: {document_type}

CLAUSES TO ANALYZE:
{clauses_text}

Provide your analysis *only* in the following JSON format. Do not add any text before or after the JSON object.
{{
    "document_summary": {{
        "document_type": "detected document type",
        "overall_risk_score": 0.0-1.0,
        "key_concerns": ["concern1", "concern2"],
        "document_category": "category"
    }},
    "clause_analyses": [
        {{
            "clause_id": "CLAUSE_1",
            "severity": "🔴/🟡/🟢",
            "risk_score": 0.0-1.0,
            "category": "legal category",
            "explanation": "plain English explanation",
            "recommendations": ["recommendation1", "recommendation2"],
            "legal_implications": "detailed implications"
        }}
    ]
}}
[/INST]
"""
        return prompt

    def _get_comprehensive_prompt(self) -> str:
        """Get the comprehensive analysis prompt."""
        return """
You are an expert legal analyst specializing in contract and legal document review. 
Your task is to analyze legal clauses and identify risks, unusual terms, and potential issues.

For each clause, consider:
1. Risk Level: High (🔴), Medium (🟡), or Low (🟢)
2. Risk Score: 0.0 (no risk) to 1.0 (maximum risk)
3. Legal Category: Contract type, obligation type, etc.
4. Plain English Explanation: What does this mean for a regular person?
5. Recommendations: What should the user consider or negotiate?
6. Legal Implications: Potential consequences and legal considerations

Focus on:
- Unfair or one-sided terms
- Unusual penalty clauses
- Vague or ambiguous language
- Terms that limit rights or increase obligations
- Industry-standard vs non-standard clauses
- Financial implications
- Termination and dispute resolution terms
"""

    def _get_basic_prompt(self) -> str:
        """Get the basic analysis prompt."""
        return """
You are a legal document analyzer. Review each clause and identify:
1. Risk level (🔴 High, 🟡 Medium, 🟢 Low)
2. Brief explanation of concerns
3. Overall document risk assessment

Focus on the most important risks that could significantly impact the user.
"""

    def _get_detailed_prompt(self) -> str:
        """Get the detailed analysis prompt."""
        return """
You are a senior legal counsel providing detailed contract analysis.
For each clause, provide comprehensive analysis including:

1. Legal precedent and standard practices
2. Negotiation points and alternatives
3. Jurisdiction-specific considerations
4. Risk mitigation strategies
5. Detailed legal implications
6. Cross-references with other clauses
7. Industry-specific considerations

Provide thorough analysis suitable for legal professionals.
"""

    def _parse_ai_response(self, response_text: str, original_clauses: List[str]) -> Dict[str, Any]:
        """Parse the AI response into structured format."""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_response = json.loads(json_str)

                # Validate and enhance the response
                return self._validate_and_enhance_response(parsed_response, original_clauses)
            else:
                # Fallback parsing if JSON extraction fails
                return self._fallback_parsing(response_text, original_clauses)

        except Exception as e:
            print(f"Failed to parse AI response: {e}")
            return self._fallback_analysis(original_clauses, "Unknown")

    def _validate_and_enhance_response(self, parsed_response: Dict[str, Any], original_clauses: List[str]) -> Dict[
        str, Any]:
        """Validate and enhance the AI response."""

        # Ensure all required fields exist
        if 'document_summary' not in parsed_response:
            parsed_response['document_summary'] = {
                'document_type': 'Legal Document',
                'overall_risk_score': 0.5,
                'key_concerns': [],
                'document_category': 'Contract'
            }

        if 'clause_analyses' not in parsed_response:
            parsed_response['clause_analyses'] = []

        # Ensure we have analysis for all clauses
        analyzed_clauses = {analysis.get('clause_id', '') for analysis in parsed_response['clause_analyses']}

        for i, clause in enumerate(original_clauses):
            clause_id = f"CLAUSE_{i + 1}"
            if clause_id not in analyzed_clauses:
                # Add missing clause analysis
                parsed_response['clause_analyses'].append({
                    'clause_id': clause_id,
                    'severity': '🟢',
                    'risk_score': 0.2,
                    'category': 'Standard',
                    'explanation': 'Standard clause with minimal risk.',
                    'recommendations': [],
                    'legal_implications': 'Standard legal provision.'
                })

        return parsed_response

    def _fallback_parsing(self, response_text: str, original_clauses: List[str]) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails."""
        # Simple text-based parsing
        clause_analyses = []

        for i, clause in enumerate(original_clauses):
            # Basic risk assessment based on keywords
            risk_score = self._calculate_basic_risk_score(clause)
            severity = '🔴' if risk_score > 0.7 else '🟡' if risk_score > 0.3 else '🟢'

            clause_analyses.append({
                'clause_id': f"CLAUSE_{i + 1}",
                'severity': severity,
                'risk_score': risk_score,
                'category': 'General',
                'explanation': f"Clause assessed with risk score {risk_score:.2f}",
                'recommendations': [],
                'legal_implications': 'Standard legal provision.'
            })

        return {
            'document_summary': {
                'document_type': 'Legal Document',
                'overall_risk_score': sum(c['risk_score'] for c in clause_analyses) / len(clause_analyses),
                'key_concerns': ['AI analysis incomplete - manual review recommended'],
                'document_category': 'Contract'
            },
            'clause_analyses': clause_analyses
        }

    def _fallback_analysis(self, clauses: List[str], document_type: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails."""
        clause_analyses = []

        for i, clause in enumerate(clauses):
            risk_score = self._calculate_basic_risk_score(clause)
            severity = '🔴' if risk_score > 0.7 else '🟡' if risk_score > 0.3 else '🟢'

            clause_analyses.append({
                'clause_id': f"CLAUSE_{i + 1}",
                'severity': severity,
                'risk_score': risk_score,
                'category': 'General',
                'explanation': self._get_basic_explanation(clause, severity),
                'recommendations': [],
                'legal_implications': 'Manual legal review recommended.'
            })

        return {
            'document_summary': {
                'document_type': document_type,
                'overall_risk_score': sum(c['risk_score'] for c in clause_analyses) / len(clause_analyses) if clause_analyses else 0,
                'key_concerns': ['Automated analysis only - professional review recommended'],
                'document_category': 'Legal Document'
            },
            'clause_analyses': clause_analyses
        }

    def _calculate_basic_risk_score(self, clause: str) -> float:
        """Calculate basic risk score using keyword analysis."""
        high_risk_keywords = [
            'penalty', 'forfeit', 'without notice', 'sole discretion',
            'unlimited liability', 'irrevocable', 'waive', 'indemnify'
        ]

        medium_risk_keywords = [
            'reasonable discretion', 'may be deemed', 'including but not limited',
            'minor repairs', 'appropriate action', 'from time to time'
        ]

        clause_lower = clause.lower()
        risk_score = 0.1  # Base risk

        for keyword in high_risk_keywords:
            if keyword in clause_lower:
                risk_score += 0.3

        for keyword in medium_risk_keywords:
            if keyword in clause_lower:
                risk_score += 0.15

        return min(risk_score, 1.0)

    def _get_basic_explanation(self, clause: str, severity: str) -> str:
        """Get basic explanation based on severity."""
        if severity == '🔴':
            return "This clause contains terms that may be risky or heavily favor one party."
        elif severity == '🟡':
            return "This clause contains language that may be unclear or subject to interpretation."
        else:
            return "This appears to be a standard clause with minimal risk."