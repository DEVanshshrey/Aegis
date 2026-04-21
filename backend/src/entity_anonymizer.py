import spacy
from google.cloud import language_v1
import re
import uuid
from typing import Dict, List, Tuple, Any
import json


class EntityAnonymizer:
    """Handles Named Entity Recognition and anonymization of sensitive information."""

    def __init__(self):
        # Load spaCy model for additional NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, use blank model
            self.nlp = spacy.blank("en")

        # Initialize Google Cloud Language client
        self.language_client = language_v1.LanguageServiceClient()

        # Define sensitive entity types
        self.sensitive_entities = {
            'PERSON': 'PERSON_{}',
            'ORGANIZATION': 'ORG_{}',
            'LOCATION': 'LOC_{}',
            'ADDRESS': 'ADDR_{}',
            'PHONE_NUMBER': 'PHONE_{}',
            'EMAIL': 'EMAIL_{}',
            'DATE': 'DATE_{}',
            'MONEY': 'AMT_{}',
            'PERCENT': 'PCT_{}',
            'NUMBER': 'NUM_{}'
        }

        # Regex patterns for legal-specific entities
        self.legal_patterns = {
            'CONTRACT_ID': r'Contract\s*(?:No\.?|#|ID)\s*:?\s*([A-Z0-9\-/]+)',
            'CASE_NUMBER': r'Case\s*(?:No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
            'LICENSE_NUMBER': r'License\s*(?:No\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
            'PAN_NUMBER': r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
            'AADHAR_NUMBER': r'\d{4}\s?\d{4}\s?\d{4}',
            'GST_NUMBER': r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}'
        }

    def anonymize_document(self, text: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Anonymize sensitive entities in the document.

        Returns:
            Tuple of (anonymized_text, entity_mappings)
        """
        entity_mappings = {}
        anonymized_text = text

        # Step 1: Use Google Cloud NLP for entity detection
        gcp_entities = self._extract_entities_gcp(text)

        # Step 2: Use spaCy for additional entity detection
        spacy_entities = self._extract_entities_spacy(text)

        # Step 3: Use regex for legal-specific patterns
        legal_entities = self._extract_legal_entities(text)

        # Step 4: Combine and deduplicate entities
        all_entities = self._combine_entities(gcp_entities, spacy_entities, legal_entities)

        # Step 5: Anonymize entities
        for entity in sorted(all_entities, key=lambda x: x['start'], reverse=True):
            if entity['type'] in self.sensitive_entities:
                # Generate unique placeholder
                placeholder = self._generate_placeholder(entity['type'], session_id)

                # Replace in text
                start, end = entity['start'], entity['end']
                original_text = anonymized_text[start:end]
                anonymized_text = anonymized_text[:start] + placeholder + anonymized_text[end:]

                # Store mapping
                entity_mappings[placeholder] = {
                    'original': original_text,
                    'anonymized': placeholder,
                    'entity_type': entity['type'],
                    'confidence': entity.get('confidence', 0.95),
                    'start': start,
                    'end': end
                }

        return anonymized_text, entity_mappings

    def restore_entities(self, anonymized_text: str, entity_mappings: Dict[str, Any]) -> str:
        """Restore original entities from anonymized text."""
        restored_text = anonymized_text

        # Sort by position to avoid index shifting issues
        sorted_mappings = sorted(
            entity_mappings.items(),
            key=lambda x: x[1].get('start', 0),
            reverse=True
        )

        for placeholder, mapping in sorted_mappings:
            restored_text = restored_text.replace(placeholder, mapping['original'])

        return restored_text

    def _extract_entities_gcp(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Google Cloud Natural Language API."""
        try:
            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

            # Call the API
            entities_response = self.language_client.analyze_entities(
                request={'document': document, 'encoding_type': language_v1.EncodingType.UTF8}
            )

            entities = []
            for entity in entities_response.entities:
                for mention in entity.mentions:
                    entities.append({
                        'text': mention.text.content,
                        'type': entity.type_.name,
                        'start': mention.text.begin_offset,
                        'end': mention.text.begin_offset + len(mention.text.content),
                        'confidence': entity.salience
                    })

            return entities

        except Exception as e:
            print(f"GCP NER failed: {e}")
            return []

    def _extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy."""
        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9  # spaCy doesn't provide confidence scores
                })

            return entities

        except Exception as e:
            print(f"spaCy NER failed: {e}")
            return []

    def _extract_legal_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal-specific entities using regex patterns."""
        entities = []

        for entity_type, pattern in self.legal_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                entities.append({
                    'text': match.group(0),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.85
                })

        return entities

    def _combine_entities(self, *entity_lists) -> List[Dict[str, Any]]:
        """Combine entities from different sources and remove duplicates."""
        combined = []
        seen_spans = set()

        for entity_list in entity_lists:
            for entity in entity_list:
                span = (entity['start'], entity['end'])

                # Skip if we've already seen this span
                if span not in seen_spans:
                    combined.append(entity)
                    seen_spans.add(span)

        return combined

    def _generate_placeholder(self, entity_type: str, session_id: str) -> str:
        """Generate unique placeholder for entity type."""
        template = self.sensitive_entities.get(entity_type, f'{entity_type}_{{}}')
        unique_id = str(uuid.uuid4())[:8]
        return template.format(unique_id)