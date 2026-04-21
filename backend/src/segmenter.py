import re
from typing import List
import spacy


class UniversalClauseSegmenter:
    """Universal clause segmenter for all types of legal documents."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = spacy.blank("en")

        # Universal patterns for legal documents
        self.clause_patterns = {
            'numbered': r'(?:^|\n)(\d+\.?\s+.+?)(?=\n\d+\.?\s+|$)',
            'lettered': r'(?:^|\n)\(([a-z])\)\s*(.+?)(?=\n\([a-z]\)|$)',
            'article': r'(?:^|\n)(Article\s+\w+.+?)(?=\nArticle\s+\w+|$)',
            'section': r'(?:^|\n)(Section\s+\w+.+?)(?=\nSection\s+\w+|$)',
            'clause': r'(?:^|\n)(Clause\s+\w+.+?)(?=\nClause\s+\w+|$)',
            'paragraph': r'(?:^|\n)(\w+\.?\s*\w+.*?)(?=\n\w+\.?\s*\w+|$)',
        }

    def segment_document(self, text: str) -> List[str]:
        """Segment document into clauses using multiple strategies."""
        if not text.strip():
            return []

        # Clean text
        text = self._clean_text(text)

        # Try different segmentation strategies
        segments = []

        # Strategy 1: Formal structure detection
        formal_segments = self._extract_formal_structure(text)
        if formal_segments and len(formal_segments) > 3:
            segments = formal_segments

        # Strategy 2: Sentence-based with legal context
        if not segments:
            segments = self._extract_legal_sentences(text)

        # Strategy 3: Paragraph-based fallback
        if not segments:
            segments = self._extract_paragraphs(text)

        # Filter and clean segments
        return self._filter_segments(segments)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        # Remove multiple empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def _extract_formal_structure(self, text: str) -> List[str]:
        """Extract formally structured clauses."""
        all_segments = []

        for pattern_name, pattern in self.clause_patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

            if matches:
                if pattern_name in ['lettered']:
                    # Handle tupled matches from lettered pattern
                    segments = [f"({match[0]}) {match[1]}" for match in matches]
                else:
                    segments = [match if isinstance(match, str) else ' '.join(match)
                                for match in matches]

                if len(segments) > len(all_segments):
                    all_segments = segments

        return all_segments

    def _extract_legal_sentences(self, text: str) -> List[str]:
        """Extract sentences with legal context awareness."""
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")

        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            # Skip very short sentences
            if len(sentence_text) < 20:
                continue

            # Legal sentence indicators
            legal_indicators = [
                'shall', 'hereby', 'whereas', 'therefore', 'provided that',
                'subject to', 'in accordance with', 'notwithstanding'
            ]

            # Enhance sentence if it contains legal language
            if any(indicator in sentence_text.lower() for indicator in legal_indicators):
                sentences.append(sentence_text)
            elif len(sentence_text) > 50:  # Include longer sentences
                sentences.append(sentence_text)

        return sentences

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs as fallback segmentation."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if len(p.strip()) > 30]

    def _filter_segments(self, segments: List[str]) -> List[str]:
        """Filter and clean segments."""
        filtered = []

        for segment in segments:
            segment = segment.strip()

            # Skip empty or very short segments
            if len(segment) < 15:
                continue

            # Skip headers, footers, page numbers
            if self._is_metadata(segment):
                continue

            filtered.append(segment)

        return filtered

    def _is_metadata(self, text: str) -> bool:
        """Check if text is metadata (headers, footers, page numbers)."""
        text_lower = text.lower()

        metadata_indicators = [
            'page', 'confidential', 'draft', 'header', 'footer',
            'copyright', '©', 'all rights reserved'
        ]

        # Check if text is very short and contains metadata indicators
        if len(text) < 50 and any(indicator in text_lower for indicator in metadata_indicators):
            return True

        # Check if text is just numbers (page numbers)
        if re.match(r'^\d+$', text.strip()):
            return True

        return False