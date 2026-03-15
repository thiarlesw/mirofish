"""Text processing service."""

from typing import List, Optional
from ..utils.file_parser import FileParser, split_text_into_chunks


class TextProcessor:
    """Text processor."""

    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """Extract text from multiple files."""
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """Split text into chunks. Args: text, chunk_size, overlap. Returns list of chunks."""
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text: normalize line breaks, collapse blank lines, trim lines. Args: text. Returns processed text."""
        import re

        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Get text statistics (total_chars, total_lines, total_words)."""
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }

