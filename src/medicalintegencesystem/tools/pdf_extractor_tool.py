"""
PDF Text Extractor Tool — downloads and extracts text from open-access medical papers.
Primarily works with PMC open-access PDFs. Falls back to abstract text if PDF unavailable.
Uses PyMuPDF (fitz) for high-quality text extraction with cleanup.
"""

import io
import logging
import re
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

MAX_CHARS = 8000  # Token-safe limit for LLM consumption


class PDFExtractorInput(BaseModel):
    """Input schema for PDFTextExtractorTool."""
    pdf_url: str = Field(
        description="Direct URL to a PDF file. Works best with PMC open-access URLs like "
                    "'https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/pdf/'. "
                    "Can also accept any open-access DOI-resolved PDF URL."
    )
    paper_title: str = Field(
        default="Unknown Paper",
        description="Title of the paper (for logging and output labeling)."
    )
    max_chars: int = Field(
        default=6000,
        description="Maximum number of characters to extract (default 6000 for LLM efficiency).",
        ge=1000,
        le=8000
    )


class PDFTextExtractorTool(BaseTool):
    """
    Downloads open-access PDFs and extracts clean text content.
    Handles PMC open-access papers and other public PDFs.
    Falls back gracefully if PDF is inaccessible (paywalled, down, etc.).
    """
    name: str = "PDF Text Extractor Tool"
    description: str = (
        "Download and extract text from open-access medical paper PDFs. "
        "Works best with PMC (PubMed Central) open-access paper URLs. "
        "Provide the direct PDF URL and paper title. Returns extracted text "
        "up to 6000 characters, focusing on Methods, Results, and Discussion sections. "
        "If PDF is unavailable or paywalled, returns a clear error message."
    )
    args_schema: Type[BaseModel] = PDFExtractorInput

    def _run(self, pdf_url: str, paper_title: str = "Unknown Paper", max_chars: int = 6000) -> str:
        """Download PDF and extract text content."""
        try:
            logger.info(f"[PDFExtractor] Downloading: {pdf_url}")
            pdf_bytes = self._download_pdf(pdf_url)
            text = self._extract_text(pdf_bytes)
            cleaned = self._clean_text(text)
            truncated = self._smart_truncate(cleaned, max_chars)

            return (
                f"## Full Text Extraction: {paper_title}\n"
                f"**Source:** {pdf_url}\n"
                f"**Characters extracted:** {len(truncated):,} / {len(cleaned):,} total\n\n"
                f"---\n\n{truncated}"
            )

        except PaperPaywallError as e:
            return (
                f"## PDF Extraction Failed: {paper_title}\n"
                f"**Reason:** Paper is paywalled or requires institutional access.\n"
                f"**URL:** {pdf_url}\n"
                f"**Recommendation:** Use the abstract from PubMed instead.\n"
                f"**Error:** {str(e)}"
            )
        except Exception as e:
            logger.error(f"[PDFExtractor] Error extracting {pdf_url}: {e}")
            return (
                f"## PDF Extraction Failed: {paper_title}\n"
                f"**Reason:** {str(e)}\n"
                f"**URL:** {pdf_url}\n"
                f"**Recommendation:** Work with the abstract from PubMed instead."
            )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def _download_pdf(self, url: str) -> bytes:
        """Download PDF bytes from URL with proper headers."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; MedicalResearchBot/1.0; "
                "+https://example.com/bot)"
            ),
            "Accept": "application/pdf,*/*",
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)

        # Check for paywall/auth redirects
        content_type = response.headers.get("content-type", "")
        if "html" in content_type and "pdf" not in content_type:
            raise PaperPaywallError(
                f"URL returned HTML instead of PDF (likely paywalled or login required). "
                f"Content-Type: {content_type}"
            )

        if response.status_code == 403:
            raise PaperPaywallError("HTTP 403 Forbidden — access denied (paywalled)")
        if response.status_code == 401:
            raise PaperPaywallError("HTTP 401 Unauthorized — authentication required")

        response.raise_for_status()

        # Read with size limit (max 20MB)
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            total += len(chunk)
            if total > 20 * 1024 * 1024:
                break

        return b"".join(chunks)

    def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is not installed. Run: uv add PyMuPDF"
            )

        text_parts = []
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                if doc.page_count == 0:
                    raise ValueError("PDF has no pages")

                for page_num in range(min(doc.page_count, 20)):  # Max 20 pages
                    page = doc.load_page(page_num)
                    text_parts.append(page.get_text("text"))

        except Exception as e:
            raise ValueError(f"PyMuPDF extraction failed: {str(e)}")

        return "\n".join(text_parts)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: remove headers/footers, fix whitespace."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # Remove lines that look like page numbers or headers (short lines with numbers)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip near-empty lines, standalone page numbers, and DOI/URL-only lines
            if len(stripped) < 3:
                continue
            if re.match(r'^[\d\s\-–/]+$', stripped) and len(stripped) < 10:
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Intelligently truncate text, prioritizing Methods/Results/Discussion sections.
        """
        if len(text) <= max_chars:
            return text

        # Try to find key sections
        section_patterns = [
            r'(abstract[\s\S]{0,200})',
            r'(introduction[\s\S]{0,500})',
            r'(methods?\s[\s\S]{0,2000})',
            r'(results?\s[\s\S]{0,2000})',
            r'(discussion\s[\s\S]{0,1500})',
            r'(conclusion[\s\S]{0,500})',
        ]

        # If we can find structured sections, prioritize them
        found_sections = []
        for pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found_sections.append(match.group(1))

        if found_sections:
            combined = "\n\n".join(found_sections)
            if len(combined) <= max_chars:
                return combined
            return combined[:max_chars] + "\n\n[... text truncated for length ...]"

        # Fallback: simple truncation
        return text[:max_chars] + "\n\n[... text truncated for length ...]"


class PaperPaywallError(Exception):
    """Raised when a PDF is behind a paywall or requires authentication."""
    pass
