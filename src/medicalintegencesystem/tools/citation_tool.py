"""
Citation Formatter Tool — formats academic citations in Vancouver/NLM style.
Supports Vancouver (numbered), APA, and NLM styles commonly used in medical literature.
"""

import re
import logging
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CitationFormatterInput(BaseModel):
    """Input schema for CitationFormatterTool."""
    papers_data: str = Field(
        description="A list of paper metadata as a formatted string or JSON-like text. "
                    "Each paper should have: title, authors, journal, year, volume, "
                    "issue (optional), pages (optional), pmid or doi. "
                    "You can pass the raw paper list text from the PubMed Search Tool."
    )
    citation_style: str = Field(
        default="vancouver",
        description="Citation style to use: 'vancouver' (numbered, standard in medicine), "
                    "'apa' (author-year), or 'nlm' (National Library of Medicine style). "
                    "Default is 'vancouver' — preferred for medical journals."
    )


class CitationFormatterTool(BaseTool):
    """
    Formats academic paper citations in medical citation styles.
    Produces numbered Vancouver/NLM reference lists standard in medical publications.
    """
    name: str = "Citation Formatter Tool"
    description: str = (
        "Format academic paper citations in Vancouver, APA, or NLM style. "
        "Pass the paper metadata text from the PubMed search results. "
        "Returns a properly formatted, numbered reference list suitable for "
        "inclusion in medical research reports. Vancouver style is standard "
        "for most medical journals (NEJM, Lancet, JAMA)."
    )
    args_schema: Type[BaseModel] = CitationFormatterInput

    def _run(self, papers_data: str, citation_style: str = "vancouver") -> str:
        """Parse paper data and format citations."""
        try:
            logger.info(f"[CitationFormatter] Formatting citations in {citation_style} style")
            papers = self._parse_papers_from_text(papers_data)

            if not papers:
                return (
                    "Could not extract paper metadata from the provided text. "
                    "Please provide text containing PMID, title, authors, journal, and year fields."
                )

            citations = []
            for i, paper in enumerate(papers, 1):
                citation = self._format_citation(paper, i, citation_style)
                citations.append(citation)

            style_name = {
                "vancouver": "Vancouver (Numbered)",
                "apa": "APA 7th Edition",
                "nlm": "NLM / MEDLINE"
            }.get(citation_style, citation_style)

            header = [
                f"## References ({style_name} Style)",
                f"*{len(citations)} citations formatted*",
                "",
            ]

            return "\n".join(header) + "\n".join(citations)

        except Exception as e:
            logger.error(f"[CitationFormatter] Error: {e}")
            return f"Citation formatting failed: {str(e)}"

    def _parse_papers_from_text(self, text: str) -> list[dict]:
        """Extract paper metadata from unstructured text (PubMed tool output)."""
        papers = []

        # Split into paper blocks by the "### Paper N" pattern
        blocks = re.split(r'###\s+Paper\s+\d+', text)

        for block in blocks:
            if not block.strip():
                continue

            paper = {}

            # Extract PMID
            pmid_match = re.search(r'\*\*PMID:\*\*\s*(\d+)', block)
            if pmid_match:
                paper["pmid"] = pmid_match.group(1)

            # Extract Title
            title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?=\n)', block)
            if title_match:
                paper["title"] = title_match.group(1).strip()

            # Extract Authors
            authors_match = re.search(r'\*\*Authors:\*\*\s*(.+?)(?=\n)', block)
            if authors_match:
                paper["authors"] = authors_match.group(1).strip()

            # Extract Journal and Year
            journal_match = re.search(r'\*\*Journal:\*\*\s*(.+?)\s*\((\d{4})\)', block)
            if journal_match:
                paper["journal"] = journal_match.group(1).strip()
                paper["year"] = journal_match.group(2)

            # Extract DOI
            doi_match = re.search(r'\*\*DOI:\*\*\s*https?://doi\.org/(\S+)', block)
            if doi_match:
                paper["doi"] = doi_match.group(1)

            # Extract PubMed URL
            url_match = re.search(r'\*\*PubMed URL:\*\*\s*(https?://\S+)', block)
            if url_match:
                paper["pubmed_url"] = url_match.group(1).strip()

            # Only add if we have minimum info
            if paper.get("title") and paper.get("authors"):
                papers.append(paper)

        return papers

    def _format_citation(self, paper: dict, number: int, style: str) -> str:
        """Format a single citation in the specified style."""
        title = paper.get("title", "Unknown title")
        authors = paper.get("authors", "Unknown authors")
        journal = paper.get("journal", "Unknown journal")
        year = paper.get("year", "n.d.")
        pmid = paper.get("pmid", "")
        doi = paper.get("doi", "")
        pubmed_url = paper.get("pubmed_url", "")

        # Build access string
        access_parts = []
        if doi:
            access_parts.append(f"doi: {doi}")
        if pmid:
            access_parts.append(f"PMID: {pmid}")
        access_str = ". ".join(access_parts)

        if style == "vancouver":
            # Vancouver: 1. Author AA, Author BB, et al. Title. Journal. Year; doi/PMID.
            return (
                f"{number}. {authors}. {title}. "
                f"*{journal}*. {year}."
                f"{f' {access_str}' if access_str else ''}"
            )

        elif style == "apa":
            # APA 7: Author, A. A., & Author, B. B. (Year). Title. Journal. DOI
            # Simplify author formatting from "Smith J, Jones K, et al" → keep as-is
            return (
                f"{number}. {authors} ({year}). {title}. "
                f"*{journal}*."
                f"{f' https://doi.org/{doi}' if doi else ''}"
                f"{f' PMID: {pmid}' if pmid and not doi else ''}"
            )

        elif style == "nlm":
            # NLM: Author AA, Author BB. Title. Abbreviated Journal. Year Month;vol(issue):pages.
            return (
                f"{number}. {authors}. {title}. "
                f"{journal}. {year}."
                f"{f' {access_str}' if access_str else ''}"
            )

        else:
            return f"{number}. {authors}. {title}. {journal}. {year}."
