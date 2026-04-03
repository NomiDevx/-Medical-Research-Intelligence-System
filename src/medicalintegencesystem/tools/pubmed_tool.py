"""
PubMed Search Tool — integrates with NCBI E-utilities REST API.
Searches PubMed for medical papers and retrieves full metadata including abstracts.
Rate-limited to 3 req/sec (10/sec with API key). Uses tenacity for retries.
"""

import os
import time
import logging
import xml.etree.ElementTree as ET
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
REQUEST_DELAY = 0.34 if not NCBI_API_KEY else 0.1  # seconds between requests


class PubMedSearchInput(BaseModel):
    """Input schema for PubMedSearchTool."""
    query: str = Field(
        description="PubMed search query string. Can include MeSH terms, Boolean operators "
                    "(AND, OR, NOT), field tags like [MeSH], [Title/Abstract], [Author]. "
                    "Example: '(diabetes mellitus type 2[MeSH]) AND (metformin[Title/Abstract]) "
                    "AND (randomized controlled trial[pt])'"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of papers to retrieve (1-20). Default is 10.",
        ge=1,
        le=20
    )
    sort_by: str = Field(
        default="relevance",
        description="Sort order: 'relevance' or 'date' (most recent first)."
    )


class PubMedSearchTool(BaseTool):
    """
    Searches PubMed via NCBI E-utilities and retrieves paper metadata.
    Returns structured results with titles, abstracts, authors, journals, PMIDs and DOIs.
    """
    name: str = "PubMed Search Tool"
    description: str = (
        "Search PubMed for medical research papers using keywords, MeSH terms, and Boolean "
        "operators. Returns metadata including PMID, title, authors, journal, year, abstract, "
        "and DOI for each paper. Use this to find relevant medical literature."
    )
    args_schema: Type[BaseModel] = PubMedSearchInput

    def _run(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> str:
        """Execute PubMed search and return formatted paper metadata."""
        try:
            logger.info(f"[PubMedSearchTool] Searching PubMed: '{query}' (max={max_results})")
            pmids = self._esearch(query, max_results, sort_by)

            if not pmids:
                return f"No papers found for query: '{query}'. Try broadening your search terms."

            logger.info(f"[PubMedSearchTool] Found {len(pmids)} papers. Fetching details...")
            papers = self._efetch(pmids)

            return self._format_results(papers, query)

        except Exception as e:
            logger.error(f"[PubMedSearchTool] Error: {e}")
            return f"PubMed search failed: {str(e)}. Please try with a different query."

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def _esearch(self, query: str, max_results: int, sort_by: str) -> list[str]:
        """Run ESearch to get PMIDs matching the query."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance" if sort_by == "relevance" else "pub+date",
            "usehistory": "n",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        time.sleep(REQUEST_DELAY)
        response = requests.get(f"{NCBI_BASE_URL}/esearch.fcgi", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def _efetch(self, pmids: list[str]) -> list[dict]:
        """Fetch full records for given PMIDs using EFetch."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        time.sleep(REQUEST_DELAY)
        response = requests.get(f"{NCBI_BASE_URL}/efetch.fcgi", params=params, timeout=30)
        response.raise_for_status()
        return self._parse_xml(response.text)

    def _parse_xml(self, xml_text: str) -> list[dict]:
        """Parse PubMed XML response into structured paper dicts."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"[PubMedSearchTool] XML parse error: {e}")
            return papers

        for article in root.findall(".//PubmedArticle"):
            paper = {}

            # PMID
            pmid_el = article.find(".//PMID")
            paper["pmid"] = pmid_el.text if pmid_el is not None else "Unknown"

            # Title
            title_el = article.find(".//ArticleTitle")
            paper["title"] = (title_el.text or "").strip() if title_el is not None else "Unknown"

            # Abstract
            abstract_parts = article.findall(".//AbstractText")
            if abstract_parts:
                abstract_texts = []
                for part in abstract_parts:
                    label = part.get("Label", "")
                    text = part.text or ""
                    abstract_texts.append(f"{label}: {text}" if label else text)
                paper["abstract"] = " ".join(abstract_texts)[:3000]  # cap at 3000 chars
            else:
                paper["abstract"] = "Abstract not available."

            # Authors (first 3 + et al.)
            authors = []
            for author in article.findall(".//Author")[:3]:
                last = getattr(author.find("LastName"), "text", "")
                initials = getattr(author.find("Initials"), "text", "")
                if last:
                    authors.append(f"{last} {initials}".strip())
            author_count = len(article.findall(".//Author"))
            if author_count > 3:
                authors.append("et al.")
            paper["authors"] = ", ".join(authors) if authors else "Unknown"

            # Journal
            journal_el = article.find(".//Journal/Title")
            paper["journal"] = journal_el.text if journal_el is not None else "Unknown"

            # Year
            year_el = article.find(".//PubDate/Year")
            if year_el is None:
                year_el = article.find(".//PubDate/MedlineDate")
            paper["year"] = year_el.text[:4] if year_el is not None else "Unknown"

            # DOI
            doi_el = article.find(".//ArticleId[@IdType='doi']")
            paper["doi"] = doi_el.text if doi_el is not None else ""
            paper["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"

            # PMC (open access PDF link)
            pmc_el = article.find(".//ArticleId[@IdType='pmc']")
            paper["pmc_id"] = pmc_el.text if pmc_el is not None else ""
            if paper["pmc_id"]:
                paper["pdf_url"] = (
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper['pmc_id']}/pdf/"
                )
            else:
                paper["pdf_url"] = ""

            papers.append(paper)

        return papers

    def _format_results(self, papers: list[dict], query: str) -> str:
        """Format paper list into readable structured text for the agent."""
        lines = [
            f"## PubMed Search Results",
            f"**Query:** `{query}`",
            f"**Papers Found:** {len(papers)}",
            "",
        ]

        for i, p in enumerate(papers, 1):
            lines.append(f"### Paper {i}")
            lines.append(f"- **PMID:** {p['pmid']}")
            lines.append(f"- **Title:** {p['title']}")
            lines.append(f"- **Authors:** {p['authors']}")
            lines.append(f"- **Journal:** {p['journal']} ({p['year']})")
            lines.append(f"- **PubMed URL:** {p['pubmed_url']}")
            if p.get("doi"):
                lines.append(f"- **DOI:** https://doi.org/{p['doi']}")
            if p.get("pdf_url"):
                lines.append(f"- **Open Access PDF:** {p['pdf_url']}")
            lines.append(f"- **Abstract:**")
            lines.append(f"  {p['abstract'][:1000]}{'...' if len(p['abstract']) > 1000 else ''}")
            lines.append("")

        return "\n".join(lines)
