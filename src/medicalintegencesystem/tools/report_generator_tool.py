"""
Report Generator Tool — assembles and saves comprehensive medical research reports.
Takes all agent outputs and produces a structured markdown document with all required
sections: executive summary, findings, clinical evidence, GRADE table, recommendations,
and references.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Output directory — relative to project root, created at runtime
OUTPUT_DIR = Path(__file__).parents[3] / "output"


class ReportGeneratorInput(BaseModel):
    """Input schema for ReportGeneratorTool."""
    report_content: str = Field(
        description="The full compiled report content in markdown format. Should include "
                    "all sections: Executive Summary, Background, Key Findings, Clinical "
                    "Trial Evidence, Evidence Quality (GRADE), Clinical Recommendations, "
                    "Research Gaps, and References. Pass the complete report text here."
    )
    research_query: str = Field(
        description="The original research query this report addresses. Used for the title "
                    "and filename generation."
    )
    report_title: str = Field(
        default="",
        description="Optional custom title for the report. If empty, auto-generated from query."
    )


class ReportGeneratorTool(BaseTool):
    """
    Saves the final medical research report to the output directory as a markdown file.
    Adds a professional header with metadata (date, query, system info) before saving.
    Returns the absolute file path and a preview of the saved content.
    """
    name: str = "Report Generator Tool"
    description: str = (
        "Save the final compiled medical research report to the output directory. "
        "Pass the complete report content in markdown format along with the research query. "
        "The tool will add a professional header, timestamp, and save it as a .md file. "
        "Returns the saved file path for confirmation. Use this as the LAST step after "
        "all other agents have completed their analysis."
    )
    args_schema: Type[BaseModel] = ReportGeneratorInput

    def _run(
        self,
        report_content: str,
        research_query: str,
        report_title: str = ""
    ) -> str:
        """Assemble and save the final report."""
        try:
            # Ensure output directory exists
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Generate title and filename
            if not report_title:
                clean_query = research_query[:60].replace("/", "-").replace("\\", "-")
                report_title = f"Medical Research Report: {clean_query}"

            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M UTC")
            filename = f"medical_research_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
            output_path = OUTPUT_DIR / filename

            # Build professional header
            header = self._build_header(report_title, research_query, date_str, time_str)

            # Build footer with system info
            footer = self._build_footer(timestamp)

            # Assemble full document
            full_report = f"{header}\n\n---\n\n{report_content}\n\n---\n\n{footer}"

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_report)

            file_size_kb = output_path.stat().st_size / 1024
            word_count = len(full_report.split())

            logger.info(f"[ReportGenerator] Report saved: {output_path}")

            return (
                f"## ✅ Report Successfully Generated\n\n"
                f"**File:** `{output_path}`\n"
                f"**Size:** {file_size_kb:.1f} KB\n"
                f"**Word Count:** {word_count:,} words\n"
                f"**Generated:** {date_str} at {time_str}\n\n"
                f"### Report Preview (first 500 chars)\n\n"
                f"```\n{full_report[:500]}...\n```\n\n"
                f"The complete report has been saved and is ready for review."
            )

        except Exception as e:
            logger.error(f"[ReportGenerator] Error saving report: {e}")
            return (
                f"## ❌ Report Generation Failed\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Fallback:** Here is the report content for manual saving:\n\n"
                f"{report_content[:2000]}..."
            )

    def _build_header(
        self,
        title: str,
        query: str,
        date_str: str,
        time_str: str
    ) -> str:
        """Build a professional report header with metadata."""
        return f"""# {title}

---

| Field | Details |
|-------|---------|
| **Research Query** | {query} |
| **Generated Date** | {date_str} |
| **Generated Time** | {time_str} |
| **System** | Medical Research Intelligence System v1.0 |
| **Framework** | CrewAI Multi-Agent Pipeline |
| **LLM Backend** | DeepSeek (deepseek-chat) |
| **Agents Used** | Research Coordinator, Literature Scout, Paper Analyzer, Clinical Data Interpreter, Medical Validator, Report Writer |

> **⚠️ Disclaimer:** This report is generated by an AI-powered research system and is intended
> for informational and research purposes only. It does not constitute medical advice.
> All findings should be verified by qualified medical professionals before clinical application.
> Always consult current clinical guidelines and a licensed healthcare provider for patient care decisions.

---"""

    def _build_footer(self, timestamp: datetime) -> str:
        """Build a report footer with generation metadata."""
        return f"""## Document Information

- **Document Type:** AI-Generated Medical Research Report
- **Generation System:** Medical Research Intelligence System (MRIS) v1.0
- **Pipeline:** 6-Agent CrewAI Sequential Workflow
- **Databases Searched:** PubMed (NCBI E-utilities), ClinicalTrials.gov API v2
- **Generated:** {timestamp.strftime("%B %d, %Y at %H:%M")}

---
*End of Report*
"""
