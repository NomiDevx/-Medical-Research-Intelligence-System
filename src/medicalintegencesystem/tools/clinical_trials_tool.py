"""
ClinicalTrials.gov Search Tool — integrates with ClinicalTrials.gov API v2.
Searches for relevant clinical trials by condition/intervention and retrieves
trial details including phase, status, endpoints, and results.
"""

import logging
import time
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

CT_API_BASE = "https://clinicaltrials.gov/api/v2"


class ClinicalTrialsSearchInput(BaseModel):
    """Input schema for ClinicalTrialsSearchTool."""
    query: str = Field(
        description="Search query for clinical trials. Can be a condition name, drug/intervention "
                    "name, or combination. Example: 'type 2 diabetes metformin', "
                    "'breast cancer immunotherapy', 'COVID-19 antiviral treatment'"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of trials to retrieve (1-20).",
        ge=1,
        le=20
    )
    status_filter: str = Field(
        default="COMPLETED",
        description="Trial status filter: 'COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING', "
                    "or 'ALL' to include all statuses."
    )
    phase_filter: str = Field(
        default="ALL",
        description="Phase filter: 'PHASE2', 'PHASE3', 'PHASE4', or 'ALL'."
    )


class ClinicalTrialsSearchTool(BaseTool):
    """
    Searches ClinicalTrials.gov API v2 for clinical trial data.
    Returns trial phases, status, enrollment, interventions, endpoints, and results.
    """
    name: str = "Clinical Trials Search Tool"
    description: str = (
        "Search ClinicalTrials.gov for clinical trial data related to a medical condition "
        "or intervention. Returns trial NCT IDs, phases, status, enrollment numbers, "
        "interventions, primary endpoints, and available results. Use to understand the "
        "clinical trial evidence landscape for a research topic."
    )
    args_schema: Type[BaseModel] = ClinicalTrialsSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        status_filter: str = "COMPLETED",
        phase_filter: str = "ALL"
    ) -> str:
        """Execute ClinicalTrials.gov search and return formatted trial data."""
        try:
            logger.info(f"[ClinicalTrialsTool] Searching: '{query}' | status={status_filter}")
            trials = self._search_trials(query, max_results, status_filter, phase_filter)

            if not trials:
                return (
                    f"No clinical trials found for query: '{query}' with status={status_filter}. "
                    f"Try using status_filter='ALL' or broader search terms."
                )

            return self._format_results(trials, query)

        except Exception as e:
            logger.error(f"[ClinicalTrialsTool] Error: {e}")
            return f"ClinicalTrials.gov search failed: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def _search_trials(
        self,
        query: str,
        max_results: int,
        status_filter: str,
        phase_filter: str
    ) -> list[dict]:
        """Fetch trial list from ClinicalTrials.gov API v2."""
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,EnrollmentCount,"
                "EnrollmentType,StartDate,CompletionDate,Condition,InterventionName,"
                "InterventionType,PrimaryOutcomeMeasure,SecondaryOutcomeMeasure,"
                "LeadSponsorName,BriefSummary,DetailedDescription,HasResults"
            ),
        }

        if status_filter != "ALL":
            params["filter.overallStatus"] = status_filter

        if phase_filter != "ALL":
            params["filter.phase"] = phase_filter

        time.sleep(0.5)
        response = requests.get(f"{CT_API_BASE}/studies", params=params, timeout=20)
        response.raise_for_status()

        data = response.json()
        studies = data.get("studies", [])
        return [self._extract_trial_data(s) for s in studies]

    def _extract_trial_data(self, study: dict) -> dict:
        """Extract and flatten relevant fields from a ClinicalTrials study record."""
        proto = study.get("protocolSection", {})
        id_module = proto.get("identificationModule", {})
        status_module = proto.get("statusModule", {})
        design_module = proto.get("designModule", {})
        arms_module = proto.get("armsInterventionsModule", {})
        outcomes_module = proto.get("outcomesModule", {})
        sponsor_module = proto.get("sponsorCollaboratorsModule", {})
        desc_module = proto.get("descriptionModule", {})
        cond_module = proto.get("conditionsModule", {})
        results_section = study.get("resultsSection", {})

        # Interventions
        interventions = []
        for interv in arms_module.get("interventions", [])[:5]:
            name = interv.get("name", "")
            itype = interv.get("type", "")
            if name:
                interventions.append(f"{name} ({itype})" if itype else name)

        # Primary outcomes
        primary_outcomes = [
            o.get("measure", "") for o in outcomes_module.get("primaryOutcomes", [])[:3]
        ]

        # Secondary outcomes
        secondary_outcomes = [
            o.get("measure", "") for o in outcomes_module.get("secondaryOutcomes", [])[:3]
        ]

        # Results availability
        has_results = bool(results_section)
        baseline_summary = ""
        if has_results:
            outcome_measures = results_section.get("outcomeMeasuresModule", {})
            baseline_summary = (
                f"Results available. "
                f"{len(outcome_measures.get('outcomeMeasures', []))} outcome measures reported."
            )

        return {
            "nct_id": id_module.get("nctId", "Unknown"),
            "title": id_module.get("briefTitle", id_module.get("officialTitle", "Unknown")),
            "status": status_module.get("overallStatus", "Unknown"),
            "phase": design_module.get("phases", ["Unknown"])[0] if design_module.get("phases") else "Unknown",
            "enrollment": design_module.get("enrollmentInfo", {}).get("count", "Unknown"),
            "start_date": status_module.get("startDateStruct", {}).get("date", "Unknown"),
            "completion_date": status_module.get("completionDateStruct", {}).get("date", "Unknown"),
            "conditions": ", ".join(cond_module.get("conditions", [])[:3]),
            "interventions": interventions,
            "sponsor": sponsor_module.get("leadSponsor", {}).get("name", "Unknown"),
            "primary_outcomes": primary_outcomes,
            "secondary_outcomes": secondary_outcomes,
            "summary": desc_module.get("briefSummary", "")[:500],
            "has_results": has_results,
            "results_summary": baseline_summary,
            "url": f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}",
        }

    def _format_results(self, trials: list[dict], query: str) -> str:
        """Format trial data into readable structured text."""
        # Stats summary
        phases = [t["phase"] for t in trials if t["phase"] != "Unknown"]
        completed = sum(1 for t in trials if t["status"] == "COMPLETED")
        with_results = sum(1 for t in trials if t["has_results"])

        lines = [
            f"## ClinicalTrials.gov Search Results",
            f"**Query:** `{query}`",
            f"**Trials Found:** {len(trials)}",
            f"**Completed:** {completed} | **With Published Results:** {with_results}",
            f"**Phases:** {', '.join(set(phases)) if phases else 'Mixed'}",
            "",
        ]

        for i, t in enumerate(trials, 1):
            lines.append(f"### Trial {i}: {t['nct_id']}")
            lines.append(f"**Title:** {t['title']}")
            lines.append(f"**Status:** {t['status']} | **Phase:** {t['phase']}")
            lines.append(f"**Enrollment:** {t['enrollment']} participants")
            lines.append(f"**Dates:** {t['start_date']} → {t['completion_date']}")
            lines.append(f"**Sponsor:** {t['sponsor']}")
            lines.append(f"**Conditions:** {t['conditions']}")
            if t["interventions"]:
                lines.append(f"**Interventions:** {', '.join(t['interventions'])}")
            if t["primary_outcomes"]:
                lines.append(f"**Primary Endpoints:** {'; '.join(t['primary_outcomes'])}")
            if t["secondary_outcomes"]:
                lines.append(f"**Secondary Endpoints:** {'; '.join(t['secondary_outcomes'])}")
            if t["summary"]:
                lines.append(f"**Summary:** {t['summary']}")
            lines.append(f"**Results Available:** {'✅ Yes' if t['has_results'] else '❌ No'}")
            if t["results_summary"]:
                lines.append(f"**Results:** {t['results_summary']}")
            lines.append(f"**URL:** {t['url']}")
            lines.append("")

        return "\n".join(lines)
