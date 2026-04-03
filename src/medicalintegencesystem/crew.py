"""
Medical Research Intelligence System — Crew Definition
Defines the 6-agent sequential pipeline with custom tools for each specialist agent.
Uses DeepSeek LLM via OpenAI-compatible endpoint.
"""

import os
import logging
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from medicalintegencesystem.tools import (
    PubMedSearchTool,
    ClinicalTrialsSearchTool,
    PDFTextExtractorTool,
    CitationFormatterTool,
    ReportGeneratorTool,
)

logger = logging.getLogger(__name__)


def get_deepseek_llm() -> LLM:
    """
    Configure DeepSeek LLM using CrewAI's LLM class.
    DeepSeek is OpenAI-compatible — we use the openai/ prefix with base_url override.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file."
        )

    return LLM(
        model="deepseek/deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.3,        # Lower temp for factual medical content
        max_tokens=4096,
        timeout=120,
    )


@CrewBase
class Medicalintegencesystem():
    """
    Medical Research Intelligence System Crew.

    6-agent sequential pipeline:
    1. Research Coordinator  — query decomposition & PICO framework
    2. Literature Scout      — PubMed search & paper retrieval
    3. Paper Analysis        — deep analysis & cross-paper synthesis
    4. Clinical Data Interpreter — ClinicalTrials.gov data & stats
    5. Medical Knowledge Validator — fact-checking & GRADE grading
    6. Report Synthesis      — publication-quality report generation
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    # ─── Agents ───────────────────────────────────────────────────────────────

    @agent
    def research_coordinator(self) -> Agent:
        """Research Coordinator — orchestrates the workflow and decomposes queries."""
        return Agent(
            config=self.agents_config['research_coordinator'],  # type: ignore[index]
            llm=get_deepseek_llm(),
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=2,
        )

    @agent
    def literature_scout(self) -> Agent:
        """Literature Scout — searches PubMed and retrieves paper metadata."""
        return Agent(
            config=self.agents_config['literature_scout'],  # type: ignore[index]
            llm=get_deepseek_llm(),
            tools=[
                PubMedSearchTool(),
                PDFTextExtractorTool(),
            ],
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            max_retry_limit=2,
        )

    @agent
    def paper_analysis(self) -> Agent:
        """Paper Analysis Agent — deep analysis and cross-paper synthesis."""
        return Agent(
            config=self.agents_config['paper_analysis'],  # type: ignore[index]
            llm=get_deepseek_llm(),
            tools=[
                PDFTextExtractorTool(),
            ],
            verbose=True,
            allow_delegation=False,
            max_iter=4,
            max_retry_limit=2,
        )

    @agent
    def clinical_data_interpreter(self) -> Agent:
        """Clinical Data Interpreter — fetches and analyzes ClinicalTrials.gov data."""
        return Agent(
            config=self.agents_config['clinical_data_interpreter'],  # type: ignore[index]
            llm=get_deepseek_llm(),
            tools=[
                ClinicalTrialsSearchTool(),
            ],
            verbose=True,
            allow_delegation=False,
            max_iter=4,
            max_retry_limit=2,
        )

    @agent
    def medical_knowledge_validator(self) -> Agent:
        """Medical Knowledge Validator — fact-checks and applies GRADE evidence grading."""
        return Agent(
            config=self.agents_config['medical_knowledge_validator'],  # type: ignore[index]
            llm=get_deepseek_llm(),
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=2,
        )

    @agent
    def report_synthesis(self) -> Agent:
        """Report Synthesis Agent — produces the final publication-quality report."""
        return Agent(
            config=self.agents_config['report_synthesis'],  # type: ignore[index]
            llm=get_deepseek_llm(),
            tools=[
                CitationFormatterTool(),
                ReportGeneratorTool(),
            ],
            verbose=True,
            allow_delegation=False,
            max_iter=4,
            max_retry_limit=2,
        )

    # ─── Tasks ────────────────────────────────────────────────────────────────

    @task
    def query_decomposition_task(self) -> Task:
        """Task 1: Break down the research query using PICO framework."""
        return Task(
            config=self.tasks_config['query_decomposition_task'],  # type: ignore[index]
        )

    @task
    def literature_search_task(self) -> Task:
        """Task 2: Search PubMed and retrieve relevant paper metadata."""
        return Task(
            config=self.tasks_config['literature_search_task'],  # type: ignore[index]
            context=[self.query_decomposition_task()],
        )

    @task
    def paper_analysis_task(self) -> Task:
        """Task 3: Deep analysis of retrieved papers with cross-paper synthesis."""
        return Task(
            config=self.tasks_config['paper_analysis_task'],  # type: ignore[index]
            context=[self.literature_search_task()],
        )

    @task
    def clinical_data_task(self) -> Task:
        """Task 4: Retrieve and interpret ClinicalTrials.gov data."""
        return Task(
            config=self.tasks_config['clinical_data_task'],  # type: ignore[index]
            context=[self.query_decomposition_task(), self.paper_analysis_task()],
        )

    @task
    def validation_task(self) -> Task:
        """Task 5: Validate all claims, terminology, and grade evidence quality."""
        return Task(
            config=self.tasks_config['validation_task'],  # type: ignore[index]
            context=[
                self.literature_search_task(),
                self.paper_analysis_task(),
                self.clinical_data_task(),
            ],
        )

    @task
    def report_generation_task(self) -> Task:
        """Task 6: Generate the final publication-quality research report."""
        return Task(
            config=self.tasks_config['report_generation_task'],  # type: ignore[index]
            context=[
                self.query_decomposition_task(),
                self.literature_search_task(),
                self.paper_analysis_task(),
                self.clinical_data_task(),
                self.validation_task(),
            ],
            output_file='output/medical_research_report.md',
        )

    # ─── Crew ────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """
        Assembles the Medical Research Intelligence System crew.
        Sequential process ensures each agent builds on the previous agent's output.
        """
        logger.info("[MRIS] Initializing 6-agent Medical Research Intelligence System...")

        return Crew(
            agents=self.agents,   # Auto-populated by @agent decorators
            tasks=self.tasks,     # Auto-populated by @task decorators
            process=Process.sequential,
            verbose=True,
            output_log_file='output/crew_execution.log',
        )
