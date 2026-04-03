Medical Research Intelligence System
Project Overview
An AI-powered research assistant that helps healthcare professionals, researchers, and pharmaceutical companies analyze medical literature, extract clinical insights, and generate comprehensive research reports with citations.

🎯 Why This Stands Out to Recruiters

High-Impact Domain: Healthcare AI is booming with massive investment
Complex Orchestration: Demonstrates advanced agent coordination
Real-World Value: Solves actual pain points in medical research
Technical Depth: Shows API integration, data processing, and LLM orchestration
Portfolio-Ready: Creates tangible outputs (reports, visualizations, summaries)


🤖 Agent Architecture (6 Agents)
1. Research Coordinator Agent

Role: Project manager and orchestrator
Responsibilities:

Breaks down research queries into sub-tasks
Delegates to specialist agents
Ensures workflow coherence
Synthesizes final deliverables



2. Literature Scout Agent

Role: Medical literature researcher
Tools:

PubMed API integration
arXiv API for pre-prints
Google Scholar scraping
Semantic Scholar API


Responsibilities:

Search medical databases
Filter relevant papers by quality/citations
Extract metadata (authors, journals, impact factor)



3. Paper Analysis Agent

Role: Deep paper analyzer
Tools:

PDF extraction tools
Custom summarization functions
Citation graph builder


Responsibilities:

Extract key findings from papers
Identify methodology and limitations
Map relationships between studies
Detect contradictions in research



4. Clinical Data Interpreter Agent

Role: Statistical and clinical data expert
Tools:

Statistical analysis libraries
Data visualization tools
Clinical trial database APIs (ClinicalTrials.gov)


Responsibilities:

Interpret trial results
Analyze efficacy data
Compare treatment outcomes
Generate visual insights



5. Medical Knowledge Validator Agent

Role: Fact-checker and quality assurance
Tools:

Medical ontology APIs (UMLS, MeSH)
Drug interaction databases
FDA approval databases


Responsibilities:

Verify medical claims
Check drug interactions
Validate terminology
Flag potential contradictions



6. Report Synthesis Agent

Role: Technical writer and document generator
Tools:

Document generation libraries
Citation formatting tools
Visualization embedding


Responsibilities:

Create comprehensive research reports
Format citations (APA, Vancouver style)
Generate executive summaries
Create visual presentations




🔧 Technical Implementation
Core Technologies
python- CrewAI (agent orchestration)
- LangChain (tool integration)
- OpenAI/Anthropic APIs (LLM backbone)
- PubMed/Semantic Scholar APIs (data sources)
- PyMuPDF/pdfplumber (PDF processing)
- Pandas/NumPy (data analysis)
- Plotly/Matplotlib (visualizations)
- FastAPI (optional: REST API wrapper)
Custom Tools to Build

PubMed Search Tool

Query PubMed with filters (date, journal, citations)
Return structured metadata


PDF Analyzer Tool

Extract full text from medical papers
Identify sections (abstract, methods, results, conclusion)


Citation Network Builder

Map paper relationships
Identify seminal studies


Clinical Trial Parser

Extract trial phases, endpoints, outcomes
Compare efficacy across studies


Medical Term Validator

Check against medical ontologies
Suggest standardized terminology


Trace : https://app.crewai.com/crewai_plus/ephemeral_trace_batches/26d4cbe0-20de-4af9-85cd-505466983ddb?access_code=TRACE-c0254da951



## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/medicalintegencesystem/config/agents.yaml` to define your agents
- Modify `src/medicalintegencesystem/config/tasks.yaml` to define your tasks
- Modify `src/medicalintegencesystem/crew.py` to add your own logic, tools and specific args
- Modify `src/medicalintegencesystem/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the MedicalIntegenceSystem Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The MedicalIntegenceSystem Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the Medicalintegencesystem Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
