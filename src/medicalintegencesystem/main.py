#!/usr/bin/env python
"""
Medical Research Intelligence System — Main Entry Point
Accepts a research query and orchestrates the 6-agent pipeline.

Usage:
    # Interactive mode (prompted input):
    uv run medicalintegencesystem

    # Command line argument:
    uv run medicalintegencesystem "What are the latest treatments for Type 2 Diabetes?"

    # Via run_crew script:
    uv run run_crew "Effect of GLP-1 agonists on cardiovascular outcomes in diabetic patients"
"""

import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env BEFORE importing crew
load_dotenv()

from medicalintegencesystem.crew import Medicalintegencesystem

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ─── Logging Setup ────────────────────────────────────────────────────────────
output_dir = Path(__file__).parents[3] / "output"
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "mris_run.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ─── Banner ───────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║        🏥  Medical Research Intelligence System (MRIS) v1.0  🏥            ║
║                                                                              ║
║  Powered by: CrewAI Multi-Agent Framework + DeepSeek LLM                   ║
║  Pipeline:   6 Specialized Agents | Sequential Workflow                     ║
║  Sources:    PubMed (NCBI) + ClinicalTrials.gov                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Agents:
  [1] 🎯 Research Coordinator    — Query decomposition & PICO framework
  [2] 🔍 Literature Scout        — PubMed search & paper retrieval
  [3] 📄 Paper Analyzer          — Deep analysis & cross-paper synthesis
  [4] 📊 Clinical Data Interpreter — ClinicalTrials.gov stats & trials
  [5] ✅ Medical Validator        — Fact-checking & GRADE evidence grading
  [6] 📝 Report Writer           — Publication-quality report generation
"""


def _get_query_from_args() -> str:
    """Get research query from CLI args or interactive prompt."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        logger.info(f"Research query from CLI: '{query}'")
        return query.strip()

    print("\n" + "─" * 78)
    print("Enter your medical research query below.")
    print("Examples:")
    print("  • What are the latest treatments for Type 2 Diabetes?")
    print("  • Efficacy of immunotherapy in non-small cell lung cancer")
    print("  • GLP-1 agonists cardiovascular outcomes in diabetic patients")
    print("─" * 78)

    query = input("\n🔬 Research Query: ").strip()
    if not query:
        print("ERROR: No research query provided. Exiting.")
        sys.exit(1)
    return query


def run():
    """
    Main execution function — runs the 6-agent research pipeline.
    Called by: `uv run medicalintegencesystem` or `uv run run_crew`
    """
    print(BANNER)

    query = _get_query_from_args()
    current_year = str(datetime.now().year)

    inputs = {
        "research_query": query,
        "current_year": current_year,
        "max_papers": "10",
    }

    print(f"\n{'═' * 78}")
    print(f"  Research Query : {query}")
    print(f"  Year           : {current_year}")
    print(f"  Max Papers     : {inputs['max_papers']}")
    print(f"{'═' * 78}\n")

    start_time = datetime.now()
    logger.info(f"[MRIS] Starting pipeline for query: '{query}'")

    try:
        result = Medicalintegencesystem().crew().kickoff(inputs=inputs)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"[MRIS] Pipeline completed in {duration:.1f}s")

        print(f"\n{'═' * 78}")
        print(f"  ✅ Pipeline Complete! ({duration:.1f}s)")
        print(f"  📁 Report saved to: output/medical_research_report.md")
        print(f"  📋 Execution log:  output/crew_execution.log")
        print(f"{'═' * 78}\n")

        return result

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"[MRIS] Pipeline failed after {duration:.1f}s: {e}", exc_info=True)
        print(f"\n{'═' * 78}")
        print(f"  ❌ Pipeline Failed: {type(e).__name__}")
        print(f"  💬 Error: {e}")
        print(f"  📋 Check logs: output/mris_run.log")
        print(f"{'═' * 78}\n")
        raise Exception(f"Medical Research Pipeline failed: {e}") from e


def train():
    """Train the crew for a given number of iterations."""
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Effect of metformin on HbA1c in T2DM"
    inputs = {
        "research_query": query,
        "current_year": str(datetime.now().year),
        "max_papers": "5",
    }
    try:
        Medicalintegencesystem().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=f"output/training_{datetime.now().strftime('%Y%m%d')}.pkl",
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"Training failed: {e}") from e


def replay():
    """Replay the crew execution from a specific task."""
    try:
        Medicalintegencesystem().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"Replay failed: {e}") from e


def test():
    """Test the crew execution and returns the results."""
    inputs = {
        "research_query": "What is the efficacy of statins in primary prevention of cardiovascular disease?",
        "current_year": str(datetime.now().year),
        "max_papers": "5",
    }
    try:
        Medicalintegencesystem().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-chat",
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"Test failed: {e}") from e


def run_with_trigger():
    """Run the crew with a JSON trigger payload (for automation/webhook use)."""
    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Pass JSON as argument: "
            "run_with_trigger '{\"research_query\": \"...\"}'"
        )

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON trigger payload: {e}") from e

    inputs = {
        "research_query": trigger_payload.get("research_query", ""),
        "current_year": trigger_payload.get("current_year", str(datetime.now().year)),
        "max_papers": str(trigger_payload.get("max_papers", 10)),
        "crewai_trigger_payload": trigger_payload,
    }

    if not inputs["research_query"]:
        raise Exception("Trigger payload must include 'research_query' field.")

    try:
        result = Medicalintegencesystem().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"Trigger run failed: {e}") from e


if __name__ == "__main__":
    run()
