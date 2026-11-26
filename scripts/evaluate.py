"""Run the evaluation arms on a labelled scenario set and write the comparison (#32).

    PYTHONPATH=src python scripts/evaluate.py data/scenarios/elec2.jsonl

Arms 1-3 always run. Arms 5-6 (LLM agent, with and without memory) run when
GROQ_API_KEY is set; their replies are cached under results/llm_cache, so a
rerun costs nothing. Arm 4 (classifier, #31) is added here once it exists.

Episodes are fed in file order, which matters for arm 6: memory only holds
episodes that came earlier.
"""

import json
import os
import sys
from pathlib import Path

from autonoma.agent.arms import alert_only, always_retrain
from autonoma.agent.llm import GroqComplete, LLMAgent
from autonoma.agent.memory import IncidentMemory
from autonoma.agent.rules import RuleArm
from autonoma.core.bundle import read_jsonl
from autonoma.evaluation.metrics import evaluate

OUT = Path("results/evaluation")


def main(scenarios: Path) -> None:
    rows = read_jsonl(scenarios)
    arms = {
        "1 alert-only": lambda: alert_only,
        "2 always-retrain": lambda: always_retrain,
        "3 threshold rules": RuleArm,
    }
    llm = None
    if os.getenv("GROQ_API_KEY"):
        llm = GroqComplete()
        OUT.mkdir(parents=True, exist_ok=True)
        memory_db = OUT / "incident_memory.db"
        memory_db.unlink(missing_ok=True)  # each run starts with an empty memory
        arms["5 llm agent"] = lambda: LLMAgent(llm)
        # ponytail: the faithfulness replay reuses this memory as it stands at the end of
        # the run, slightly more than the agent knew at decision time.
        memory = IncidentMemory(f"sqlite:///{memory_db}")
        arms["6 llm agent + memory"] = lambda: LLMAgent(llm, memory)

    report = evaluate(arms, rows)
    if llm:
        report["llm_usage"] = {
            "model": llm.model,
            "api_calls": llm.api_calls,
            "cache_hits": llm.cache_hits,
            "prompt_tokens": llm.prompt_tokens,
            "completion_tokens": llm.completion_tokens,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
