from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # Fallback implementation provided below

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Reads JSON conversation dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Returns recall score (0.0 to 1.0) based on proportion of expected facts found."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score combining recall, structure, and length appropriateness."""
    if not answer or not answer.strip():
        return 0.0
    rec = recall_points(answer, expected)
    length = len(answer.strip())
    len_score = 1.0 if 20 <= length <= 600 else (0.5 if length < 20 else 0.8)
    has_struct = answer.strip().startswith("-") or answer.strip().endswith((".", "!", "?"))
    struct_score = 1.0 if has_struct else 0.8
    quality = 0.6 * rec + 0.25 * len_score + 0.15 * struct_score
    return round(min(1.0, quality), 3)


def run_agent_benchmark(agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: Any) -> BenchmarkRow:
    """Evaluates an agent over a benchmark conversation dataset.

    Flow:
    1. Feed all turns to the agent in the main thread.
    2. Track `agent tokens only` and `prompt tokens processed`.
    3. Ask recall questions in fresh threads (measuring cross-session recall).
    4. Compute average recall and heuristic quality.
    5. Record persistent memory growth (bytes) and compaction count.
    """
    all_recalls: list[float] = []
    all_qualities: list[float] = []
    all_users: set[str] = set()

    for conv in conversations:
        user_id = conv["user_id"]
        all_users.add(user_id)
        conv_id = conv.get("id", "conv")
        main_thread = f"{conv_id}_main"

        # 1. Feed conversation turns
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=main_thread, message=turn)

        # 2. Ask recall questions in fresh threads
        for idx, q_item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{conv_id}_recall_{idx}"
            res = agent.reply(user_id=user_id, thread_id=recall_thread, message=q_item["question"])
            answer = str(res.get("content", "") or res.get("reply", "") or res.get("response", ""))
            expected = q_item.get("expected_contains", [])
            rec = recall_points(answer, expected)
            qual = heuristic_quality(answer, expected)
            all_recalls.append(rec)
            all_qualities.append(qual)

    agent_tokens = agent.token_usage()
    prompt_tokens = agent.prompt_token_usage()
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    avg_quality = sum(all_qualities) / len(all_qualities) if all_qualities else 0.0

    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(u) for u in all_users)

    compactions = 0
    if hasattr(agent, "compaction_count"):
        compactions = agent.compaction_count()

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Formats benchmark rows into a GitHub-style markdown table with the 6 required metrics."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,}",
            f"{r.compactions:,}",
        ])

    if tabulate is not None:
        return tabulate(table_data, headers=headers, tablefmt="github")

    # Fallback Markdown formatter
    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in table_data
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def main() -> None:
    """Runs Standard and Long-Context Stress benchmarks, comparing Baseline vs Advanced."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_data_path)
    stress_convs = load_conversations(stress_data_path)

    def clean_profile_state(users: list[str]) -> None:
        profiles_dir = config.state_dir / "profiles"
        if not profiles_dir.exists():
            return
        for u in users:
            slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", u.strip()).lower()
            u_dir = profiles_dir / slug
            if u_dir.exists():
                shutil.rmtree(u_dir, ignore_errors=True)

    # 1. Standard Benchmark
    print("\n" + "=" * 70)
    print(" 1. STANDARD BENCHMARK (data/conversations.json)")
    print("=" * 70)
    clean_profile_state(["dungct"])
    base_std = BaselineAgent(config, force_offline=True)
    row_base_std = run_agent_benchmark("Baseline Agent", base_std, std_convs, config)

    clean_profile_state(["dungct"])
    adv_std = AdvancedAgent(config, force_offline=True)
    row_adv_std = run_agent_benchmark("Advanced Agent", adv_std, std_convs, config)

    print(format_rows([row_base_std, row_adv_std]))

    # 2. Long-Context Stress Benchmark
    print("\n" + "=" * 70)
    print(" 2. LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=" * 70)
    clean_profile_state(["dungct_stress"])
    base_stress = BaselineAgent(config, force_offline=True)
    row_base_stress = run_agent_benchmark("Baseline Agent", base_stress, stress_convs, config)

    clean_profile_state(["dungct_stress"])
    adv_stress = AdvancedAgent(config, force_offline=True)
    row_adv_stress = run_agent_benchmark("Advanced Agent", adv_stress, stress_convs, config)

    print(format_rows([row_base_stress, row_adv_stress]))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
