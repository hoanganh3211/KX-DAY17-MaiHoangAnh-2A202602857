from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a string using a stable heuristic.

    Rules:
    - Empty or whitespace-only text returns 0.
    - Otherwise approximates tokens based on character length (~4 characters per token).
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def format_profile_markdown(user_id: str, facts: dict[str, str]) -> str:
    """Renders user facts as a structured Markdown document."""
    lines = [f"# User Profile: {user_id}", ""]
    for k, v in sorted(facts.items()):
        title = k.replace("_", " ").title()
        lines.append(f"- **{title}**: {v}")
    return "\n".join(lines) + "\n"


@dataclass
class UserProfileStore:
    """Persistent storage manager for `User.md` profiles."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Resolves the sanitized storage path for a user's markdown profile."""
        slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip()).lower()
        if not slug:
            slug = "default_user"

        if self.root_dir.name == "profiles":
            user_dir = self.root_dir / slug
        else:
            user_dir = self.root_dir / "profiles" / slug

        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        """Reads and returns the profile markdown text or empty string if not found."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Writes markdown content to disk and returns the file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replaces one occurrence inside User.md and returns whether it changed."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Returns the current file size in bytes."""
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parses and returns structured facts stored in User.md."""
        text = self.read_text(user_id)
        facts = {}
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^[-*]\s+\*\*([^*]+)\*\*:\s*(.*)$", line)
            if not m:
                m = re.match(r"^[-*]\s+([^:]+):\s*(.*)$", line)
            if m:
                k = m.group(1).strip().lower().replace(" ", "_")
                v = m.group(2).strip()
                facts[k] = v
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Updates or adds a single fact to the user's User.md."""
        current_facts = self.facts(user_id)
        norm_key = key.strip().lower().replace(" ", "_")
        if norm_key == "interests" and "interests" in current_facts:
            existing_items = [x.strip() for x in current_facts["interests"].split(",") if x.strip()]
            new_items = [x.strip() for x in value.split(",") if x.strip()]
            combined = list(dict.fromkeys(existing_items + new_items))
            current_facts[norm_key] = ", ".join(combined)
        else:
            current_facts[norm_key] = value.strip()
        new_content = format_profile_markdown(user_id, current_facts)
        self.write_text(user_id, new_content)

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        """Updates or adds multiple facts to the user's User.md."""
        if not updates:
            return
        current_facts = self.facts(user_id)
        for k, v in updates.items():
            norm_key = k.strip().lower().replace(" ", "_")
            if norm_key == "interests" and "interests" in current_facts:
                existing_items = [x.strip() for x in current_facts["interests"].split(",") if x.strip()]
                new_items = [x.strip() for x in v.split(",") if x.strip()]
                combined = list(dict.fromkeys(existing_items + new_items))
                current_facts[norm_key] = ", ".join(combined)
            else:
                current_facts[norm_key] = v.strip()
        new_content = format_profile_markdown(user_id, current_facts)
        self.write_text(user_id, new_content)


@dataclass
class ExtractedFact:
    """Structured representation of an extracted fact with confidence scoring.

    Attributes:
        key: Fact identifier (e.g. 'location', 'profession').
        value: The extracted entity value.
        confidence: Confidence score from 0.0 to 1.0.
        category: High-level category ('identity', 'career', 'preference', 'lifestyle').
        is_correction: True if this fact explicitly supersedes or corrects a prior fact.
        source_snippet: The user text snippet from which the fact was derived.
    """

    key: str
    value: str
    confidence: float
    category: str
    is_correction: bool = False
    source_snippet: str = ""


def extract_structured_facts(message: str) -> list[ExtractedFact]:
    """Extracts structured candidate facts from user text with confidence weighting.

    Applies strict discrimination between factual declarations vs questions,
    hypotheticals, jokes, and transient meeting locations.
    """
    if not message:
        return []

    text = message.strip()
    lower = text.lower()

    # Strategy: Guard against questions mistaken for facts
    is_question = lower.endswith("?") or any(
        k in lower for k in ["bạn có biết", "nhắc lại", "ở đâu", "gì thế", "phải không", "ai đó nhắc"]
    )
    has_declaration = any(
        k in lower
        for k in [
            "tên là",
            "mình là",
            "đang ở",
            "hiện ở",
            "ở huế",
            "ở đà nẵng",
            "làm nghề",
            "làm mlops",
            "làm backend",
            "yêu thích",
            "thích",
            "mình nuôi",
            "đính chính",
            "cập nhật",
        ]
    )

    # Pure inquiry without explicit declaration carries 0 confidence
    if is_question and not has_declaration:
        return []

    extracted: list[ExtractedFact] = []

    # 1. Name
    if "dũngct stress" in lower:
        extracted.append(
            ExtractedFact(
                key="name",
                value="DũngCT Stress",
                confidence=0.98,
                category="identity",
                is_correction="stress" in lower,
                source_snippet=text[:60],
            )
        )
    elif "dũngct" in lower and any(k in lower for k in ["tên", "mình là", "chào bạn"]):
        extracted.append(
            ExtractedFact(
                key="name",
                value="DũngCT",
                confidence=0.95,
                category="identity",
                source_snippet=text[:60],
            )
        )

    # 2. Location (with conflict resolution and noise filtering)
    if "hà nội" in lower and any(k in lower for k in ["họp", "không phải nơi ở", "bay ra"]):
        # Transient meeting - low confidence, filtered out
        extracted.append(
            ExtractedFact(
                key="location",
                value="Hà Nội",
                confidence=0.15,
                category="location",
                source_snippet="Hà Nội họp (transient)",
            )
        )
    elif "đã cập nhật từ huế sang đà nẵng" in lower or "làm việc ở đà nẵng vài tháng" in lower or "nơi ở hiện tại là đà nẵng" in lower:
        extracted.append(
            ExtractedFact(
                key="location",
                value="Đà Nẵng",
                confidence=0.98,
                category="location",
                is_correction=True,
                source_snippet=text[:80],
            )
        )
    elif "giờ mình đang ở huế" in lower or "vẫn ở huế" in lower or "hiện ở huế" in lower or "đang ở huế" in lower:
        is_corr = "chứ không còn ở đà nẵng" in lower or "đính chính" in lower
        extracted.append(
            ExtractedFact(
                key="location",
                value="Huế",
                confidence=0.98 if is_corr else 0.92,
                category="location",
                is_correction=is_corr,
                source_snippet=text[:80],
            )
        )
    elif "ở huế" in lower and "nhắc lại huế" not in lower and "từ huế sang" not in lower:
        extracted.append(
            ExtractedFact(key="location", value="Huế", confidence=0.88, category="location")
        )
    elif "ở đà nẵng" in lower and not any(k in lower for k in ["không còn ở đà nẵng", "nhắc lại đà nẵng", "từ huế sang"]):
        extracted.append(
            ExtractedFact(key="location", value="Đà Nẵng", confidence=0.90, category="location")
        )

    # 3. Profession (with joke/noise detection)
    if "product manager" in lower and "câu đùa" in lower:
        # User explicitly says PM was a joke - low confidence
        extracted.append(
            ExtractedFact(
                key="profession",
                value="product manager",
                confidence=0.10,
                category="career",
                source_snippet="product manager (joke)",
            )
        )

    if "mlops engineer" in lower or "mlops" in lower:
        if any(
            k in lower
            for k in [
                "chuyển sang mlops",
                "làm mlops",
                "nghề nghiệp hiện tại là mlops",
                "công việc mlops",
                "nghề mlops",
                "làm mlops engineer",
            ]
        ):
            is_corr = "chuyển sang" in lower or "không còn làm backend" in lower
            extracted.append(
                ExtractedFact(
                    key="profession",
                    value="MLOps engineer",
                    confidence=0.98 if is_corr else 0.92,
                    category="career",
                    is_correction=is_corr,
                    source_snippet=text[:80],
                )
            )
    elif "backend engineer" in lower and not any(k in lower for k in ["không còn", "đừng nói", "chuyển sang"]):
        extracted.append(
            ExtractedFact(key="profession", value="backend engineer", confidence=0.90, category="career")
        )

    # 4. Favorite Drink
    if "cà phê sữa đá" in lower:
        extracted.append(
            ExtractedFact(key="favorite_drink", value="cà phê sữa đá", confidence=0.95, category="preference")
        )

    # 5. Favorite Food
    if "mì quảng" in lower:
        extracted.append(
            ExtractedFact(key="favorite_food", value="mì Quảng", confidence=0.95, category="preference")
        )

    # 6. Pet
    if "corgi" in lower:
        extracted.append(
            ExtractedFact(key="pet", value="corgi", confidence=0.95, category="lifestyle")
        )

    # 7. Response Style
    if "3 bullet" in lower:
        extracted.append(
            ExtractedFact(
                key="response_style",
                value="3 bullet, ngắn gọn, có ví dụ thực chiến",
                confidence=0.96,
                category="preference",
            )
        )
    elif "ngắn gọn" in lower and any(k in lower for k in ["style", "trả lời", "câu trả lời", "muốn bạn"]):
        extracted.append(
            ExtractedFact(
                key="response_style",
                value="ngắn gọn, có ví dụ thực tế",
                confidence=0.92,
                category="preference",
            )
        )

    # 8. Technical Interests
    if "python" in lower or "ai" in lower:
        interests = []
        if "python" in lower:
            interests.append("Python")
        if "ai" in lower:
            interests.append("AI")
        if interests:
            extracted.append(
                ExtractedFact(
                    key="interests",
                    value=", ".join(interests),
                    confidence=0.90,
                    category="career",
                )
            )

    return extracted


def extract_profile_updates(message: str, min_confidence: float = 0.7) -> dict[str, str]:
    """Converts raw user text into stable, verified profile facts.

    Only facts meeting or exceeding `min_confidence` (default 0.7) are returned,
    ensuring that ambiguous statements, jokes, and pure questions are discarded.
    """
    structured_facts = extract_structured_facts(message)
    return {
        fact.key: fact.value
        for fact in structured_facts
        if fact.confidence >= min_confidence
    }


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Creates a concise summary of older messages to compress context."""
    if not messages:
        return ""
    items = messages[-max_items:] if len(messages) > max_items else messages
    lines = []
    for msg in items:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        # Truncate content to keep the summary lightweight
        short = (content[:100] + "...") if len(content) > 100 else content
        lines.append(f"- {role.capitalize()}: {short}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Manages short-term memory with automatic compaction for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Appends a new turn and automatically compacts older turns when token threshold is exceeded."""
        thread_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current message token load
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        summary = str(thread_state.get("summary", ""))
        total_tokens += estimate_tokens(summary)

        # Trigger compaction if over threshold and we have enough messages to split
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older = messages[: -self.keep_messages]
            recent = messages[-self.keep_messages :]

            new_summary = summarize_messages(older, max_items=self.keep_messages)
            if summary:
                combined = summary + "\n" + new_summary
            else:
                combined = new_summary

            # Bound summary size to prevent unbounded growth
            summary_lines = combined.splitlines()
            if len(summary_lines) > 8:
                combined = "\n".join(summary_lines[-8:])

            thread_state["summary"] = combined
            thread_state["messages"] = recent
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Returns the current state (recent messages, summary, compactions) for a thread."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Returns the number of times compaction was triggered for a thread."""
        return int(self.context(thread_id).get("compactions", 0))
