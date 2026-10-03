from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Builds an isolated test configuration pointing state_dir to tmp_path."""
    base_cfg = load_config()
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=base_cfg.data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=50,  # Low threshold so compaction triggers quickly in tests
        compact_keep_messages=2,
        model=base_cfg.model,
        judge_model=base_cfg.judge_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verifies that `User.md` can be created, read, edited, and queried for facts."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dungct_test"

    # 1. Fresh user has empty profile and 0 file size
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # 2. Write profile and verify persistence
    content = "# User Profile: dungct_test\n\n- **Name**: DũngCT\n- **Location**: Đà Nẵng\n"
    path = store.write_text(user_id, content)
    assert path.exists()
    assert store.file_size(user_id) > 0
    assert "Đà Nẵng" in store.read_text(user_id)

    # 3. Edit text (e.g. location correction)
    changed = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert changed is True
    assert "Huế" in store.read_text(user_id)
    assert "Đà Nẵng" not in store.read_text(user_id)

    # 4. Upsert structured facts
    store.upsert_fact(user_id, "Profession", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"
    assert facts.get("location") == "Huế"
    assert facts.get("name") == "DũngCT"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verifies that long threads trigger compaction once the token threshold is exceeded."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)
    thread_id = "thread_compact_test"
    user_id = "compact_user"

    assert agent.compaction_count(thread_id) == 0

    # Append turns that exceed the low threshold (50 tokens)
    for i in range(6):
        agent.reply(
            user_id=user_id,
            thread_id=thread_id,
            message=f"Lượt tin nhắn số {i} với độ dài tương đối để chắc chắn vượt ngưỡng compact threshold trong test.",
        )

    # Compaction should have fired at least once
    assert agent.compaction_count(thread_id) > 0
    ctx = agent.compact_memory.context(thread_id)
    assert len(ctx["messages"]) <= cfg.compact_keep_messages + 1
    assert len(str(ctx["summary"])) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verifies that AdvancedAgent remembers across threads while BaselineAgent forgets."""
    cfg = make_config(tmp_path)
    base = BaselineAgent(cfg, force_offline=True)
    adv = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct_cross_test"
    thread_1 = "thread_session_1"
    thread_2 = "thread_session_2"

    # Session 1: User introduces facts
    intro = "Chào bạn, mình tên là DũngCT, hiện ở Huế và làm MLOps engineer."
    base.reply(user_id=user_id, thread_id=thread_1, message=intro)
    adv.reply(user_id=user_id, thread_id=thread_1, message=intro)

    # Session 2: Recall question in a fresh thread
    recall_q = "Hiện tại mình đang ở đâu và làm nghề gì?"
    base_res = base.reply(user_id=user_id, thread_id=thread_2, message=recall_q)
    adv_res = adv.reply(user_id=user_id, thread_id=thread_2, message=recall_q)

    # Baseline Agent must NOT remember in a new thread
    assert "Huế" not in base_res["content"]
    assert "MLOps engineer" not in base_res["content"]

    # Advanced Agent MUST remember via persistent User.md
    assert "Huế" in adv_res["content"]
    assert "MLOps engineer" in adv_res["content"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compares prompt load of baseline vs advanced on a long conversation thread."""
    cfg = make_config(tmp_path)
    base = BaselineAgent(cfg, force_offline=True)
    adv = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct_stress_test"
    thread_id = "long_thread_comparison"

    long_message = (
        "Nhiệm vụ Artemis III của NASA hướng đến việc đưa phi hành gia quay trở lại Mặt Trăng. "
        "Việc tích hợp nhiều thành phần quỹ đạo và kiểm thử hệ thống hỗ trợ sự sống đòi hỏi quản trị "
        "các cột mốc phụ thuộc kỹ thuật một cách chặt chẽ trước khi triển khai chính thức."
    )

    for _ in range(12):
        base.reply(user_id=user_id, thread_id=thread_id, message=long_message)
        adv.reply(user_id=user_id, thread_id=thread_id, message=long_message)

    base_prompt_tokens = base.prompt_token_usage(thread_id)
    adv_prompt_tokens = adv.prompt_token_usage(thread_id)

    # Advanced must trigger compaction
    assert adv.compaction_count(thread_id) > 0

    # Advanced prompt tokens must be substantially lower than uncompressed baseline
    assert adv_prompt_tokens < base_prompt_tokens


def test_confidence_threshold_and_noise_filtering(tmp_path: Path) -> None:
    """Verifies bonus: confidence thresholding prevents jokes and transient noise from being stored."""
    from memory_store import extract_profile_updates, extract_structured_facts

    # 1. Joke text
    joke_msg = "Có lúc mình đùa chuyển sang product manager cho đỡ mệt, nhưng đó chỉ là câu đùa."
    facts = extract_structured_facts(joke_msg)
    assert any(f.key == "profession" and f.confidence < 0.5 for f in facts)
    # Filtered by default confidence threshold (0.7)
    updates = extract_profile_updates(joke_msg)
    assert "profession" not in updates

    # 2. Transient business trip
    meeting_msg = "Hà Nội chỉ là nơi mình vừa bay ra họp 2 ngày thôi nhé."
    facts = extract_structured_facts(meeting_msg)
    assert any(f.key == "location" and f.confidence < 0.5 for f in facts)
    updates = extract_profile_updates(meeting_msg)
    assert "location" not in updates

    # 3. Explicit verified fact
    real_msg = "Mình đính chính là giờ mình đang ở Huế rồi."
    facts = extract_structured_facts(real_msg)
    assert any(f.key == "location" and f.confidence >= 0.9 for f in facts)
    updates = extract_profile_updates(real_msg)
    assert updates.get("location") == "Huế"


def test_question_not_mistaken_for_fact(tmp_path: Path) -> None:
    """Verifies bonus: pure inquiries and questions are not mistakenly stored as profile facts."""
    from memory_store import extract_profile_updates

    question_1 = "Hiện tại mình đang ở đâu thế bạn?"
    assert extract_profile_updates(question_1) == {}

    question_2 = "Bạn có biết DũngCT thích uống gì không?"
    assert extract_profile_updates(question_2) == {}

