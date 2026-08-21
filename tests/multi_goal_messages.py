"""Lightweight checks for multi-object VLM conversation construction."""

from longnav.utils.rollout_core import (
    build_goal_switch_messages,
    build_initial_messages,
)


START_TEMPLATE = [
    {"role": "user", "content": [{"type": "text", "text": "Find $instr_or_goal"}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]
GOAL_TEMPLATE = [
    {"role": "user", "content": [{"type": "text", "text": "Existing next goal: $instr_or_goal"}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]
OBS = {
    "instr_or_goal": "chair",
    "goal_sequence": ["chair", "plant", "bed"],
}


def text_items(messages):
    return [
        item["text"]
        for message in messages
        for item in message["content"]
        if "text" in item
    ]


def main():
    base = {
        "convo_start_template": START_TEMPLATE,
        "convo_goal_template": GOAL_TEMPLATE,
        "reveal_all_goals": False,
        "multi_goal_transition": "next_goal",
    }

    assert text_items(build_initial_messages(base, OBS))[0] == "Find chair"
    assert text_items(build_goal_switch_messages(base, "plant"))[0] == "Existing next goal: plant"

    revealed = base | {"reveal_all_goals": True}
    expected_initial = "Find chair, then plant, then bed, in that order"
    assert text_items(build_initial_messages(revealed, OBS))[0] == expected_initial

    expected_transition = "You have found the previous target. Your new target is “plant”."
    expected = {
        "stop_then_next": ["**stop**", expected_transition, "**forward**"],
        "stop_only": ["**stop**", "**forward**"],
        "found_only": ["**found**", "**forward**"],
        "found_then_next": ["**found**", expected_transition, "**forward**"],
    }
    for mode, texts in expected.items():
        cfg = revealed | {"multi_goal_transition": mode}
        assert text_items(build_goal_switch_messages(cfg, "plant")) == texts

    try:
        build_goal_switch_messages(revealed | {"multi_goal_transition": "bad"}, "plant")
    except ValueError as exc:
        assert "invalid multi_goal_transition" in str(exc)
    else:
        raise AssertionError("invalid transition mode was accepted")

    try:
        build_initial_messages(revealed, {"instr_or_goal": "chair"})
    except ValueError as exc:
        assert "goal_sequence" in str(exc)
    else:
        raise AssertionError("missing goal sequence was accepted")

    print("OK: all-goals message construction")


if __name__ == "__main__":
    main()
