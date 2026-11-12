"""Evaluation-prompt parity tests.

The expensive failure we are guarding against: training renders prompts one way,
evaluation another, and GSM8K accuracy collapses even though the checkpoint is
fine. These tests need no torch and no datasets.
"""

from atom_slm import evaluate, prompts


class PlainTokenizer:
    """SmolLM2 base: no chat template."""

    chat_template = None


class ChatTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(message["content"] for message in messages)


def test_default_is_zero_shot():
    # The SFT data is 0-shot; few-shot prompts measure a different task.
    assert evaluate.parse_args([]).num_fewshot == 0


def test_plain_eval_prompt_equals_training_prefix():
    question = "A farmer has 3 apples and buys 4 more. How many?"
    eval_prompt = evaluate.build_eval_prompt(PlainTokenizer(), question)
    train_text = prompts.plain_example(question, "He has 7 apples.\n\n#### 7")

    assert eval_prompt == prompts.plain_prompt(question)
    assert train_text.startswith(eval_prompt)
    assert eval_prompt.endswith("### Solution:\n")


def test_zero_shot_prefix_is_empty_and_does_not_load_gsm8k():
    assert evaluate.build_fewshot_prefix(0) == ""


def test_fewshot_blocks_keep_the_training_cue():
    rows = [
        {"question": "Q1", "answer": "Step one.\n#### 3"},
        {"question": "Q2", "answer": "Step two.\n#### 5"},
    ]

    class FakeDataset:
        def select(self, indices):
            return [rows[i] for i in indices]

        def __len__(self):
            return len(rows)

    original = evaluate.load_gsm8k
    evaluate.load_gsm8k = lambda split="train": FakeDataset()
    try:
        prefix = evaluate.build_fewshot_prefix(2)
    finally:
        evaluate.load_gsm8k = original

    assert prefix.count("### Solution:") == 2
    assert prefix.endswith("\n\n")
    assert "#### 3" in prefix and "#### 5" in prefix

    prompt = evaluate.build_eval_prompt(PlainTokenizer(), "Q3", prefix)
    assert prompt.startswith(prefix)
    assert prompt.endswith(prompts.plain_prompt("Q3"))


def test_chat_tokenizer_keeps_the_instruction():
    prompt = evaluate.build_eval_prompt(ChatTokenizer(), "What is 2+2?")
    assert prompts.INSTRUCTION in prompt
    assert "What is 2+2?" in prompt
