"""Tests for prompts module."""

from kvguard.prompts import FEWSHOT_EXAMPLES, format_prompt


class TestFewshotExamples:
    def test_has_five_examples(self) -> None:
        assert len(FEWSHOT_EXAMPLES) == 5

    def test_each_has_question_and_answer(self) -> None:
        for ex in FEWSHOT_EXAMPLES:
            assert "question" in ex
            assert "answer" in ex
            assert len(ex["question"]) > 0
            assert len(ex["answer"]) > 0

    def test_answers_contain_hash_delimiter(self) -> None:
        """Each answer should contain #### with a numeric answer."""
        for ex in FEWSHOT_EXAMPLES:
            assert "####" in ex["answer"]


class TestFormatPrompt:
    def test_includes_question(self) -> None:
        prompt = format_prompt("How many apples?", num_fewshot=0)
        assert "How many apples?" in prompt

    def test_includes_fewshot_examples(self) -> None:
        prompt = format_prompt("test", num_fewshot=3)
        # Should include the first 3 few-shot examples
        for ex in FEWSHOT_EXAMPLES[:3]:
            assert ex["question"] in prompt
            assert ex["answer"] in prompt

    def test_zero_fewshot(self) -> None:
        prompt = format_prompt("test", num_fewshot=0)
        # Should not include any few-shot examples
        for ex in FEWSHOT_EXAMPLES:
            assert ex["question"] not in prompt

    def test_fewshot_count_respected(self) -> None:
        prompt = format_prompt("test", num_fewshot=2)
        assert FEWSHOT_EXAMPLES[0]["question"] in prompt
        assert FEWSHOT_EXAMPLES[1]["question"] in prompt
        assert FEWSHOT_EXAMPLES[2]["question"] not in prompt

    def test_ends_with_answer_prefix(self) -> None:
        prompt = format_prompt("test", num_fewshot=0)
        assert prompt.rstrip().endswith("A:")

    def test_starts_with_instruction(self) -> None:
        prompt = format_prompt("test", num_fewshot=0)
        assert prompt.startswith("Solve the following math problem")

    def test_default_fewshot_is_three(self) -> None:
        prompt = format_prompt("test")
        # Default should include 3 examples
        assert FEWSHOT_EXAMPLES[2]["question"] in prompt
        assert FEWSHOT_EXAMPLES[3]["question"] not in prompt

    def test_qa_format(self) -> None:
        prompt = format_prompt("What is 2+2?", num_fewshot=1)
        assert "Q: What is 2+2?" in prompt
        assert "Q:" in prompt
        assert "A:" in prompt
