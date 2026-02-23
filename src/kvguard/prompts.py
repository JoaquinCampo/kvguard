"""GSM8K prompt loading and formatting."""

from datasets import load_dataset
from loguru import logger

# Few-shot examples for chain-of-thought prompting
# fmt: off
FEWSHOT_EXAMPLES = [
    {
        "question": (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove"
            " today. After they are done, there will be 21 trees. How many trees did the"
            " grove workers plant today?"
        ),
        "answer": (
            "There are 15 trees originally. Then there were 21 trees after some more were"
            " planted. So there must have been 21 - 15 = 6. #### 6"
        ),
    },
    {
        "question": (
            "If there are 3 cars in the parking lot and 2 more cars arrive,"
            " how many cars are in the parking lot?"
        ),
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "question": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35,"
            " how many pieces do they have left in total?"
        ),
        "answer": (
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had"
            " 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
        ),
    },
    {
        "question": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12"
            " lollipops. How many lollipops did Jason give to Denny?"
        ),
        "answer": (
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny."
            " So he gave Denny 20 - 12 = 8. #### 8"
        ),
    },
    {
        "question": (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad."
            " How many toys does he have now?"
        ),
        "answer": (
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then"
            " that is 2 + 2 = 4 more toys. 5 + 4 = 9. #### 9"
        ),
    },
]
# fmt: on


def load_gsm8k(num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
    """Load GSM8K test prompts."""
    logger.info(f"Loading {num_prompts} GSM8K prompts...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)

    prompts = []
    for i, row in enumerate(ds):
        if i >= num_prompts:
            break
        # Extract ground truth answer (after ####)
        answer_text = row["answer"]
        gt = answer_text.split("####")[-1].strip().replace(",", "")
        prompts.append(
            {
                "id": f"gsm8k_{i}",
                "question": row["question"],
                "ground_truth": gt,
                "full_answer": answer_text,
            }
        )
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def format_prompt(question: str, num_fewshot: int = 3) -> str:
    """Format a GSM8K question with few-shot chain-of-thought examples."""
    parts = [
        "Solve the following math problem step by step.",
        "Show your reasoning and give the final answer after ####.\n",
    ]

    for ex in FEWSHOT_EXAMPLES[:num_fewshot]:
        parts.append(f"Q: {ex['question']}")
        parts.append(f"A: {ex['answer']}\n")

    parts.append(f"Q: {question}")
    parts.append("A:")

    return "\n".join(parts)
