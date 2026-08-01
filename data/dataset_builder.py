import argparse
import json
import random
import re
import sys
from pathlib import Path

import wandb
from datasets import load_dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import WANDB_PROJECT  # noqa: E402

from schema import DISTRACTOR_SEP, MASK_TOKEN, SEP_TOKEN, TaskType, TrainingTask

MAX_CONTEXT_CHARS = 2000
MAX_EXAMPLES_PER_TASK = 16_667
MASKING_CHANCE = 0.3
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42
DATASET_ARTIFACT_NAME = "qa-pair-distractor-dataset"

# Synthetic short-answer distractor settings.
SHORT_ANSWER_MAX_WORDS = 3
NUM_DISTRACTORS = 3
NUMERIC_SHIFT_RANGE = (1, 50)  # +/- range applied to numeric answers


def build_qa_pair_examples(squad_split, seed: int) -> list[TrainingTask]:
    """Convert a SQuAD split into qa_pair examples with answer masking.

    Args:
        squad_split: A HuggingFace SQuAD dataset split.
        seed: Random seed controlling which rows get their answer masked.

    Returns:
        List of qa_pair TrainingTask objects. Rows without a valid answer
        are skipped.
    """
    rng = random.Random(seed)
    examples = []

    for row in squad_split:
        answers = row["answers"]["text"]
        if not answers:
            continue

        context = row["context"][:MAX_CONTEXT_CHARS]
        answer = answers[0]
        question = row["question"]

        answer_for_input = MASK_TOKEN if rng.random() < MASKING_CHANCE else answer
        input_text = f"generate qa pair: answer: {answer_for_input} context: {context}"
        target = f"{answer} {SEP_TOKEN} {question}"

        examples.append(TrainingTask(TaskType.QA_PAIR, input_text, target))

    return examples


def build_distractor_examples(race_split) -> list[TrainingTask]:
    """Convert a RACE split into distractor generation examples.

    Args:
        race_split: A HuggingFace RACE dataset split.

    Returns:
        List of Distractor TrainingTask objects. Malformed rows are skipped.
    """
    examples = []
    for row in race_split:
        options = row["options"]
        answer_letter = row["answer"]

        if len(options) != 4 or answer_letter not in "ABCD":
            continue

        answer_idx = "ABCD".index(answer_letter)
        correct_answer = options[answer_idx]
        distractors = [opt for i, opt in enumerate(options) if i != answer_idx]

        context = row["article"][:MAX_CONTEXT_CHARS]
        input_text = (
            f"generate distractor: question: {row['question']} "
            f"answer: {correct_answer} context: {context}"
        )
        target = DISTRACTOR_SEP.join(distractors)

        examples.append(TrainingTask(TaskType.DISTRACTOR, input_text, target))

    return examples


def _extract_numbers_from_text(text: str) -> list[str]:
    """Extract standalone numeric tokens (years, counts, etc.) from text.

    Args:
        text: Text to scan.

    Returns:
        List of matched number strings (as they appear, no dedup).
    """
    return re.findall(r"\b\d+\b", text)


def _shift_number(value: str, rng: random.Random) -> str:
    """Produce a plausible wrong number by shifting a numeric string.

    Args:
        value: Original numeric string (e.g. a year or count).
        rng: Random generator for the shift amount and direction.

    Returns:
        A shifted numeric string. If `value` is not purely numeric, it is
        returned unchanged (defensive fallback, should not normally happen
        given the caller only invokes this on digit-only answers).
    """
    if not value.isdigit():
        return value
    shift = rng.randint(*NUMERIC_SHIFT_RANGE) * rng.choice([-1, 1])
    shifted = max(0, int(value) + shift)
    return str(shifted)


def _build_numeric_distractors(answer: str, context: str, rng: random.Random) -> list[str]:
    """Build distractors for a numeric answer using in-context numbers and shifts.

    Args:
        answer: The correct numeric answer.
        context: Source passage, scanned for other numbers to reuse as
            distractors (usually more plausible than a fully synthetic one).
        rng: Random generator.

    Returns:
        List of exactly NUM_DISTRACTORS distractor strings, guaranteed
        distinct from `answer` and from each other.
    """
    candidates = {n for n in _extract_numbers_from_text(context) if n != answer}

    while len(candidates) < NUM_DISTRACTORS:
        candidates.add(_shift_number(answer, rng))
        candidates.discard(answer)

    return rng.sample(sorted(candidates), NUM_DISTRACTORS)


def _build_text_distractors(
    answer: str, answer_pool: list[str], rng: random.Random
) -> list[str]:
    """Build distractors for a short text answer by sampling similar-length answers.

    Args:
        answer: The correct answer.
        answer_pool: Pool of other short SQuAD answers to sample from,
            pre-filtered to a similar word-count bucket by the caller.
        rng: Random generator.

    Returns:
        List of exactly NUM_DISTRACTORS distractor strings, guaranteed
        distinct from `answer` and from each other. Falls back to sampling
        without the length constraint if the pool is too small.
    """
    candidates = [a for a in answer_pool if a.lower() != answer.lower()]
    if len(candidates) < NUM_DISTRACTORS:
        return []  # Pool too small; caller skips this example.

    return rng.sample(candidates, NUM_DISTRACTORS)


def build_synthetic_short_distractor_examples(squad_split, seed: int) -> list[TrainingTask]:
    """Build distractor examples for short factual answers, derived from SQuAD.

    Closes the distribution gap between qa_pair's short SQuAD-style answers
    and the distractor model, which otherwise only ever saw RACE's long
    phrase-style answers during training.

    Args:
        squad_split: A HuggingFace SQuAD dataset split.
        seed: Random seed for sampling.

    Returns:
        List of Distractor TrainingTask objects built from short answers.
    """
    rng = random.Random(seed)

    # First pass: collect all short answers, bucketed by word count, to use
    # as a sampling pool for text-based distractors.
    short_answer_pool_by_length: dict[int, list[str]] = {}
    rows = []
    for row in squad_split:
        answers = row["answers"]["text"]
        if not answers:
            continue
        answer = answers[0]
        word_count = len(answer.split())
        if word_count == 0 or word_count > SHORT_ANSWER_MAX_WORDS:
            continue

        rows.append(row)
        short_answer_pool_by_length.setdefault(word_count, []).append(answer)

    # Second pass: build a distractor example per short-answer row.
    examples = []
    for row in rows:
        answer = row["answers"]["text"][0]
        question = row["question"]
        context = row["context"][:MAX_CONTEXT_CHARS]
        word_count = len(answer.split())

        if answer.isdigit():
            distractors = _build_numeric_distractors(answer, context, rng)
        else:
            pool = short_answer_pool_by_length.get(word_count, [])
            distractors = _build_text_distractors(answer, pool, rng)

        if not distractors:
            continue

        input_text = (
            f"generate distractor: question: {question} "
            f"answer: {answer} context: {context}"
        )
        target = DISTRACTOR_SEP.join(distractors)
        examples.append(TrainingTask(TaskType.DISTRACTOR, input_text, target))

    return examples


def sample_examples(examples: list[TrainingTask], max_examples: int, seed: int) -> list[TrainingTask]:
    """Sample a bounded subset of examples for a balanced dataset size.

    Args:
        examples: List of examples to sample from.
        max_examples: Maximum number of examples to return.
        seed: Random seed for reproducibility.

    Returns:
        The original list if it's already within the limit, otherwise a
        shuffled random subset of size `max_examples`.
    """
    if len(examples) <= max_examples:
        return examples

    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


def split_examples(
    examples: list[TrainingTask], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[TrainingTask], list[TrainingTask], list[TrainingTask]]:
    """Shuffle and split examples into train/val/test subsets.

    Args:
        examples: Examples to split.
        val_ratio: Fraction reserved for validation.
        test_ratio: Fraction reserved for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) example lists.
    """
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test


def write_jsonl(examples: list[TrainingTask], path: Path) -> None:
    """Write a list of examples to a JSONL file.

    Args:
        examples: Examples to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def log_dataset_artifact(output_path: Path, counts: dict) -> None:
    """Log the processed dataset directory as a versioned W&B Artifact.

    Args:
        output_path: Directory containing train/val/test JSONL files.
        counts: Dict with per-split example counts, stored as metadata.
    """
    run = wandb.init(project=WANDB_PROJECT, job_type="build-dataset")
    artifact = wandb.Artifact(
        name=DATASET_ARTIFACT_NAME,
        type="dataset",
        metadata={
            "source_versions": {"squad": "plain_text/1.1", "race": "all"},
            "masking_chance": MASKING_CHANCE,
            **counts,
        },
    )
    artifact.add_dir(str(output_path))
    run.log_artifact(artifact)
    run.finish()
    logger.info(f"Dataset logged as W&B Artifact '{DATASET_ARTIFACT_NAME}'")


def main(output_dir: str) -> None:
    """Build the full multi-task dataset, write splits, and log to W&B.

    Args:
        output_dir: Directory where the resulting JSONL files are written.
    """
    output_path = Path(output_dir)

    logger.info("Loading SQuAD...")
    squad = load_dataset("squad")["train"]

    logger.info("Loading RACE (this can take a while)...")
    race = load_dataset("race", "all")["train"]

    logger.info("Building qa_pair examples (with answer masking)...")
    qa_pair_examples = build_qa_pair_examples(squad, RANDOM_SEED)

    logger.info("Building distractor examples from RACE (long-phrase answers)...")
    race_distractor_examples = build_distractor_examples(race)

    logger.info("Building synthetic distractor examples from SQuAD (short answers)...")
    synthetic_distractor_examples = build_synthetic_short_distractor_examples(squad, RANDOM_SEED)

    distractor_examples = race_distractor_examples + synthetic_distractor_examples

    logger.info("Sampling examples to a balanced size per task...")
    qa_pair_examples = sample_examples(qa_pair_examples, MAX_EXAMPLES_PER_TASK, RANDOM_SEED)
    distractor_examples = sample_examples(distractor_examples, MAX_EXAMPLES_PER_TASK, RANDOM_SEED)

    logger.info(
        f"Counts -> qa_pair: {len(qa_pair_examples)}, distractor: {len(distractor_examples)} "
        f"(race: {len(race_distractor_examples)}, synthetic: {len(synthetic_distractor_examples)})"
    )

    all_train, all_val, all_test = [], [], []
    for task_examples in (qa_pair_examples, distractor_examples):
        train, val, test = split_examples(task_examples, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    write_jsonl(all_train, output_path / "train.jsonl")
    write_jsonl(all_val, output_path / "val.jsonl")
    write_jsonl(all_test, output_path / "test.jsonl")

    counts = {
        "train_count": len(all_train),
        "val_count": len(all_val),
        "test_count": len(all_test),
    }
    logger.info(f"Done. {counts}")

    log_dataset_artifact(output_path, counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the qa_pair/distractor dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to write train/val/test JSONL files.",
    )
    args = parser.parse_args()
    main(args.output_dir)
