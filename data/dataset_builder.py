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
MAX_EXAMPLES_PER_TASK = 50_000
MASKING_CHANCE = 0.3
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42

QA_PAIR_ARTIFACT_NAME = "qa-pair-dataset"
DISTRACTOR_ARTIFACT_NAME = "distractor-dataset"

SHORT_ANSWER_MAX_WORDS = 10
NUM_DISTRACTORS = 3
NUMERIC_SHIFT_RANGE = (1, 50)


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
        input_text = f"generate question answer: {answer_for_input}. {context}"
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
        input_text = f"generate distractors: {row['question']}. {correct_answer}. {context}"
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
        The original list if within the limit, otherwise a shuffled subset.
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


def write_splits_and_log(
    examples: list[TrainingTask],
    output_dir: Path,
    artifact_name: str,
    extra_metadata: dict,
) -> None:
    """Split, write, and log one task's examples as a versioned W&B Artifact.

    Args:
        examples: All examples for this task (pre-sampling cap already applied).
        output_dir: Directory to write this task's train/val/test JSONL files.
        artifact_name: W&B artifact name for this task's dataset.
        extra_metadata: Additional metadata to attach to the artifact.
    """
    train, val, test = split_examples(examples, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    counts = {"train_count": len(train), "val_count": len(val), "test_count": len(test)}
    logger.info(f"[{artifact_name}] {counts}")

    run = wandb.init(project=WANDB_PROJECT, job_type="build-dataset", name=f"build-{artifact_name}")
    artifact = wandb.Artifact(name=artifact_name, type="dataset", metadata={**extra_metadata, **counts})
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    run.finish()
    logger.info(f"Dataset logged as W&B Artifact '{artifact_name}'")


def main(output_dir: str) -> None:
    """Build both datasets independently, write splits, and log to W&B.

    Args:
        output_dir: Parent directory; qa_pair and distractor data are
            written to `output_dir/qa_pair/` and `output_dir/distractor/`
            respectively.
    """
    output_path = Path(output_dir)

    logger.info("Loading SQuAD...")
    squad = load_dataset("squad")["train"]

    logger.info("Loading RACE (this can take a while)...")
    race = load_dataset("race", "all")["train"]

    logger.info("Building qa_pair examples (with answer masking)...")
    qa_pair_examples = build_qa_pair_examples(squad, RANDOM_SEED)
    qa_pair_examples = sample_examples(qa_pair_examples, MAX_EXAMPLES_PER_TASK, RANDOM_SEED)

    logger.info("Building distractor examples...")
    distractor_examples = build_distractor_examples(race)
    distractor_examples = sample_examples(distractor_examples, MAX_EXAMPLES_PER_TASK, RANDOM_SEED)

    logger.info(
        f"Counts -> qa_pair: {len(qa_pair_examples)}, distractor: {len(distractor_examples)}"
    )

    write_splits_and_log(
        qa_pair_examples,
        output_path / "qa_pair",
        QA_PAIR_ARTIFACT_NAME,
        {"source": "squad/plain_text/1.1", "masking_chance": MASKING_CHANCE},
    )
    write_splits_and_log(
        distractor_examples,
        output_path / "distractor",
        DISTRACTOR_ARTIFACT_NAME,
        {
            "source_race": "race/all",
            "source_synthetic": "squad/plain_text/1.1 (heuristic)",
            "race_count": len(distractor_examples),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the qa_pair and distractor datasets.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Parent directory for qa_pair/ and distractor/ subfolders.",
    )
    args = parser.parse_args()
    main(args.output_dir)
