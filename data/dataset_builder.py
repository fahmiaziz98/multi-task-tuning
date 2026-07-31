"""Build the multi-task dataset from SQuAD (QA/QG) and RACE (Distractor),
and log it as a versioned W&B Artifact.

Downloads the source datasets via HuggingFace `datasets`, converts each into
the unified `Example` schema, splits into train/val/test, writes JSONL
files, and logs them as a W&B dataset artifact for lineage tracking.

Usage:
    python data/build_dataset.py --output_dir ./data/processed
"""

import argparse
import json
import random
from pathlib import Path

import wandb
from datasets import load_dataset
from loguru import logger

from schema import DISTRACTOR_SEP, TaskType, TrainingTask

MAX_CONTEXT_CHARS = 2000
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42
WANDB_PROJECT = "multi-task-t5-quiz-generator"


def build_question_answer_generation(squad_split, task_type: TaskType) -> list[TrainingTask]:
    """
    Convert a SQuAD split into:
        QG examples (context+answer -> question)
        QA examples (context+question -> answer)

    Args:
        squad_split: A Huggingface SQuAD dataset split.

    Returns:
        List of QG example object. Items without a validation answer are skipped.
    """
    examples = []
    for row in squad_split:

        answer = row["answers"]["text"]
        if not answer:
            continue

        context = row["context"][:MAX_CONTEXT_CHARS]
        question = row["question"]

        input_text = ""
        target = ""

        if task_type.value == TaskType.QG.value:
            input_text = f"generate question: context: {context} answer: {answer[0]}"
            target = question

        elif task_type.value == TaskType.QA.value:
            input_text = f"answer the question: question: {question} context: {context}"
            target = answer[0]

        if input_text and target:
            examples.append(
                TrainingTask(
                    task_type,
                    input_text,
                    target
                )
            )

    return examples


def build_distractor_generation(race_split) -> list[TrainingTask]:
    """
    Convert a RACE split into  distractor generation examples

    Args:
        race_split: A Huggingface RACE dataset split.

    Returns:
        List of Distractor example object. Mallformed rows are skipped.
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
            f"generate distractor: context: {context}"
            f"question: {row['question']} answer: {correct_answer}"
        )
        target = DISTRACTOR_SEP.join(distractors)

        examples.append(
            TrainingTask(
                TaskType.DISTRACTOR,
                input_text,
                target
            )
        )
    return examples


def split_examples(
    examples: list[TrainingTask],
    val_ratio: float,
    test_ratio: float,
    seed: int
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
        name="qg-qa-distractor-dataset",
        type="dataset",
        metadata={
            "source_versions": {"squad": "plain_text/1.1", "race": "all"},
            **counts,
        },
    )
    artifact.add_dir(str(output_path))
    run.log_artifact(artifact)
    run.finish()
    logger.info("Dataset logged as W&B Artifact 'qg-qa-distractor-dataset'")


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

    logger.info("Building QA examples...")
    qa_examples = build_question_answer_generation(squad, TaskType.QA)

    logger.info("Building QG examples...")
    qg_examples = build_question_answer_generation(squad, TaskType.QG)

    logger.info("Building Distractor examples...")
    distractor_examples = build_distractor_generation(race)

    logger.info(
        f"Counts -> QA: {len(qa_examples)}, QG: {len(qg_examples)}, "
        f"Distractor: {len(distractor_examples)}"
    )

    all_train, all_val, all_test = [], [], []
    for task_examples in (qa_examples, qg_examples, distractor_examples):
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
    print(f"Done. {counts}")

    log_dataset_artifact(output_path, counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the multi-task QG/QA/Distractor dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory to write train/val/test JSONL files.",
    )
    args = parser.parse_args()
    main(args.output_dir)
