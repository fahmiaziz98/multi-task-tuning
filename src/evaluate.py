"""Evaluate a fine-tuned multi-task T5 model separately for each task, and
attach the results to the corresponding W&B training run.

Metrics per task:
    - QA: Exact Match (EM) and F1 (token overlap).
    - QG: BLEU-4 and ROUGE-L.
    - Distractor: distinct-1/2 (diversity) plus a validity check that
      distractors are not identical to the gold answer.

Usage:
    python src/evaluate.py --checkpoint ./checkpoints/multitask-t5 \
        --test_file ./data/processed/test.jsonl --run_name run-xxxx
"""

import argparse
import json
import re
from collections import defaultdict

import evaluate as hf_evaluate
import torch
import wandb
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from schema import DISTRACTOR_SEP, TaskType

WANDB_PROJECT = "multitask-t5-quiz-generator"
BATCH_SIZE = 16
MAX_TARGET_LENGTH = 64


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for EM/F1 comparison.

    Args:
        text: Raw text.

    Returns:
        Normalized text.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def compute_em_f1(prediction: str, reference: str) -> tuple[float, float]:
    """Compute exact match and token-level F1 between a prediction and reference.

    Args:
        prediction: Model-generated answer.
        reference: Gold answer.

    Returns:
        Tuple of (exact_match, f1), each in [0, 1].
    """
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    exact_match = float(pred_norm == ref_norm)

    pred_tokens = pred_norm.split()
    ref_tokens = ref_norm.split()
    if not pred_tokens or not ref_tokens:
        return exact_match, float(pred_tokens == ref_tokens)

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)
    if num_common == 0:
        return exact_match, 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return exact_match, f1


def distinct_n(texts: list[str], n: int) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams.

    Args:
        texts: List of generated texts (e.g. individual distractors).
        n: The n-gram size.

    Returns:
        Distinct-n score in [0, 1]. Returns 0.0 if there are no n-grams.
    """
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def generate_predictions(
    model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, inputs: list[str], device: str
) -> list[str]:
    """Run batched generation over a list of input texts.

    Args:
        model: The fine-tuned Seq2Seq model.
        tokenizer: Matching tokenizer.
        inputs: List of raw input strings (already including task prefixes).
        device: Torch device string, e.g. "cuda" or "cpu".

    Returns:
        List of decoded prediction strings, in the same order as `inputs`.
    """
    predictions = []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[i : i + BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(**encoded, max_length=MAX_TARGET_LENGTH, num_beams=4)

        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions


def load_test_examples(test_file: str) -> dict[str, list[dict]]:
    """Load and group test examples by task.

    Args:
        test_file: Path to the test JSONL file.

    Returns:
        Dict mapping task name to list of example dicts.

    Raises:
        ValueError: If an example has an unrecognized task label.
    """
    grouped = defaultdict(list)
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            task = example["task"]
            if task not in {t.value for t in TaskType}:
                raise ValueError(f"Unrecognized task label: {task}")
            grouped[task].append(example)
    return grouped


def evaluate_qa(model, tokenizer, examples: list[dict], device: str) -> dict:
    """Evaluate the QA task using EM and F1.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of QA example dicts.
        device: Torch device string.

    Returns:
        Dict with "em" and "f1" scores, averaged over all examples.
    """
    inputs = [ex["input_text"] for ex in examples]
    references = [ex["target"] for ex in examples]
    predictions = generate_predictions(model, tokenizer, inputs, device)

    em_scores, f1_scores = [], []
    for pred, ref in zip(predictions, references):
        em, f1 = compute_em_f1(pred, ref)
        em_scores.append(em)
        f1_scores.append(f1)

    return {"em": sum(em_scores) / len(em_scores), "f1": sum(f1_scores) / len(f1_scores)}


def evaluate_qg(model, tokenizer, examples: list[dict], device: str) -> dict:
    """Evaluate the QG task using BLEU-4 and ROUGE-L.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of QG example dicts.
        device: Torch device string.

    Returns:
        Dict with "bleu" and "rougeL" scores.
    """
    inputs = [ex["input_text"] for ex in examples]
    references = [ex["target"] for ex in examples]
    predictions = generate_predictions(model, tokenizer, inputs, device)

    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")

    bleu_result = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)

    return {"bleu": bleu_result["score"], "rougeL": rouge_result["rougeL"]}


def evaluate_distractor(model, tokenizer, examples: list[dict], device: str) -> dict:
    """Evaluate the Distractor task using distinct-n and validity rate.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of Distractor example dicts.
        device: Torch device string.

    Returns:
        Dict with "distinct_1", "distinct_2", and "validity_rate" (fraction
        of generated distractors that are not identical to the gold answer).
    """
    inputs = [ex["input_text"] for ex in examples]
    predictions = generate_predictions(model, tokenizer, inputs, device)

    all_distractors = []
    valid_count = 0
    total_count = 0

    for pred, ex in zip(predictions, examples):
        gold_answer = ex["input_text"].split("answer: ")[-1].strip().lower()
        distractors = [d.strip() for d in pred.split(DISTRACTOR_SEP.strip()) if d.strip()]

        for d in distractors:
            total_count += 1
            if d.lower() != gold_answer:
                valid_count += 1
        all_distractors.extend(distractors)

    validity_rate = valid_count / total_count if total_count > 0 else 0.0

    return {
        "distinct_1": distinct_n(all_distractors, n=1),
        "distinct_2": distinct_n(all_distractors, n=2),
        "validity_rate": validity_rate,
    }


def evaluate_all(checkpoint: str, test_file: str) -> dict:
    """Run evaluation for all three tasks and return a combined report.

    Args:
        checkpoint: Path to the fine-tuned model checkpoint.
        test_file: Path to the test JSONL file.

    Returns:
        Dict mapping task name to its metric dict.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.eval()

    grouped_examples = load_test_examples(test_file)

    report = {}
    if TaskType.QA.value in grouped_examples:
        report["answer-generation"] = evaluate_qa(model, tokenizer, grouped_examples[TaskType.QA.value], device)
    if TaskType.QG.value in grouped_examples:
        report["question-generation"] = evaluate_qg(model, tokenizer, grouped_examples[TaskType.QG.value], device)
    if TaskType.DISTRACTOR.value in grouped_examples:
        report["distractor"] = evaluate_distractor(
            model, tokenizer, grouped_examples[TaskType.DISTRACTOR.value], device
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the multi-task T5 model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="W&B run_name of the training run being evaluated (attaches eval metrics to it).",
    )
    args = parser.parse_args()

    results = evaluate_all(args.checkpoint, args.test_file)
    print(json.dumps(results, indent=2))

    # Resume the training run so eval metrics live alongside its loss curve
    # and artifact lineage, instead of a disconnected report.
    run = wandb.init(project=WANDB_PROJECT, id=args.run_name, resume="must")
    for task, metrics in results.items():
        for metric_name, value in metrics.items():
            run.summary[f"eval/{task}/{metric_name}"] = value
    run.finish()
    logger.info(f"Eval metrics attached to W&B run '{args.run_name}'")
