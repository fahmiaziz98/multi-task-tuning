import argparse
import json
import re
import sys
from pathlib import Path

import evaluate as hf_evaluate
import torch
import wandb
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from schema import DISTRACTOR_SEP, MASK_TOKEN, SEP_TOKEN  # noqa: E402
from config import WANDB_PROJECT


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
        prediction: Model-generated text.
        reference: Gold text.

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


def load_examples(test_file: str) -> list[dict]:
    """Load a task's test examples from JSONL.

    Args:
        test_file: Path to the test JSONL file.

    Returns:
        List of example dicts.
    """
    with open(test_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_qa_pair_target(text: str) -> tuple[str, str]:
    """Split a "answer <sep> question" string into its two parts.

    Args:
        text: Raw text expected to contain the separator token once.

    Returns:
        Tuple of (answer_part, question_part). If the separator is missing,
        the whole text is returned as the answer part.
    """
    if SEP_TOKEN not in text:
        return text.strip(), ""
    answer_part, _, question_part = text.partition(SEP_TOKEN)
    return answer_part.strip(), question_part.strip()


def evaluate_qa_pair(model, tokenizer, examples: list[dict], device: str) -> dict:
    """Evaluate the qa_pair model, overall and broken down by mode.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of qa_pair example dicts.
        device: Torch device string.

    Returns:
        Dict with "answer_em", "answer_f1", "question_bleu",
        "question_rougeL" (overall), plus the same metrics prefixed with
        "masked_" and "answer_aware_" for each mode separately.
    """
    inputs = [ex["input_text"] for ex in examples]
    is_masked = [MASK_TOKEN in ex["input_text"] for ex in examples]
    predictions = generate_predictions(model, tokenizer, inputs, device)

    pred_answers, pred_questions = [], []
    ref_answers, ref_questions = [], []
    for pred, ex in zip(predictions, examples):
        pred_answer, pred_question = split_qa_pair_target(pred)
        ref_answer, ref_question = split_qa_pair_target(ex["target"])
        pred_answers.append(pred_answer)
        pred_questions.append(pred_question)
        ref_answers.append(ref_answer)
        ref_questions.append(ref_question)

    def score_subset(indices: list[int]) -> dict:
        """Compute answer EM/F1 and question BLEU/ROUGE-L over a subset."""
        if not indices:
            return {}

        em_scores, f1_scores = [], []
        for i in indices:
            em, f1 = compute_em_f1(pred_answers[i], ref_answers[i])
            em_scores.append(em)
            f1_scores.append(f1)

        bleu_metric = hf_evaluate.load("sacrebleu")
        rouge_metric = hf_evaluate.load("rouge")
        subset_preds = [pred_questions[i] for i in indices]
        subset_refs = [ref_questions[i] for i in indices]
        bleu_result = bleu_metric.compute(predictions=subset_preds, references=[[r] for r in subset_refs])
        rouge_result = rouge_metric.compute(predictions=subset_preds, references=subset_refs)

        return {
            "answer_em": sum(em_scores) / len(em_scores),
            "answer_f1": sum(f1_scores) / len(f1_scores),
            "question_bleu": bleu_result["score"],
            "question_rougeL": rouge_result["rougeL"],
        }

    all_indices = list(range(len(examples)))
    masked_indices = [i for i, m in enumerate(is_masked) if m]
    answer_aware_indices = [i for i, m in enumerate(is_masked) if not m]

    result = score_subset(all_indices)
    for key, value in score_subset(masked_indices).items():
        result[f"masked_{key}"] = value
    for key, value in score_subset(answer_aware_indices).items():
        result[f"answer_aware_{key}"] = value

    return result


def evaluate_distractor(model, tokenizer, examples: list[dict], device: str) -> dict:
    """Evaluate the distractor model using distinct-n and validity rate.

    Args:
        model: Fine-tuned model.
        tokenizer: Matching tokenizer.
        examples: List of Distractor example dicts.
        device: Torch device string.

    Returns:
        Dict with "distinct_1", "distinct_2", and "validity_rate".
    """
    inputs = [ex["input_text"] for ex in examples]
    predictions = generate_predictions(model, tokenizer, inputs, device)

    all_distractors = []
    valid_count = 0
    total_count = 0

    for pred, ex in zip(predictions, examples):
        # Gold answer sits between "answer: " and " context:" since context
        # is placed last in the prompt.
        gold_answer = ex["input_text"].split("answer: ")[-1].split(" context:")[0].strip().lower()
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


EVALUATORS = {
    "qa_pair": evaluate_qa_pair,
    "distractor": evaluate_distractor,
}


def evaluate_task(task: str, checkpoint: str, test_file: str) -> dict:
    """Run evaluation for a single task's model.

    Args:
        task: Either "qa_pair" or "distractor".
        checkpoint: Path to the fine-tuned model checkpoint.
        test_file: Path to that task's test JSONL file.

    Returns:
        Dict of metric name to value.

    Raises:
        ValueError: If `task` is not a recognized task name.
    """
    if task not in EVALUATORS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(EVALUATORS)}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.eval()

    examples = load_examples(test_file)
    return EVALUATORS[task](model, tokenizer, examples, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single-task T5 model.")
    parser.add_argument("--task", type=str, required=True, choices=["qa_pair", "distractor"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True, help="W&B run ID (e.g. s2d46sr0)")
    args = parser.parse_args()

    results = evaluate_task(args.task, args.checkpoint, args.test_file)
    print(json.dumps(results, indent=2))

    run = wandb.init(project=WANDB_PROJECT, job_type="evaluation", name=f"eval-{args.run_id}", tags=[args.task])
    wandb.log({f"{args.task}/{k}": v for k, v in results.items()})
    run.finish()
    logger.info(f"Eval metrics logged as W&B run 'eval-{args.run_id}'")
