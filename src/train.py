import argparse
import subprocess
from dataclasses import dataclass

import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from config import WANDB_PROJECT, TrainingConfig

SPECIAL_TOKENS = ["<sep>", "[MASK]"]


@dataclass
class Preprocessor:
    """Tokenizes raw text examples into model-ready input/label ids.

    Attributes:
        tokenizer: The tokenizer used to encode text.
        max_input_length: Max token length for the encoder input.
        max_target_length: Max token length for the decoder target.
    """

    tokenizer: AutoTokenizer
    max_input_length: int
    max_target_length: int

    def __call__(self, batch: dict) -> dict:
        """Tokenize a batch of raw examples.

        Args:
            batch: Dict with "input_text" and "target" list fields.

        Returns:
            Dict with "input_ids", "attention_mask", and "labels".
        """
        model_inputs = self.tokenizer(
            batch["input_text"], max_length=self.max_input_length, truncation=True
        )
        labels = self.tokenizer(
            text_target=batch["target"], max_length=self.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None if not in a git repo.

    Returns:
        The commit hash string, or None if it cannot be determined.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_tokenizer_and_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Load the tokenizer and model, adding custom special tokens.

    Args:
        model_name: HuggingFace model id or local path.

    Returns:
        Tuple of (tokenizer, model), with model embeddings resized to
        account for any newly added special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_added = tokenizer.add_tokens(SPECIAL_TOKENS)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def build_dataset(data_dir: str, tokenizer: AutoTokenizer, config: TrainingConfig):
    """Load JSONL splits from a directory and tokenize them for training.

    Args:
        data_dir: Directory containing train.jsonl and val.jsonl (typically
            the local path returned by downloading a W&B dataset artifact).
        tokenizer: Tokenizer used for preprocessing.
        config: Training configuration with max length settings.

    Returns:
        A HuggingFace DatasetDict with "train" and "validation" splits,
        already tokenized.
    """
    data_files = {
        "train": f"{data_dir}/train.jsonl",
        "validation": f"{data_dir}/val.jsonl",
    }
    raw_dataset = load_dataset("json", data_files=data_files)

    preprocessor = Preprocessor(
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )
    return raw_dataset.map(
        preprocessor, batched=True, remove_columns=raw_dataset["train"].column_names
    )


def train(config: TrainingConfig) -> None:
    """Run the full fine-tuning loop for one task, with W&B tracking and lineage.

    Args:
        config: Training configuration for the task being trained.
    """
    run = wandb.init(
        project=WANDB_PROJECT,
        name=config.run_name,
        config=config.__dict__,
        job_type="train",
        tags=[config.task],
    )

    dataset_artifact = run.use_artifact(config.dataset_artifact)
    dataset_dir = dataset_artifact.download()

    tokenizer, model = load_tokenizer_and_model(config.model_name)
    dataset = build_dataset(dataset_dir, tokenizer, config)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        optim=config.optim,
        fp16=config.fp16,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        logging_steps=100,
        report_to=["wandb"],
        run_name=config.run_name,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    model_artifact = wandb.Artifact(
        name=f"{config.task}-t5-model",
        type="model",
        metadata={
            "task": config.task,
            "base_model": config.model_name,
            "run_name": config.run_name,
            "git_commit": get_git_commit(),
        },
    )
    model_artifact.add_dir(config.output_dir)
    run.log_artifact(model_artifact)

    if config.push_to_hub:
        api = HfApi()
        api.create_repo(repo_id=config.hf_repo_id, exist_ok=True)
        model.push_to_hub(config.hf_repo_id)
        tokenizer.push_to_hub(config.hf_repo_id)
        logger.info(f"Model pushed to https://huggingface.co/{config.hf_repo_id}")

    run.finish()
    logger.info(f"Training complete for task '{config.task}'. Model saved to {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a T5 model on one task.")
    parser.add_argument(
        "--task", type=str, required=True, choices=["qa_pair", "distractor"], help="Which task to train."
    )
    args = parser.parse_args()

    train(TrainingConfig(task=args.task))
