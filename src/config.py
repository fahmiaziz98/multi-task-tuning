"""Central configuration for training the multi-task T5 model.

Keeping hyperparameters and shared constants (like the W&B project name) in
one place avoids duplication and drift across build_dataset.py, train.py,
and evaluate.py.
"""

import uuid
from dataclasses import dataclass, field

# Single source of truth for the W&B project name. Previously this was
# duplicated (and inconsistently spelled) across three files.
WANDB_PROJECT = "multitask-t5-quiz-generator"


@dataclass
class TrainingConfig:
    """Hyperparameters, paths, and tracking/deployment settings.

    Attributes:
        model_name: HuggingFace model id or local path to start from.
        output_dir: Local directory where checkpoints are saved.
        max_input_length: Max token length for encoder input.
        max_target_length: Max token length for decoder target.
        per_device_train_batch_size: Batch size per device during training.
        per_device_eval_batch_size: Batch size per device during evaluation.
        gradient_accumulation_steps: Steps to accumulate before an optimizer
            step, used to simulate a larger effective batch size.
        learning_rate: Optimizer learning rate.
        num_train_epochs: Number of full passes over the training data.
        warmup_steps: Number of linear warmup steps for the LR scheduler.
        fp16: Whether to use mixed precision training.
        seed: Random seed for reproducibility.
        run_name: Unique identifier for this run, used as the W&B run id.
        dataset_artifact: W&B artifact reference for the training dataset,
            e.g. "qg-qa-distractor-dataset:latest".
        push_to_hub: Whether to push the final model to the HF Hub.
        hf_repo_id: Target HF Hub repo id, used only if push_to_hub is True.
    """

    model_name: str = "t5-small"
    output_dir: str = "./checkpoints/multitask-t5"

    max_input_length: int = 512
    max_target_length: int = 64

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    learning_rate: float = 3e-4
    num_train_epochs: int = 3
    warmup_steps: int = 500

    fp16: bool = True
    seed: int = 42

    run_name: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    dataset_artifact: str = "qg-qa-distractor-dataset:latest"

    push_to_hub: bool = True
    hf_repo_id: str = "fahmiaziz/multitask-t5-quiz-generator"
