import uuid
from dataclasses import dataclass, field


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
            e.g. "qa-pair-distractor-dataset:latest".
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

    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    max_grad_norm: float = 1.0
    optim: str = "adafactor"
    warmup_steps: int = 500

    fp16: bool = False
    seed: int = 42

    run_name: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    dataset_artifact: str = "qa-pair-distractor-dataset:latest"

    push_to_hub: bool = True
    hf_repo_id: str = "fahmiaziz/multitask-t5-quiz-generator"
