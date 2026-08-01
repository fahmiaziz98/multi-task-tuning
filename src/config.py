"""Central configuration for training the qa_pair and distractor models.

Each task now trains its own model, so TrainingConfig is parameterized by
task rather than duplicated into two files. get_config(task) returns the
right defaults, including the artifact/repo names that differ per task.

Stability note: T5 + fp16 is a known unstable combination — gradients can
overflow mid-training (visible as grad_norm: nan). Defaults here use
Adafactor (the optimizer used in the original T5 paper, more numerically
stable for this architecture) and fp16 disabled. Re-enable fp16 only after
confirming training is stable without it.
"""

import uuid
from dataclasses import dataclass, field

WANDB_PROJECT = "multitask-t5-quiz-generator"

_ARTIFACT_BY_TASK = {
    "qa_pair": "qa-pair-dataset:latest",
    "distractor": "distractor-dataset:latest",
}
_HF_REPO_BY_TASK = {
    "qa_pair": "fahmiaziz/qa-pair-generator",
    "distractor": "fahmiaziz/distractor-generator",
}


@dataclass
class TrainingConfig:
    """Hyperparameters, paths, and tracking/deployment settings for one task.

    Attributes:
        task: Which task this config trains for ("qa_pair" or "distractor").
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
        optim: Optimizer name passed to Seq2SeqTrainingArguments.
        fp16: Whether to use mixed precision training.
        max_grad_norm: Gradient clipping threshold.
        seed: Random seed for reproducibility.
        run_name: Unique identifier for this run, used as the W&B run id.
        dataset_artifact: W&B artifact reference for this task's dataset.
        push_to_hub: Whether to push the final model to the HF Hub.
        hf_repo_id: Target HF Hub repo id, used only if push_to_hub is True.
    """

    task: str
    model_name: str = "t5-small"
    output_dir: str = ""

    max_input_length: int = 512
    max_target_length: int = 64

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    learning_rate: float = 1e-4
    num_train_epochs: int = 10
    warmup_steps: int = 500
    optim: str = "adafactor"

    fp16: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42

    run_name: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    dataset_artifact: str = ""

    push_to_hub: bool = True
    hf_repo_id: str = ""

    def __post_init__(self):
        if self.task not in _ARTIFACT_BY_TASK:
            raise ValueError(f"Unknown task '{self.task}'. Expected one of {list(_ARTIFACT_BY_TASK)}.")
        if not self.output_dir:
            self.output_dir = f"./checkpoints/{self.task}"
        if not self.dataset_artifact:
            self.dataset_artifact = _ARTIFACT_BY_TASK[self.task]
        if not self.hf_repo_id:
            self.hf_repo_id = _HF_REPO_BY_TASK[self.task]
