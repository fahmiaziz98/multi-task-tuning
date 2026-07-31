from dataclasses import dataclass
from enum import Enum

# Single special token used both to join multiple distractors and to
# separate the answer from the question in qa_pair targets.
SEP_TOKEN = "<sep>"
DISTRACTOR_SEP = f" {SEP_TOKEN} "

# Token substituted for the answer in qa_pair inputs when training the
# "fully automatic" mode (model must pick its own answer from context).
MASK_TOKEN = "[MASK]"


class TaskType(str, Enum):
    """Enumerates the supported fine-tuning tasks.

    Attributes:
        QA_PAIR: answer (or MASK) + context -> "answer <sep> question".
            Trained with partial answer-masking so the model learns both
            answer-aware and fully-automatic question generation.
        DISTRACTOR: question + answer + context -> distractor options.
    """

    QA_PAIR = "qa_pair"
    DISTRACTOR = "distractor"


@dataclass
class TrainingTask:
    """A single unified training example.

    Attributes:
        task: Which task this example belongs to.
        input_text: The full text-to-text input, including the task prefix.
            Short fields (question/answer/mask) are placed before the long
            context field, since tokenizer truncation cuts from the end —
            this keeps the critical fields safe from being truncated away.
        target: The expected output text.
    """

    task: TaskType
    input_text: str
    target: str

    def to_dict(self) -> dict:
        """Convert the example into a JSON-serializable dictionary."""
        return {
            "task": self.task.value,
            "input_text": self.input_text,
            "target": self.target,
        }
