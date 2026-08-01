from dataclasses import dataclass
from enum import Enum


SEP_TOKEN = "<sep>"
DISTRACTOR_SEP = f" {SEP_TOKEN} "
MASK_TOKEN = "[MASK]"


class TaskType(str, Enum):
    """Enumerates the two independently-trained tasks.

    Attributes:
        QA_PAIR: answer (or MASK) + context -> "answer <sep> question".
        DISTRACTOR: question + answer + context -> distractor options.
    """

    QA_PAIR = "qa_pair"
    DISTRACTOR = "distractor"


@dataclass
class TrainingTask:
    """A single training example for one task.

    Attributes:
        task: Which task this example belongs to.
        input_text: The full text-to-text input, including the task prefix.
            Short fields (question/answer/mask) are placed before the long
            context field, since tokenizer truncation cuts from the end.
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
