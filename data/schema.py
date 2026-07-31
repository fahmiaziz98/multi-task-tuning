"""Data schema definitions for the multi-task QG/QA/Distractor dataset.

Defines the canonical example format shared across all three tasks
(question answering, question generation, distractor generation) so that
dataset building, training, and evaluation code rely on one consistent
structure.
"""

from dataclasses import dataclass
from enum import Enum


DISTRACTOR_SEP = " <sep> "

class TaskType(str, Enum):
    """
    Enumerates the supported fine-tuning tasks.
    """
    QA = "answer-generation"
    QG = "question-genaration"
    DISTRACTOR = "distractor"


@dataclass
class TrainingTask:
    """
    A single unified training example

    Attributes:
        task: Which task this example belong to.
        input_text: The full text2text input.
        target: The expected output.
    """
    task: TaskType
    input_text: str
    target: str

    def to_dict(self) -> dict:
        """Convert the example into a JSON serializable dictionary"""
        return {
            "task": self.task.value,
            "input_text": self.input_text,
            "target": self.target
        }
