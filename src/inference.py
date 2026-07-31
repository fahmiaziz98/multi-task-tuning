"""End-to-end inference pipeline: context -> question -> answer -> distractors.

Wraps the fine-tuned multi-task model in a single class so that generating a
full multiple-choice question from raw text only takes one method call.

Usage:
    from inference import QuizGenerator

    generator = QuizGenerator("your-username/multitask-t5-quiz-generator")
    quiz = generator.generate_quiz(context="...")
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from schema import DISTRACTOR_SEP

DEFAULT_MAX_TARGET_LENGTH = 64
DEFAULT_NUM_BEAMS = 4


@dataclass
class Quiz:
    """A generated multiple-choice question.

    Attributes:
        question: The generated question.
        answer: The generated (correct) answer.
        distractors: List of generated incorrect options.
    """

    question: str
    answer: str
    distractors: list[str]


class QuizGenerator:
    """Generates full multiple-choice quiz items from raw text context."""

    def __init__(self, checkpoint: str, device: str | None = None):
        """Load the fine-tuned model and tokenizer.

        Args:
            checkpoint: Local path or HF Hub repo id of the fine-tuned model.
            device: Torch device to run on. Defaults to "cuda" if available,
                otherwise "cpu".

        Raises:
            OSError: If the checkpoint cannot be loaded.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    def _generate(self, input_text: str) -> str:
        """Run a single generation call for one task-prefixed input string.

        Args:
            input_text: Fully formatted input, including the task prefix.

        Returns:
            The decoded generated text.
        """
        encoded = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded, max_length=DEFAULT_MAX_TARGET_LENGTH, num_beams=DEFAULT_NUM_BEAMS
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_question(self, context: str, answer: str) -> str:
        """Generate a question given a context and a target answer.

        Args:
            context: Source passage.
            answer: The answer the question should target.

        Returns:
            Generated question string.
        """
        input_text = f"generate question: context: {context} answer: {answer}"
        return self._generate(input_text)

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer given a context and a question.

        Args:
            context: Source passage.
            question: Question to answer.

        Returns:
            Generated answer string.
        """
        input_text = f"answer the question: question: {question} context: {context}"
        return self._generate(input_text)

    def generate_distractors(self, context: str, question: str, answer: str) -> list[str]:
        """Generate distractor options given a context, question, and answer.

        Args:
            context: Source passage.
            question: The question being asked.
            answer: The correct answer.

        Returns:
            List of distractor strings (empty entries filtered out).
        """
        input_text = (
            f"generate distractors: context: {context} question: {question} answer: {answer}"
        )
        raw_output = self._generate(input_text)
        return [d.strip() for d in raw_output.split(DISTRACTOR_SEP.strip()) if d.strip()]

    def generate_quiz(self, context: str, answer: str | None = None) -> Quiz:
        """Generate a full multiple-choice quiz item from raw context.

        If `answer` is not provided, an answer is first extracted by asking
        the model a generic question about the context.

        Args:
            context: Source passage to build a quiz from.
            answer: Optional pre-specified answer to build the question
                around. If None, a placeholder question is used to have the
                model surface a salient answer first.

        Returns:
            A Quiz object containing the question, answer, and distractors.
        """
        if answer is None:
            answer = self.generate_answer(context, question="What is discussed in this text?")

        question = self.generate_question(context, answer)
        distractors = self.generate_distractors(context, question, answer)

        return Quiz(question=question, answer=answer, distractors=distractors)
