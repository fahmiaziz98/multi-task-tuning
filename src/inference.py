"""End-to-end inference pipeline: context -> qa_pair -> distractors.

Wraps the fine-tuned multi-task model in a single class. Two ways to build
a question:
    - generate_qa_pair(context): fully automatic — model picks its own
      salient answer AND generates the matching question, in one call.
    - generate_qa_pair(context, answer=...): answer-aware — question is
      generated to specifically target the given answer.

Usage:
    from inference import QuizGenerator

    generator = QuizGenerator("your-username/multitask-t5-quiz-generator")
    quiz = generator.generate_quiz(context="...")
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from schema import DISTRACTOR_SEP, MASK_TOKEN, SEP_TOKEN

DEFAULT_MAX_TARGET_LENGTH = 64
DEFAULT_NUM_BEAMS = 4


@dataclass
class Quiz:
    """A generated multiple-choice question.

    Attributes:
        question: The generated question.
        answer: The (model-chosen or user-specified) correct answer.
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

    def generate_qa_pair(self, context: str, answer: str | None = None) -> tuple[str, str]:
        """Generate an (answer, question) pair from context.

        Args:
            context: Source passage.
            answer: Optional target answer. If provided, the question is
                generated to match it (answer-aware mode). If None, the
                model is asked to pick its own salient answer and generate
                a matching question (fully automatic mode).

        Returns:
            Tuple of (answer, question). In automatic mode, `answer` is the
            model's own chosen answer, not necessarily the input `answer`.
        """
        answer_for_input = answer if answer is not None else MASK_TOKEN
        input_text = f"generate qa pair: context: {context} answer: {answer_for_input}"
        raw_output = self._generate(input_text)

        if SEP_TOKEN not in raw_output:
            # Malformed generation without a separator; treat the whole
            # output as the question and fall back to the given answer.
            return (answer or ""), raw_output.strip()

        generated_answer, _, generated_question = raw_output.partition(SEP_TOKEN)
        return generated_answer.strip(), generated_question.strip()

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer given a context and a user-supplied question.

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

        Args:
            context: Source passage to build a quiz from.
            answer: Optional pre-specified answer to build the question
                around (answer-aware mode). If None, the model picks its
                own answer (fully automatic mode).

        Returns:
            A Quiz object containing the question, answer, and distractors.
        """
        generated_answer, question = self.generate_qa_pair(context, answer)
        distractors = self.generate_distractors(context, question, generated_answer)

        return Quiz(question=question, answer=generated_answer, distractors=distractors)
