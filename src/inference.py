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


class _SingleTaskModel:
    """Thin wrapper around one fine-tuned T5 checkpoint for text generation."""

    def __init__(self, checkpoint: str, device: str):
        """Load a tokenizer and model for one task.

        Args:
            checkpoint: Local path or HF Hub repo id.
            device: Torch device to run on.

        Raises:
            OSError: If the checkpoint cannot be loaded.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
        self.model.eval()

    def generate(self, input_text: str) -> str:
        """Run generation for a single formatted input string.

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
                **encoded, 
                max_length=DEFAULT_MAX_TARGET_LENGTH, 
                num_beams=DEFAULT_NUM_BEAMS,
                diversity_penalty=0.5,
#                do_sample=True, top_k=50, top_p=0.9
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


class QuizGenerator:
    """Generates full multiple-choice quiz items using two separate models."""

    def __init__(
        self,
        qa_pair_checkpoint: str,
        distractor_checkpoint: str,
        device: str | None = None,
    ):
        """Load both fine-tuned models.

        Args:
            qa_pair_checkpoint: Local path or HF Hub repo id of the
                qa_pair model.
            distractor_checkpoint: Local path or HF Hub repo id of the
                distractor model.
            device: Torch device to run both models on. Defaults to "cuda"
                if available, otherwise "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._qa_pair_model = _SingleTaskModel(qa_pair_checkpoint, self.device)
        self._distractor_model = _SingleTaskModel(distractor_checkpoint, self.device)

    def generate_qa_pair(self, context: str, answer: str | None = None) -> tuple[str, str]:
        """Generate an (answer, question) pair from context.

        Args:
            context: Source passage.
            answer: Optional target answer (answer-aware mode). If None,
                the model picks its own salient answer (automatic mode).

        Returns:
            Tuple of (answer, question).
        """
        answer_for_input = answer if answer is not None else MASK_TOKEN
        input_text = f"generate qa pair: answer: {answer_for_input} context: {context}"
        raw_output = self._qa_pair_model.generate(input_text)

        if SEP_TOKEN not in raw_output:
            return (answer or ""), raw_output.strip()

        generated_answer, _, generated_question = raw_output.partition(SEP_TOKEN)
        return generated_answer.strip(), generated_question.strip()

    def generate_distractors(self, context: str, question: str, answer: str) -> list[str]:
        """Generate distractor options given a question, answer, and context.

        Args:
            context: Source passage.
            question: The question being asked.
            answer: The correct answer.

        Returns:
            List of distractor strings (empty entries filtered out).
        """
        input_text = f"generate distractor: question: {question} answer: {answer} context: {context}"
        raw_output = self._distractor_model.generate(input_text)
        return [d.strip() for d in raw_output.split(DISTRACTOR_SEP.strip()) if d.strip()]

    def generate_quiz(self, context: str, answer: str | None = None) -> Quiz:
        """Generate a full multiple-choice quiz item from raw context.

        Args:
            context: Source passage to build a quiz from.
            answer: Optional pre-specified answer (answer-aware mode).

        Returns:
            A Quiz object containing the question, answer, and distractors.
        """
        generated_answer, question = self.generate_qa_pair(context, answer)
        distractors = self.generate_distractors(context, question, generated_answer)

        return Quiz(question=question, answer=generated_answer, distractors=distractors)
