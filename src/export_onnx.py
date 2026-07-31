"""Export a fine-tuned T5 checkpoint to ONNX for fast CPU inference.

Usage:
    python src/export_onnx.py --checkpoint your-username/multitask-t5-quiz-generator \
        --output_dir ./exported/onnx
"""

import argparse

from loguru import logger
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer


def export_to_onnx(checkpoint: str, output_dir: str) -> None:
    """Convert and save a T5 checkpoint in ONNX format.

    Args:
        checkpoint: Local path or HF Hub repo id of the fine-tuned model.
        output_dir: Directory to save the exported ONNX model and tokenizer.

    Raises:
        OSError: If the checkpoint cannot be loaded or the export fails.
    """
    model = ORTModelForSeq2SeqLM.from_pretrained(checkpoint, export=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"ONNX model exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a T5 checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output_dir)
