import argparse
import shutil
from pathlib import Path

import wandb
from loguru import logger

WANDB_PROJECT = "multitask-t5-quiz-generator"


def download_dataset_artifact(artifact_ref: str, output_dir: str) -> Path:
    """Download a W&B dataset artifact and copy its files into output_dir.

    Args:
        artifact_ref: Artifact reference, e.g. "qa-pair-dataset:latest" or
            "distractor-dataset:v2".
        output_dir: Local folder to place the downloaded files into
            (e.g. "./data/processed/qa_pair").

    Returns:
        Path to output_dir, containing train.jsonl, val.jsonl, test.jsonl.
    """
    run = wandb.init(project=WANDB_PROJECT, job_type="download-dataset")
    artifact = run.use_artifact(artifact_ref)
    downloaded_dir = Path(artifact.download())
    run.finish()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in downloaded_dir.glob("*.jsonl"):
        shutil.copy(file, output_path / file.name)

    logger.info(f"Downloaded '{artifact_ref}' -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset artifact from W&B.")
    parser.add_argument("--artifact", type=str, required=True, help="e.g. qa-pair-dataset:latest")
    parser.add_argument("--output_dir", type=str, required=True, help="e.g. ./data/processed/qa_pair")
    args = parser.parse_args()

    download_dataset_artifact(args.artifact, args.output_dir)
