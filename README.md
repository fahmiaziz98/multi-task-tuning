# Quiz Generator: qa_pair + Distractor

Two independently fine-tuned T5-small models that together turn a raw text
passage into a complete multiple-choice quiz item: a question, its answer,
and plausible wrong options.

- **qa_pair model** — generates an `answer <sep> question` pair from a
  context, either answer-aware (you supply the answer) or fully automatic
  (the model picks its own salient answer via masking).
- **distractor model** — generates 3 incorrect-but-plausible options given
  a context, question, and correct answer.

Both are `t5-small`, trained and versioned independently, and chained
together at inference time by `QuizGenerator`.

---

## Quickstart (Colab)

```python
!git clone https://github.com/<username>/qa-pair-distractor-t5.git
%cd qa-pair-distractor-t5
!pip install --system -q -r requirements.txt

import sys
sys.path.append("./data")
sys.path.append("./src")

import wandb
wandb.login()

from huggingface_hub import login
login()

# 1. Build both datasets (qa_pair from SQuAD, distractor from RACE + synthetic short-answer examples)
!python data/build_dataset.py --output_dir ./data/processed

# 2. Train each model independently
!python src/train.py --task qa_pair
!python src/train.py --task distractor --model_name t5-small --epochs 8

# 3. Evaluate each model against its own test set
!python src/evaluate.py --task qa_pair \
    --checkpoint ./checkpoints/qa_pair \
    --test_file ./data/processed/qa_pair/test.jsonl \
    --run_id <run_id_from_wandb>

!python src/evaluate.py --task distractor \
    --checkpoint ./checkpoints/distractor \
    --test_file ./data/processed/distractor/test.jsonl \
    --run_id <run_id_from_wandb>
```

If you already have datasets logged as W&B Artifacts and just want to pull
them locally (e.g. a fresh Colab session, or local debugging), skip step 1
and use:

```python
!python data/load_artifact.py --artifact qa-pair-dataset:latest --output_dir ./data/processed/qa_pair
!python data/load_artifact.py --artifact distractor-dataset:latest --output_dir ./data/processed/distractor
```

---

## Inference

```python
from inference import QuizGenerator

generator = QuizGenerator(
    qa_pair_checkpoint="your-username/t5-qa-pair-generator",
    distractor_checkpoint="your-username/t5-distractor-generator",
)

quiz = generator.generate_quiz(
    context="The mitochondria is the organelle responsible for producing "
            "ATP, the main energy currency of the cell."
)

print(quiz.question)      # "What is the main energy currency of the cell?"
print(quiz.answer)        # "ATP"
print(quiz.distractors)   # e.g. ['adolescence', 'symbiosis', 'Aristotle']
```

`generate_quiz()` also accepts an explicit `answer=` argument to force
answer-aware mode instead of letting the model pick its own answer.

```python
!python test_quiz_generator.py \
    --qa_pair_checkpoint ./checkpoints/qa_pair \
    --distractor_checkpoint ./checkpoints/distractor

# or use HF hub checkpoints
!python test_quiz_generator.py \
    --qa_pair_checkpoint fahmiaziz/t5-qa-pair-generator \
    --distractor_checkpoint fahmiaziz/t5-distractor-generator
```

---

## Repository structure

```
multi-task-tuning/
├── data/
│   ├── build_dataset.py       # builds both datasets, logs 2 W&B artifacts
│   ├── load_artifact.py        # pulls a dataset artifact into a local folder
│   └── schema.py                # shared TrainingTask/TaskType/token definitions
├── src/
│   ├── config.py                 # TrainingConfig, parameterized by task
│   ├── train.py                   # Seq2SeqTrainer loop, run with --task
│   ├── evaluate.py                 # per-task metrics, logged to W&B
│   ├── export_onnx.py             # exports model to ONNX
│   └── inference.py                 # QuizGenerator: loads both models, chains them
├── test_quiz_generator.py           # manual sanity-check script, 5 varied contexts
├── requirements.txt
├── SYSTEM_DESIGN.md
└── README.md
```

---

## Key design decisions

- **Two separate models, not one multi-task model.** An earlier 3-task
  single-model design (qa_pair + qa + distractor) was replaced after
  diagnosing degenerate distractor generation
- **Prompt field ordering matters.** Short fields (question/answer/mask
  marker) are always placed before the long context field in the input
  text, because tokenizer truncation cuts from the end — this protects the
  critical fields from being silently dropped.
- **Diverse beam search + repetition control at decoding time**
  (`num_beams=8, num_beam_groups=4, diversity_penalty=0.5,
  repetition_penalty=1.3, no_repeat_ngram_size=2`) was necessary to avoid
  degenerate repetition loops in distractor generation (e.g. "the sacrament
  of the sacrament of...").
- **Post-generation filtering** removes any distractor that duplicates the
  correct answer (case-insensitive) — the model occasionally generates the
  answer itself as one of its own distractors, which decoding parameters
  alone don't fully prevent.
- **Distractor training needed more epochs than qa_pair** (10 vs 3) — its
  loss was still trending down at epoch 3, and its task is inherently
  harder (constructing plausible wrong answers vs. extracting/generating a
  right one).

## Known limitations

- **qa_pair's fully-automatic (masked) mode is meaningfully weaker** than
  its answer-aware mode (`masked_answer_em` ≈ 0.09 vs `answer_aware_answer_em`
  ≈ 0.99 in the latest eval). The model is much better at generating a
  question for a *given* answer than at picking a good answer on its own.
- **Distractor diversity (distinct-1/2) is still moderate**, partly because
  many gold answers in the augmented dataset are short, common tokens
  (numbers, common short entities) with a naturally limited space of
  plausible wrong options.
- **English only.** No multilingual data or base model in this iteration.
