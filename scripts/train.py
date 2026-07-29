"""Milestone 2: QLoRA fine-tuning of Qwen2.5-Coder-7B on Spider (train split).

GPU-only script - every function here has only ever run in Colab, per
RUNBOOK.md. Written and syntax-checked locally, per the project's
division of labor (see technical_lessons_learned.md). Treat as
unverified until a real Colab run confirms it.

Saves checkpoints to Google Drive periodically (not just at the end) -
training runs 20-60+ minutes, and a lost session mid-training should not
mean starting from zero (see technical_lessons_learned.md, section 12).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.dataset import build_training_examples, load_spider
from src.model import apply_lora_config, load_base_model
from src.utils import set_seed, setup_logging


def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the training config (see configs/train_config.yaml)."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_checkpoint_dir_writable(checkpoint_dir: str) -> None:
    """Fail fast if Google Drive isn't actually mounted/writable.

    A silent failure here (checkpoint_dir not writable) would only surface
    much later, after wasting GPU time on training that can't be saved -
    see technical_lessons_learned.md section 12. Check this BEFORE
    starting the expensive part.
    """
    path = Path(checkpoint_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as e:
        raise RuntimeError(
            f"checkpoint_dir '{checkpoint_dir}' is not writable - "
            f"is Google Drive actually mounted? ({e})"
        ) from e


def main(config_path: str = "configs/train_config.yaml") -> None:
    setup_logging()
    set_seed(42)

    config = load_config(config_path)
    print(f"Loaded config: {config}")

    verify_checkpoint_dir_writable(config["checkpoint_dir"])
    print(f"Checkpoint dir confirmed writable: {config['checkpoint_dir']}")

    print("Loading Spider train split...")
    raw_examples = load_spider(config["data"]["spider_split"], data_dir=config["data"]["data_dir"])
    print(f"Loaded {len(raw_examples)} training examples")

    print("Loading base model for training...")
    model, tokenizer = load_base_model(model_name=config["model_name"])

    print("Applying LoRA adapter...")
    model = apply_lora_config(model, config["lora"])

    print("Building tokenized training dataset...")
    train_dataset = build_training_examples(raw_examples, tokenizer)

    # Import here (not at module level) - trl/transformers are GPU-only
    # deps, keeping them out of the module-level import list means this
    # file can still be *parsed* (not run) without them installed locally.
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=config["checkpoint_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        num_train_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_every_n_steps"],
        save_total_limit=3,  # keep last 3 checkpoints, don't fill up Drive
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field="text",
    )

    print("Starting training...")
    trainer.train()

    final_path = f"{config['checkpoint_dir']}/final"
    trainer.save_model(final_path)
    print(f"\nTraining complete. Final adapter saved to {final_path}")

    with open("train_run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"final_checkpoint": final_path, "config": config}, f, indent=2)


if __name__ == "__main__":
    main()
