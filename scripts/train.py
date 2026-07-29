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
    from transformers import TrainerCallback, TrainingArguments
    from trl import SFTTrainer

    # NOTE: deliberately NOT setting save_steps/save_strategy here. HF
    # Trainer's built-in periodic checkpointing tries to torch.save() the
    # full TrainingArguments/SFTConfig object, which fails under Unsloth's
    # dynamic class patching with:
    #   _pickle.PicklingError: Can't pickle <class '...SFTConfig'>:
    #   it's not the same object as trl.trainer.sft_config.SFTConfig
    # (confirmed on a real run - see technical_lessons_learned.md).
    # Saving only the adapter weights via trainer.save_model() below
    # sidesteps this entirely - it never touches the args object.
    training_args = TrainingArguments(
        output_dir=config["checkpoint_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        num_train_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy="no",  # disable built-in checkpointing (see note above)
        report_to="none",
    )

    class PeriodicAdapterSaveCallback(TrainerCallback):
        """Saves adapter weights only, via model.save_pretrained() - NOT
        trainer.save_model(), which internally calls Trainer._save() and
        ALWAYS tries to torch.save() the full TrainingArguments/SFTConfig
        object regardless of how it's invoked (confirmed on a real run:
        even called manually here, outside the save_steps mechanism, it
        hit the same pickling error - see technical_lessons_learned.md).
        model.save_pretrained() goes through PEFT's own save path instead,
        which only writes adapter weights + config, never touching the
        Trainer's args object at all."""

        def __init__(self, model_ref: Any, tokenizer_ref: Any, checkpoint_dir: str, every_n_steps: int) -> None:
            self.model_ref = model_ref
            self.tokenizer_ref = tokenizer_ref
            self.checkpoint_dir = checkpoint_dir
            self.every_n_steps = every_n_steps

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
                path = f"{self.checkpoint_dir}/checkpoint-{state.global_step}"
                self.model_ref.save_pretrained(path)
                self.tokenizer_ref.save_pretrained(path)
                print(f"Saved adapter checkpoint: {path}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field="text",
    )
    trainer.add_callback(
        PeriodicAdapterSaveCallback(
            model_ref=model,
            tokenizer_ref=tokenizer,
            checkpoint_dir=config["checkpoint_dir"],
            every_n_steps=config["training"]["save_every_n_steps"],
        )
    )

    print("Starting training...")
    trainer.train()

    final_path = f"{config['checkpoint_dir']}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final adapter saved to {final_path}")

    with open("train_run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"final_checkpoint": final_path, "config": config}, f, indent=2)


if __name__ == "__main__":
    main()
