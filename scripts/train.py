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
from src.model import apply_lora_config, load_base_model, load_lora_checkpoint_for_training
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

    resume_path = config.get("resume_from_checkpoint")
    if resume_path:
        print(f"Resuming from existing checkpoint: {resume_path}")
        model, tokenizer = load_lora_checkpoint_for_training(resume_path, model_name=config["model_name"])
        print("Loaded - adapter weights already trained, continuing (NOT applying a fresh LoRA config)")
    else:
        print("Loading base model for training (from scratch)...")
        model, tokenizer = load_base_model(model_name=config["model_name"])
        print("Applying LoRA adapter...")
        model = apply_lora_config(model, config["lora"])

    print("Building tokenized training dataset...")
    train_dataset = build_training_examples(raw_examples, tokenizer)

    # Import here (not at module level) - trl/transformers are GPU-only
    # deps, keeping them out of the module-level import list means this
    # file can still be *parsed* (not run) without them installed locally.
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    # NOTE: deliberately NOT setting save_steps/save_strategy here. HF
    # Trainer's built-in periodic checkpointing tries to torch.save() the
    # full TrainingArguments/SFTConfig object, which fails under Unsloth's
    # dynamic class patching with:
    #   _pickle.PicklingError: Can't pickle <class '...SFTConfig'>:
    #   it's not the same object as trl.trainer.sft_config.SFTConfig
    # (confirmed on a real run - see technical_lessons_learned.md).
    # Saving adapter weights directly via model.save_pretrained() below
    # (not trainer.save_model()) sidesteps this entirely.
    #
    # Switched TrainingArguments -> SFTConfig (a TRL subclass) to get
    # `completion_only_loss` - added after a real run showed accuracy
    # getting WORSE with more training steps despite falling loss (69.5%
    # baseline -> 67.0% -> 63.0%). Literature review found this is a
    # documented SFTTrainer gotcha: without completion-only masking, loss
    # is computed over the ENTIRE prompt+completion text, so the model
    # partly "learns" to reconstruct the (already-given) schema instead
    # of concentrating on SQL generation - see technical_lessons_learned.md.
    # Requires train_dataset to have separate prompt/completion columns
    # (see dataset.py's build_training_examples), not a single "text" field.
    #
    # lr_scheduler_type/warmup_ratio added for the same reason: the
    # comparable published result we're benchmarking against (77.2% EX,
    # QLoRA, Llama 3 8B, full Spider train, 2 epochs) explicitly used a
    # cosine scheduler with 0.03 warmup ratio - our earlier runs used a
    # constant LR with no warmup at all.
    training_args = SFTConfig(
        output_dir=config["checkpoint_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        num_train_epochs=config["training"]["num_epochs"],
        max_steps=config["training"].get("max_steps", -1),  # -1 = no limit, run full num_epochs
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        warmup_ratio=config["training"].get("warmup_ratio", 0.03),
        logging_steps=config["training"]["logging_steps"],
        save_strategy="no",  # disable built-in checkpointing (see note above)
        completion_only_loss=True,  # see note above - the main fix
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
        Trainer's args object at all.

        Two fixes added after a real Colab session disconnect
        (idle/max-duration limit) forced a second resume - see
        technical_lessons_learned.md:

        1. `step_offset`: a fresh Trainer's `state.global_step` always
           restarts at 0, even when resuming from an existing adapter.
           Without an offset, a second resume's checkpoint-100,
           checkpoint-200, etc. would collide with (and silently
           overwrite) identically-named folders from the FIRST resume
           run, already sitting in the same checkpoint_dir. Adding the
           already-completed step count keeps checkpoint numbers
           globally meaningful and collision-free across any number of
           resumes.

        2. `keep_last_n`: without cleanup, checkpoints accumulate
           forever - Google Drive free space dropped from 5GB to 2.6GB
           over one run. Deleting old checkpoints as new ones are saved
           (keeping only the most recent `keep_last_n`) keeps disk usage
           roughly constant regardless of how long training runs.
        """

        def __init__(
            self,
            model_ref: Any,
            tokenizer_ref: Any,
            checkpoint_dir: str,
            every_n_steps: int,
            step_offset: int = 0,
            keep_last_n: int = 3,
        ) -> None:
            self.model_ref = model_ref
            self.tokenizer_ref = tokenizer_ref
            self.checkpoint_dir = checkpoint_dir
            self.every_n_steps = every_n_steps
            self.step_offset = step_offset
            self.keep_last_n = keep_last_n
            self.saved_steps: list[int] = []

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
                absolute_step = state.global_step + self.step_offset
                path = f"{self.checkpoint_dir}/checkpoint-{absolute_step}"
                self.model_ref.save_pretrained(path)
                self.tokenizer_ref.save_pretrained(path)
                print(f"Saved adapter checkpoint: {path}")

                self.saved_steps.append(absolute_step)
                self._cleanup_old_checkpoints()

        def _cleanup_old_checkpoints(self) -> None:
            """Delete checkpoints beyond the most recent `keep_last_n`,
            saved by THIS run only - never touches checkpoints from
            earlier resumes that predate step_offset, since those aren't
            tracked in self.saved_steps."""
            import shutil

            while len(self.saved_steps) > self.keep_last_n:
                oldest_step = self.saved_steps.pop(0)
                old_path = Path(f"{self.checkpoint_dir}/checkpoint-{oldest_step}")
                if old_path.exists():
                    shutil.rmtree(old_path)
                    print(f"Deleted old checkpoint to save space: {old_path}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        # No dataset_text_field - train_dataset has "prompt"/"completion"
        # columns (see dataset.py), which SFTTrainer auto-detects and
        # combines with completion_only_loss above for proper masking.
    )
    trainer.add_callback(
        PeriodicAdapterSaveCallback(
            model_ref=model,
            tokenizer_ref=tokenizer,
            checkpoint_dir=config["checkpoint_dir"],
            every_n_steps=config["training"]["save_every_n_steps"],
            step_offset=config["training"].get("step_offset", 0),
            keep_last_n=config["training"].get("keep_last_n_checkpoints", 3),
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
