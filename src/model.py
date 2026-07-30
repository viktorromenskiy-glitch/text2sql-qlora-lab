"""Base and LoRA model loading via Unsloth.

GPU-only module: `unsloth`/`torch` are deliberately not in requirements.txt
(see its header comment) and are not installed in this local dev
environment. This file is written and syntax/import-structure checked
locally, per the division of labor agreed at project start, but every
function in it has only ever been executed in Colab, per RUNBOOK.md - not
here. Treat it as unverified until a real Colab run confirms it.
"""

from __future__ import annotations

from typing import Any

MAX_SEQ_LENGTH = 4096  # Generous headroom for BIRD-SQL's longer schemas
# (see technical_assignment.md OOM risk note); actual prompts are usually
# much shorter after prompt_formatter's minified DDL.


def load_base_model(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> tuple[Any, Any]:
    """Load the un-tuned base model in 4-bit, for the zero-shot baseline.

    Args:
        model_name: HuggingFace model id, per technical_assignment.md.

    Returns:
        A `(model, tokenizer)` pair, ready for generation - no LoRA
        adapter attached.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # let Unsloth pick the best dtype for the available GPU
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_lora_model(checkpoint_path: str, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> tuple[Any, Any]:
    """Load the base model with a trained LoRA adapter attached.

    Args:
        checkpoint_path: Path to a LoRA adapter saved by `scripts/train.py`
            (under `train_config.yaml`'s `checkpoint_dir`).
        model_name: Base model id - must match what the adapter was
            trained against.

    Returns:
        A `(model, tokenizer)` pair with the LoRA adapter applied, ready
        for generation.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def apply_lora_config(model: Any, config: dict) -> Any:
    """Attach a trainable LoRA adapter to a freshly loaded base model.

    Args:
        model: A base model from `load_base_model` (before
            `FastLanguageModel.for_inference` - this is for training, not
            inference).
        config: The `lora` section of `train_config.yaml` (`r`, `alpha`,
            `target_modules`).

    Returns:
        The model with a LoRA adapter attached, ready for `SFTTrainer`.
    """
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        lora_alpha=config["alpha"],
        target_modules=config["target_modules"],
        use_gradient_checkpointing=True,
    )


def load_lora_checkpoint_for_training(checkpoint_path: str, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> tuple[Any, Any]:
    """Load an existing LoRA adapter checkpoint, kept TRAINABLE - for
    continuing/resuming training, not evaluation.

    Deliberately distinct from `load_lora_model()`: that one calls
    `FastLanguageModel.for_inference(model)`, which puts the model in a
    mode unsuitable for further training (real risk found while
    extending training past checkpoint-1500 - see
    technical_lessons_learned.md). This function skips that call, so
    the loaded adapter's weights remain trainable.

    Args:
        checkpoint_path: Path to a LoRA adapter saved by `scripts/train.py`.
        model_name: Base model id - must match what the adapter was
            trained against.

    Returns:
        A `(model, tokenizer)` pair with the LoRA adapter applied, ready
        to pass into `SFTTrainer` for further training.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    # Deliberately NOT calling for_inference() here - see docstring.
    return model, tokenizer
