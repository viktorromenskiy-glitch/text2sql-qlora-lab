"""Milestone 2/3: unified evaluation script - baseline and/or LoRA model
on Spider dev, honest side-by-side comparison.

Supersedes the earlier evaluate_baseline.py / evaluate_lora.py split
(kept both working during Milestone 1/2 for practical reasons - see
technical_lessons_learned.md and the explanatory memo accompanying this
refactor). This is the single entry point going forward for Spider.

BIRD-SQL is NOT yet wired in - see module_specifications.md's
"РАСХОЖДЕНИЕ С РЕАЛИЗАЦИЕЙ" note. Adding it is planned as a Milestone 3
extension to THIS file (not a new script), once BIRD-SQL is actually
downloaded and its schema-attachment gap (see src/dataset.py docstring)
is resolved.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import yaml

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import aggregate_results, evaluate_example
from src.prompt_formatter import format_prompt
from src.sanitizer import extract_sql
from src.utils import set_seed, setup_logging


def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the eval config (see configs/eval_config.yaml)."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sample(config: dict[str, Any]) -> tuple[list[dict], str]:
    """Load the fixed Spider dev sample used across baseline and LoRA runs.

    Returns (examples, db_dir) - same seed/sample_size every call, so
    baseline and LoRA are always compared on identical examples (see
    technical_assignment.md fairness requirement).
    """
    set_seed(42)
    data_dir = config["datasets"]["spider_dev"]
    examples = load_spider("dev", data_dir=data_dir)[: config["sample_size"]]
    db_dir = get_spider_db_dir("dev", data_dir=data_dir)
    return examples, db_dir


def run_eval(model: Any, tokenizer: Any, examples: list[dict], db_dir: str, timeout: int) -> dict:
    """Run one model over the fixed sample, return aggregated metrics.

    Shared by both the baseline and LoRA passes - the only thing that
    differs between calls is which (model, tokenizer) pair is passed in.
    """
    results = []
    for i, example in enumerate(examples):
        prompt = format_prompt(example["schema"], example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted_sql = extract_sql(raw_output)
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        results.append(evaluate_example(
            predicted_sql=predicted_sql,
            gold_sql=example["query"],
            db_path=db_path,
            difficulty=example["difficulty"],
            timeout=timeout,
        ))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    return aggregate_results(results)


def print_comparison(baseline: dict, lora: dict) -> None:
    """Print an honest side-by-side comparison - see evaluate_lora.py's
    original version of this, unchanged logic, just moved here."""
    print("\n=== BASELINE vs LoRA ===")
    print(f"Overall:  {baseline['overall_accuracy']:.3f} -> {lora['overall_accuracy']:.3f}")
    for difficulty in ["easy", "medium", "hard", "extra"]:
        b = baseline["accuracy_by_difficulty"].get(difficulty, 0.0)
        l = lora["accuracy_by_difficulty"].get(difficulty, 0.0)
        print(f"{difficulty:8s}: {b:.3f} -> {l:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--model", choices=["baseline", "lora", "both"], default="both")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    print(f"Loaded config: {config}")

    examples, db_dir = load_sample(config)
    print(f"Evaluating on {len(examples)} examples (fixed sample, seed=42)")

    timeout = config["sql_timeout_seconds"]
    output_dir = config["output_dir"]
    baseline_result = None
    lora_result = None

    if args.model in ("baseline", "both"):
        from src.model import load_base_model  # local import - GPU-only dep

        print("Loading base model...")
        model, tokenizer = load_base_model(model_name=config["models"]["base_model_name"])
        baseline_result = run_eval(model, tokenizer, examples, db_dir, timeout)
        print("\n=== BASELINE (zero-shot) RESULTS ===")
        print(json.dumps(baseline_result, indent=2))
        with open(f"{output_dir}/results_baseline.json", "w", encoding="utf-8") as f:
            json.dump(baseline_result, f, indent=2)
        del model  # free VRAM before loading the second model, if running "both"

    if args.model in ("lora", "both"):
        from src.model import load_lora_model  # local import - GPU-only dep

        checkpoint_path = f"{config['models']['lora_checkpoint_dir']}/{config['models']['lora_checkpoint']}"
        print(f"Loading LoRA model from {checkpoint_path}...")
        model, tokenizer = load_lora_model(checkpoint_path, model_name=config["models"]["base_model_name"])
        lora_result = run_eval(model, tokenizer, examples, db_dir, timeout)
        print("\n=== LoRA (fine-tuned) RESULTS ===")
        print(json.dumps(lora_result, indent=2))
        with open(f"{output_dir}/results_lora.json", "w", encoding="utf-8") as f:
            json.dump(lora_result, f, indent=2)

    if baseline_result is None and args.model == "lora":
        try:
            with open(f"{output_dir}/results_baseline.json", encoding="utf-8") as f:
                baseline_result = json.load(f)
        except FileNotFoundError:
            pass

    if baseline_result is not None and lora_result is not None:
        print_comparison(baseline_result, lora_result)


if __name__ == "__main__":
    main()
