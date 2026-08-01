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
from src.prompt_formatter import CHATML_IM_END, format_prompt
from src.sanitizer import extract_sql
from src.schema_pruning import prune_schema
from src.sql_postprocess import correct_column_names, correct_table_names, normalize_string_quotes
from src.utils import set_seed, setup_logging
from src.value_retrieval import find_value_hints


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


def run_eval(
    model: Any,
    tokenizer: Any,
    examples: list[dict],
    db_dir: str,
    timeout: int,
    use_postprocessing: bool = False,
    use_value_retrieval: bool = False,
    use_schema_pruning: bool = False,
) -> dict:
    """Run one model over the fixed sample, return aggregated metrics.

    Shared by both the baseline and LoRA passes - the only thing that
    differs between calls is which (model, tokenizer) pair is passed in.

    Args:
        use_postprocessing: if True, applies correct_table_names,
            correct_column_names, and normalize_string_quotes to the
            extracted SQL before scoring - see sql_postprocess.py and
            technical_lessons_learned.md (Priority 1, "Deterministic
            Execution Guard").
        use_value_retrieval: if True, searches the question for candidate
            value mentions and injects hints about matching real DB
            content into the prompt BEFORE generation - see
            value_retrieval.py, implementation_priority_plan.md Priority 2.
        use_schema_pruning: if True, removes low-relevance tables from
            the schema shown in the prompt (wide schemas only - see
            schema_pruning.py, implementation_priority_plan.md Priority
            3). Applied BEFORE building the prompt and BEFORE value
            retrieval (so hints are only searched within the tables the
            model actually sees) - but postprocessing correction below
            still uses the FULL original schema (example["schema"]), not
            the pruned one, since a hallucinated reference to a pruned
            table should still be correctable against the real schema.
        All three default to False so baseline/earlier results remain
        reproducible without these flags - always pass explicitly.
    """
    results = []
    for i, example in enumerate(examples):
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"

        prompt_schema = example["schema"]
        if use_schema_pruning:
            prompt_schema = prune_schema(prompt_schema, example["question"])

        prompt = format_prompt(prompt_schema, example["question"])

        if use_value_retrieval:
            hints = find_value_hints(example["question"], prompt_schema, db_path)
            if hints:
                # Insert hints right before the closing of the user turn
                # (last CHATML_IM_END before the open assistant turn) -
                # doesn't require modifying prompt_formatter.py itself.
                marker = f"{CHATML_IM_END}\n"
                insert_at = prompt.rfind(marker)
                hint_text = "\n" + "\n".join(hints)
                prompt = prompt[:insert_at] + hint_text + prompt[insert_at:]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted_sql = extract_sql(raw_output)
        if use_postprocessing and predicted_sql is not None:
            # Deliberately uses the FULL original schema, not prompt_schema -
            # see docstring note on use_schema_pruning above.
            predicted_sql = correct_table_names(predicted_sql, example["schema"]["table_names_original"])
            predicted_sql = correct_column_names(predicted_sql, example["schema"])
            predicted_sql = normalize_string_quotes(predicted_sql, example["schema"])

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
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply correct_table_names/correct_column_names/normalize_string_quotes "
        "before scoring - see technical_lessons_learned.md, Priority 1 (Execution Guard). "
        "Off by default so past results (results_baseline.json, results_lora.json without "
        "this flag) remain directly comparable unless explicitly re-run with it.",
    )
    parser.add_argument(
        "--value-retrieval",
        action="store_true",
        help="Inject DB-content hints into the prompt for question words that match real "
        "column values - see value_retrieval.py, implementation_priority_plan.md Priority 2. "
        "Independent of --postprocess - combine both flags to test them together, or run "
        "separately first to measure each effect in isolation.",
    )
    parser.add_argument(
        "--schema-pruning",
        action="store_true",
        help="Remove low-relevance tables from wide schemas before building the prompt - "
        "see schema_pruning.py, implementation_priority_plan.md Priority 3. Independent flag, "
        "combine with the others or run alone to measure its effect in isolation.",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    print(f"Loaded config: {config}")
    print(f"Post-processing (Levenshtein/case/quote fixes): {'ON' if args.postprocess else 'OFF'}")
    print(f"Value retrieval (DB-content hints): {'ON' if args.value_retrieval else 'OFF'}")
    print(f"Schema pruning (wide schemas only): {'ON' if args.schema_pruning else 'OFF'}")

    examples, db_dir = load_sample(config)
    print(f"Evaluating on {len(examples)} examples (fixed sample, seed=42)")

    timeout = config["sql_timeout_seconds"]
    output_dir = config["output_dir"]
    baseline_result = None
    lora_result = None

    # Suffix output filenames based on which techniques are on, so
    # different combinations don't silently overwrite each other's
    # results - keeps every combination available for comparison.
    suffix_parts = []
    if args.postprocess:
        suffix_parts.append("postprocessed")
    if args.value_retrieval:
        suffix_parts.append("valueretrieval")
    if args.schema_pruning:
        suffix_parts.append("schemapruning")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    if args.model in ("baseline", "both"):
        from src.model import load_base_model  # local import - GPU-only dep

        print("Loading base model...")
        model, tokenizer = load_base_model(model_name=config["models"]["base_model_name"])
        baseline_result = run_eval(
            model, tokenizer, examples, db_dir, timeout,
            use_postprocessing=args.postprocess, use_value_retrieval=args.value_retrieval,
            use_schema_pruning=args.schema_pruning,
        )
        print("\n=== BASELINE (zero-shot) RESULTS ===")
        print(json.dumps(baseline_result, indent=2))
        with open(f"{output_dir}/results_baseline{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(baseline_result, f, indent=2)
        del model  # free VRAM before loading the second model, if running "both"

    if args.model in ("lora", "both"):
        from src.model import load_lora_model  # local import - GPU-only dep

        checkpoint_path = f"{config['models']['lora_checkpoint_dir']}/{config['models']['lora_checkpoint']}"
        print(f"Loading LoRA model from {checkpoint_path}...")
        model, tokenizer = load_lora_model(checkpoint_path, model_name=config["models"]["base_model_name"])
        lora_result = run_eval(
            model, tokenizer, examples, db_dir, timeout,
            use_postprocessing=args.postprocess, use_value_retrieval=args.value_retrieval,
            use_schema_pruning=args.schema_pruning,
        )
        print("\n=== LoRA (fine-tuned) RESULTS ===")
        print(json.dumps(lora_result, indent=2))
        with open(f"{output_dir}/results_lora{suffix}.json", "w", encoding="utf-8") as f:
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
