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

from src.advanced_self_correction import generate_n_candidates, majority_vote
from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import aggregate_results, evaluate_example, execute_sql
from src.few_shot import find_similar_examples, format_few_shot_block
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
    train_examples: list[dict] | None = None,
    use_advanced_correction: bool = False,
    n_candidates: int = 5,
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
        train_examples: Spider TRAIN split examples (see few_shot.py) -
            REQUIRED if the caller wants few-shot examples inserted
            (pass a non-empty list to enable it). Kept as an explicit
            parameter rather than a bool flag + internal loading, since
            loading the train split is expensive and should happen ONCE
            in main(), not per run_eval() call (baseline AND lora each
            call run_eval separately).
        use_advanced_correction: if True, generates n_candidates SQL
            completions via SAMPLING (not the single greedy generation
            used otherwise) and picks by majority vote on the execution
            RESULT - see advanced_self_correction.py,
            implementation_priority_plan.md Priority 5. Expensive
            (n_candidates x the generation cost of every other flag) -
            each candidate still goes through use_postprocessing if that
            flag is also on, before voting, so voting reflects our best
            per-candidate correction, not raw model output.
        n_candidates: how many sampled candidates per example when
            use_advanced_correction is True - ignored otherwise.
        All boolean flags default to False so baseline/earlier results
        remain reproducible without them - always pass explicitly.
    """
    results = []
    for i, example in enumerate(examples):
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"

        prompt_schema = example["schema"]
        if use_schema_pruning:
            prompt_schema = prune_schema(prompt_schema, example["question"])

        prompt = format_prompt(prompt_schema, example["question"])

        if train_examples:
            similar = find_similar_examples(example["question"], train_examples, k=2)
            few_shot_block = format_few_shot_block(similar)
            if few_shot_block:
                # Insert right before "Question: " - format_prompt always
                # builds "Schema:\n{ddl}\n\nQuestion: {question}" as the
                # user turn content, so this literal marker is stable
                # without needing to modify prompt_formatter.py itself
                # (same splice approach as the value-retrieval hints below).
                marker = "\n\nQuestion: "
                insert_at = prompt.rfind(marker)
                prompt = prompt[:insert_at] + "\n\n" + few_shot_block + prompt[insert_at:]

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

        def postprocess(sql: str | None) -> str | None:
            if use_postprocessing and sql is not None:
                # Deliberately uses the FULL original schema, not
                # prompt_schema - see docstring note on use_schema_pruning.
                sql = correct_table_names(sql, example["schema"]["table_names_original"])
                sql = correct_column_names(sql, example["schema"])
                sql = normalize_string_quotes(sql, example["schema"])
            return sql

        if use_advanced_correction:
            raw_candidates = generate_n_candidates(model, tokenizer, prompt, n=n_candidates)
            candidates_with_results = []
            for candidate_sql in raw_candidates:
                candidate_sql = postprocess(candidate_sql)
                candidate_rows = execute_sql(candidate_sql, db_path, timeout) if candidate_sql is not None else None
                candidates_with_results.append((candidate_sql, candidate_rows))
            predicted_sql = majority_vote(candidates_with_results)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted_sql = postprocess(extract_sql(raw_output))

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
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Insert 2 similar examples (by question word overlap) from Spider TRAIN before "
        "the question - see few_shot.py, implementation_priority_plan.md Priority 4. "
        "Literature is contradictory on whether this helps on top of SFT - see "
        "technical_lessons_learned.md. Independent flag, combine or run alone.",
    )
    parser.add_argument(
        "--advanced-self-correction",
        action="store_true",
        help="Generate 5 candidates via sampling and pick by majority vote on execution "
        "result - see advanced_self_correction.py, implementation_priority_plan.md Priority "
        "5. EXPENSIVE (5x generation cost). Different from the already-tested, rejected "
        "simple self_correction.py (single retry, 0 effect) - see technical_lessons_learned.md.",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    print(f"Loaded config: {config}")
    print(f"Post-processing (Levenshtein/case/quote fixes): {'ON' if args.postprocess else 'OFF'}")
    print(f"Value retrieval (DB-content hints): {'ON' if args.value_retrieval else 'OFF'}")
    print(f"Schema pruning (wide schemas only): {'ON' if args.schema_pruning else 'OFF'}")
    print(f"Few-shot (2 similar train examples): {'ON' if args.few_shot else 'OFF'}")
    print(f"Advanced self-correction (5-candidate voting): {'ON' if args.advanced_self_correction else 'OFF'}")

    examples, db_dir = load_sample(config)
    print(f"Evaluating on {len(examples)} examples (fixed sample, seed=42)")

    train_examples = None
    if args.few_shot:
        print("Loading Spider TRAIN split for few-shot retrieval (never dev - no leakage)...")
        train_examples = load_spider("train", data_dir=config["datasets"]["spider_dev"])
        print(f"Loaded {len(train_examples)} train examples for retrieval")

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
    if args.few_shot:
        suffix_parts.append("fewshot")
    if args.advanced_self_correction:
        suffix_parts.append("advcorrection")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    if args.model in ("baseline", "both"):
        from src.model import load_base_model  # local import - GPU-only dep

        print("Loading base model...")
        model, tokenizer = load_base_model(model_name=config["models"]["base_model_name"])
        baseline_result = run_eval(
            model, tokenizer, examples, db_dir, timeout,
            use_postprocessing=args.postprocess, use_value_retrieval=args.value_retrieval,
            use_schema_pruning=args.schema_pruning, train_examples=train_examples,
            use_advanced_correction=args.advanced_self_correction,
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
            use_schema_pruning=args.schema_pruning, train_examples=train_examples,
            use_advanced_correction=args.advanced_self_correction,
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
