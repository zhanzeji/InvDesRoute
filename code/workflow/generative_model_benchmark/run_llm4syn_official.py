#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import torch
from pymatgen.core import Composition
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path("/home/zhzhan/models/official_precursor_baselines/LLM4SYN-tgt2ceq-dgpt2-v1.2.1")
BENCHMARK = ROOT / "data" / "benchmark_24x5.csv"
OUTPUT_DIR = ROOT / "predictions"
MODEL_NAME = "LLM4SYN-TGT2CEQ-v1.2.1-official"
SOURCE_REPO = "RyotaroOKabe/tgt2ceq_dgpt2_v1.2.1"
SOURCE_CODE = "https://github.com/RyotaroOKabe/llm4syn"
BASE_SEED = 20260729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_formula(value: str) -> str:
    return Composition(value.strip()).reduced_formula


def targets() -> list[str]:
    with BENCHMARK.open(encoding="utf-8", newline="") as handle:
        return sorted({normalize_formula(row["target"]) for row in csv.DictReader(handle)})


def first_equation(text: str) -> str:
    generated = text.split("||", 1)[-1]
    generated = re.split(r"[\x00-\x1f\x7f;]", generated, maxsplit=1)[0]
    return generated.strip()


def parse_lhs(text: str, target: str) -> list[str] | None:
    equation = first_equation(text)
    if "->" not in equation:
        return None
    lhs, rhs = equation.split("->", 1)
    normalized_target = normalize_formula(target)
    try:
        rhs_formulas = [
            normalize_formula(re.sub(r"^\s*(?:\d+(?:\.\d+)?|\.\d+)\s*", "", token))
            for token in rhs.split("+")
            if token.strip()
        ]
    except Exception:
        return None
    if normalized_target not in rhs_formulas:
        return None

    formulas = []
    for token in lhs.split("+"):
        formula = re.sub(r"^\s*(?:\d+(?:\.\d+)?|\.\d+)\s*", "", token).strip()
        if not formula:
            continue
        try:
            normalized = normalize_formula(formula)
        except Exception:
            return None
        if normalized != normalized_target:
            formulas.append(normalized)
    formulas = sorted(dict.fromkeys(formulas))
    if not formulas:
        return None
    target_elements = {str(element) for element in Composition(normalized_target).elements}
    precursor_elements = {
        str(element)
        for formula in formulas
        for element in Composition(formula).elements
    }
    return formulas if target_elements.issubset(precursor_elements) else None


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True).to(device).eval()
    prediction_rows = []
    raw_rows = []

    for target_index, target in enumerate(targets()):
        prompt = f"{target} ||"
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        candidates: list[tuple[list[str], str]] = []
        seen = set()
        for batch_index in range(10):
            seed = BASE_SEED + target_index * 100 + batch_index
            set_seed(seed)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_length=encoded["input_ids"].shape[1] * 12,
                    num_beams=2,
                    num_return_sequences=2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for text in tokenizer.batch_decode(generated, skip_special_tokens=True):
                formulas = parse_lhs(text, target)
                raw_rows.append({
                    "model": MODEL_NAME,
                    "target": target,
                    "seed": seed,
                    "raw_generation": text,
                    "first_equation": first_equation(text),
                    "parsed_precursors": ";".join(formulas or []),
                })
                if formulas is None:
                    continue
                key = tuple(formulas)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((formulas, first_equation(text)))
        for rank, (formulas, equation) in enumerate(candidates[:20], start=1):
            prediction_rows.append({
                "model": MODEL_NAME,
                "target": target,
                "rank": rank,
                "precursors": ";".join(formulas),
                "byproducts": "",
                "balanced_reaction": equation,
                "notes": "official_tgt2ceq_checkpoint; first_equation_lhs_only",
            })
        print(f"{target}: {len(candidates)} unique valid precursor sets", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "llm4syn-tgt2ceq-v1.2.1.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with output.with_name(f"{output.stem}_raw.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    metadata = {
        "model": MODEL_NAME,
        "source_repo": SOURCE_REPO,
        "source_code": SOURCE_CODE,
        "checkpoint_sha256": sha256(MODEL_PATH / "pytorch_model.bin"),
        "official_prompt_format": "{target} ||",
        "official_generation_settings_retained": {"num_beams": 2, "do_sample": True},
        "evaluation_sampling": {"fixed_base_seed": BASE_SEED, "batches_per_target": 10, "sequences_per_batch": 2},
        "postprocessing": "First generated equation only; precursor set parsed from the left-hand side; formula reduction and target-element coverage validation.",
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
