#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from pymatgen.core import Composition
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "data" / "benchmark_24x5.csv"
DEFAULT_OUTPUT_DIR = ROOT / "predictions"


@dataclass(frozen=True)
class ModelProfile:
    model_name: str
    model_path: str
    source_repo: str
    revision: str
    prompt_style: str
    trust_remote_code: bool = False


MODEL_PROFILES = {
    "chemdfm-v1.5-8b": ModelProfile(
        model_name="ChemDFM-v1.5-8B-official-zero-shot",
        model_path="/home/zhzhan/models/official_precursor_baselines/ChemDFM-v1.5-8B",
        source_repo="OpenDFM/ChemDFM-v1.5-8B",
        revision="f5790d56a903ce480b1eff8d0adf9613d8acee0c",
        prompt_style="chemdfm",
    ),
    "chemllm-7b-chat": ModelProfile(
        model_name="ChemLLM-7B-Chat-official-zero-shot",
        model_path="/home/zhzhan/models/official_precursor_baselines/ChemLLM-7B-Chat",
        source_repo="AI4Chem/ChemLLM-7B-Chat",
        revision="b8b2ea19e48f53d190fe8dced94572717f8e89a2",
        prompt_style="chemllm_internlm2",
        trust_remote_code=True,
    ),
    "mistral-7b-instruct-v0.3": ModelProfile(
        model_name="Mistral-7B-Instruct-v0.3-official-zero-shot",
        model_path="/home/zhzhan/models/official_precursor_baselines/Mistral-7B-Instruct-v0.3",
        source_repo="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        prompt_style="chat_template",
    ),
}


SYSTEM_PROMPT = (
    "You are an inorganic solid-state synthesis chemist. Predict precursor sets, not crystal structures "
    "or processing conditions. Use chemically valid formula strings and follow the requested JSON schema."
)


def task_prompt(target: str, num_candidates: int) -> str:
    return f"""Target compound: {target}

Propose {num_candidates} distinct precursor sets that could be used to synthesize this inorganic target. A set may
contain elements, oxides, carbonates, hydroxides, nitrates, phosphates, ammonium salts, or gas reagents when
chemically relevant. Every set must collectively supply all elements in the target. Do not include the target itself,
solvents, equipment, temperatures, or explanatory prose.

Return only a JSON array of arrays in descending recommendation order, for example:
[["BaCO3", "TiO2"], ["BaO", "TiO2"]]"""


def normalize_formula(value: str) -> str:
    text = str(value).strip().replace(" ", "").replace("·", ".")
    if not text or len(text) > 80:
        raise ValueError("empty or overlong formula")
    composition = Composition(text)
    return composition.reduced_formula


def candidate_is_valid(target: str, formulas: list[str]) -> bool:
    target_elements = {str(element) for element in Composition(target).elements}
    precursor_elements = {
        str(element)
        for formula in formulas
        for element in Composition(formula).elements
    }
    return target_elements.issubset(precursor_elements)


def coerce_candidate(value: object, target: str) -> list[str] | None:
    if isinstance(value, str):
        values = re.split(r"\s*(?:\+|;|,)\s*", value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return None
    normalized_target = normalize_formula(target)
    formulas = []
    for raw_formula in values:
        try:
            formula = normalize_formula(str(raw_formula))
        except Exception:
            continue
        if formula != normalized_target:
            formulas.append(formula)
    formulas = sorted(dict.fromkeys(formulas))
    if not formulas or not candidate_is_valid(normalized_target, formulas):
        return None
    return formulas


def parse_candidates(text: str, target: str, limit: int) -> tuple[list[list[str]], str]:
    candidate_blocks: list[object] = []
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and (not parsed or any(isinstance(value, (list, tuple)) for value in parsed)):
            candidate_blocks = parsed
            break
    parse_status = "json"
    if not candidate_blocks:
        for block in re.findall(r"\[[^\[\]]+\]", text):
            try:
                candidate_blocks.append(ast.literal_eval(block))
            except Exception:
                continue
        parse_status = "literal_lists"
    if not candidate_blocks:
        candidate_blocks = [line for line in re.findall(r"(?:^|\n)\s*\d+[.)]\s*([^\n]+)", text)]
        parse_status = "numbered_lines"

    output = []
    seen = set()
    for value in candidate_blocks:
        candidate = coerce_candidate(value, target)
        if candidate is None:
            continue
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
        if len(output) >= limit:
            break
    return output, parse_status if output else "unparsed_or_invalid"


def format_prompt(tokenizer, profile: ModelProfile, target: str, num_candidates: int) -> str:
    query = task_prompt(target, num_candidates)
    if profile.prompt_style == "chemdfm":
        return f"[Round 0]\nHuman: {SYSTEM_PROMPT}\n\n{query}\nAssistant:"
    if profile.prompt_style == "chemllm_internlm2":
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\nUser: {query}\nAssistant:"


def benchmark_targets(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return sorted({normalize_formula(row["target"]) for row in csv.DictReader(handle)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODEL_PROFILES))
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    args = parser.parse_args()

    profile = MODEL_PROFILES[args.model]
    model_path = Path(profile.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=profile.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=profile.trust_remote_code,
    ).to(args.device).eval()

    predictions = []
    raw_outputs = []
    for target in benchmark_targets(args.benchmark):
        prompt = format_prompt(tokenizer, profile, target, args.num_candidates)
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        candidates, parse_status = parse_candidates(text, target, args.num_candidates)
        for rank, formulas in enumerate(candidates, start=1):
            predictions.append(
                {
                    "model": profile.model_name,
                    "target": target,
                    "rank": rank,
                    "precursors": ";".join(formulas),
                    "byproducts": "",
                    "balanced_reaction": "",
                    "notes": f"parse={parse_status}; deterministic_zero_shot",
                }
            )
        raw_outputs.append(
            {
                "model": profile.model_name,
                "target": target,
                "parse_status": parse_status,
                "n_valid_candidates": len(candidates),
                "raw_generation": text,
            }
        )
        print(f"{profile.model_name} {target}: {len(candidates)} valid sets ({parse_status})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.model}.csv"
    raw_path = args.output_dir / f"{args.model}_raw.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]) if predictions else ["model", "target", "rank", "precursors", "byproducts", "balanced_reaction", "notes"])
        writer.writeheader()
        writer.writerows(predictions)
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_outputs[0]))
        writer.writeheader()
        writer.writerows(raw_outputs)
    metadata = {
        "profile": asdict(profile),
        "benchmark": str(args.benchmark),
        "n_targets": len(benchmark_targets(args.benchmark)),
        "num_candidates_requested": args.num_candidates,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False, "num_beams": 1},
        "system_prompt": SYSTEM_PROMPT,
        "task_prompt_template": task_prompt("{target}", args.num_candidates),
        "postprocessing": "JSON/list parsing, pymatgen formula reduction, candidate deduplication, and target-element coverage validation only.",
        "torch_dtype": "float16",
        "device": args.device,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
