#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from pymatgen.core import Composition
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent


def normalize_formula(value: str) -> str:
    value = str(value).strip().split("·", 1)[0]
    return Composition(value).reduced_formula


def formulas_from_text(text: str, target: str) -> list[str]:
    formulas = []
    normalized_target = normalize_formula(target)
    for token in re.findall(r"[A-Z][A-Za-z0-9().·]+", text):
        try:
            formula = normalize_formula(token)
        except Exception:
            continue
        if formula != normalized_target:
            formulas.append(formula)
    return sorted(dict.fromkeys(formulas))


def parse_precursor_sets(text: str, target: str) -> tuple[list[list[str]], str]:
    candidates = []
    for list_text in re.findall(r"\[[^\[\]]*\]", text, flags=re.S):
        try:
            parsed = ast.literal_eval(list_text)
        except Exception:
            continue
        if not isinstance(parsed, (list, tuple)):
            continue
        formulas = []
        for value in parsed:
            try:
                formula = normalize_formula(value)
            except Exception:
                continue
            if formula != normalize_formula(target):
                formulas.append(formula)
        formulas = sorted(dict.fromkeys(formulas))
        if formulas:
            candidates.append(formulas)
    if candidates:
        return candidates, "literal_lists"
    for line in re.findall(r"(?:^|\n)\s*\d+\.\s*([^\n]+)", text):
        formulas = formulas_from_text(line, target)
        if formulas:
            candidates.append(formulas)
    if candidates:
        return candidates, "numbered_methods"
    formulas = formulas_from_text(text, target)
    return ([formulas] if formulas else []), "formula_regex" if formulas else "unparsed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official CSLLM precursor LoRA on the 24-target benchmark.")
    parser.add_argument("--base-model", type=Path, default=Path("/home/zhzhan/models/csllm/llama3-8bf-hf"))
    parser.add_argument("--adapter", type=Path, default=Path("/home/zhzhan/models/csllm/precursor_llm_llama3"))
    parser.add_argument("--targets", type=Path, default=ROOT / "external_model_interface" / "external_model_targets.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "external_model_interface" / "csllm_precursor_top5.csv")
    parser.add_argument("--model-name", default="CSLLM-precursor-llama3-release")
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True)
    model = model.to(args.device).eval()
    targets = pd.read_csv(args.targets)
    output_rows = []
    raw_rows = []
    for _, target_row in targets.iterrows():
        target = str(target_row["target"])
        prompt = f'input: How can this material structure be synthesized "{target}%"?'
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                do_sample=False,
                early_stopping=args.num_beams > 1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated.sequences[:, prompt_length:], skip_special_tokens=True)
        generated_scores = getattr(generated, "sequences_scores", None)
        sequence_scores = generated_scores.detach().cpu().tolist() if generated_scores is not None else [None] * len(texts)
        seen = set()
        rank = 0
        for beam_index, (text, sequence_score) in enumerate(zip(texts, sequence_scores), start=1):
            precursor_sets, parse_status = parse_precursor_sets(text, target)
            raw_rows.append(
                {
                    "target": target,
                    "beam_index": beam_index,
                    "sequence_score": sequence_score,
                    "raw_generation": text,
                    "parse_status": parse_status,
                    "parsed_precursors": " | ".join(";".join(formulas) for formulas in precursor_sets),
                }
            )
            for set_index, formulas in enumerate(precursor_sets, start=1):
                key = tuple(formulas)
                if key in seen:
                    continue
                seen.add(key)
                rank += 1
                output_rows.append(
                    {
                        "model": args.model_name,
                        "target": target,
                        "rank": rank,
                        "precursors": ";".join(formulas),
                        "byproducts": "",
                        "balanced_reaction": "",
                        "notes": f"beam={beam_index}; set={set_index}; score={sequence_score}; parse={parse_status}",
                    }
                )
                if rank >= args.num_return_sequences:
                    break
            if rank >= args.num_return_sequences:
                break
        print(f"{target}: {rank} unique precursor sets from {len(texts)} beams", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    pd.DataFrame(raw_rows).to_csv(args.output.with_name(args.output.stem + "_raw.csv"), index=False)
    metadata = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": 'input: How can this material structure be synthesized "{target}%"?',
        "torch_dtype": "float16",
        "device": args.device,
        "model_name": args.model_name,
    }
    args.output.with_name(args.output.stem + "_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
