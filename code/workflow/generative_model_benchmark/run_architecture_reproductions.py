#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import (
    ELEMENTS,
    composition_vector,
    instantiate_template,
    precursor_template,
    route_feature,
    single_source_precursor,
    source_elements,
    split_formulas,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "predictions"
CHECKPOINTS = ROOT / "checkpoints"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_predictions(model: str, predictions: dict[str, list[tuple[tuple[str, ...], float]]]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}.csv"
    rows = []
    for target, candidates in predictions.items():
        seen = set()
        rank = 0
        for precursor_set, score in candidates:
            precursor_set = tuple(sorted(set(precursor_set)))
            if not precursor_set or precursor_set in seen:
                continue
            seen.add(precursor_set)
            rank += 1
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "rank": rank,
                    "precursors": ";".join(precursor_set),
                    "score": score,
                }
            )
            if rank >= 20:
                break
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "target", "rank", "precursors", "score"])
        writer.writeheader()
        writer.writerows(rows)
    return path


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 384) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def train_elemwise(train: list[dict[str, str]], targets: list[str], device: torch.device, epochs: int, tag: str) -> Path:
    template_counts = Counter()
    raw_examples = []
    for row in train:
        target = row["target"]
        precursors = split_formulas(row["precursors"])
        assignments = []
        valid = True
        for precursor in precursors:
            source = single_source_precursor(precursor, target)
            if source is None:
                continue
            assignments.append((source, precursor_template(precursor, source)))
        if len({source for source, _ in assignments}) != len(source_elements(target)):
            valid = False
        if valid:
            for source, template in assignments:
                template_counts[template] += 1
                raw_examples.append((target, source, template))
    templates = sorted(template for template, count in template_counts.items() if count >= 3)
    template_index = {template: index for index, template in enumerate(templates)}
    element_index = {element: index for index, element in enumerate(ELEMENTS)}
    examples = [example for example in raw_examples if example[2] in template_index]
    x = np.zeros((len(examples), len(composition_vector("Li2O")) + len(ELEMENTS)), dtype=np.float32)
    y = np.zeros(len(examples), dtype=np.int64)
    for index, (target, source, template) in enumerate(examples):
        x[index, : len(composition_vector(target))] = composition_vector(target)
        x[index, len(composition_vector(target)) + element_index[source]] = 1.0
        y[index] = template_index[template]
    model = MLP(x.shape[1], len(templates)).to(device)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = F.cross_entropy(model(batch_x), batch_y, label_smoothing=0.02)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch_x)
        if epoch in {0, epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"ElemwiseRetro-style epoch {epoch + 1}: loss={total / len(dataset):.4f}")
    predictions = {}
    model.eval()
    with torch.no_grad():
        for target in targets:
            per_source = []
            for source in source_elements(target):
                feature = np.zeros(x.shape[1], dtype=np.float32)
                feature[: len(composition_vector(target))] = composition_vector(target)
                feature[len(composition_vector(target)) + element_index[source]] = 1.0
                probabilities = F.softmax(model(torch.from_numpy(feature).to(device).unsqueeze(0)), dim=-1)[0]
                values, indices = torch.topk(probabilities, k=min(8, len(templates)))
                options = []
                for value, template_id in zip(values.cpu().tolist(), indices.cpu().tolist()):
                    formula = instantiate_template(templates[template_id], source)
                    if formula:
                        options.append((formula, value))
                per_source.append(options)
            candidates = []
            for combination in itertools.product(*per_source):
                formulas = tuple(item[0] for item in combination)
                score = float(np.prod([item[1] for item in combination]))
                candidates.append((formulas, score))
            predictions[target] = sorted(candidates, key=lambda item: item[1], reverse=True)[:100]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "templates": templates}, CHECKPOINTS / f"elemwise_reproduction{tag}.pt")
    return write_predictions(f"ElemwiseRetro-architecture-reproduction{tag}", predictions)


class RetrievalModel(nn.Module):
    def __init__(self, target_dim: int, route_dim: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.target_encoder = MLP(target_dim, latent_dim, hidden=320)
        self.route_encoder = MLP(route_dim, latent_dim, hidden=320)

    def encode_target(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_encoder(value), dim=-1)

    def encode_route(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.route_encoder(value), dim=-1)


def train_retrieval(train: list[dict[str, str]], targets: list[str], device: torch.device, epochs: int, tag: str) -> Path:
    target_x = np.stack([composition_vector(row["target"]) for row in train]).astype(np.float32)
    route_x = np.stack([route_feature(split_formulas(row["precursors"])) for row in train]).astype(np.float32)
    model = RetrievalModel(target_x.shape[1], route_x.shape[1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(target_x), torch.from_numpy(route_x)), batch_size=512, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_target, batch_route in loader:
            batch_target, batch_route = batch_target.to(device), batch_route.to(device)
            target_embedding = model.encode_target(batch_target)
            route_embedding = model.encode_route(batch_route)
            logits = target_embedding @ route_embedding.T / 0.07
            labels = torch.arange(len(batch_target), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch_target)
        if epoch in {0, epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"Retrieval-Retro-style epoch {epoch + 1}: loss={total / len(train):.4f}")
    model.eval()
    predictions = {}
    with torch.no_grad():
        train_route_embedding = []
        for start in range(0, len(route_x), 2048):
            train_route_embedding.append(model.encode_route(torch.from_numpy(route_x[start:start + 2048]).to(device)).cpu())
        train_route_embedding = torch.cat(train_route_embedding)
        for target in targets:
            query = model.encode_target(torch.from_numpy(composition_vector(target)).to(device).unsqueeze(0)).cpu()[0]
            similarities = train_route_embedding @ query
            values, indices = torch.topk(similarities, k=min(1000, len(train)))
            candidates = []
            for value, index in zip(values.tolist(), indices.tolist()):
                candidates.append((split_formulas(train[index]["precursors"]), float(value)))
            predictions[target] = candidates
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINTS / f"retrieval_reproduction{tag}.pt")
    return write_predictions(f"Retrieval-Retro-architecture-reproduction{tag}", predictions)


class ConditionalVAE(nn.Module):
    def __init__(self, condition_dim: int, output_dim: int, latent_dim: int = 48) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(condition_dim + output_dim, latent_dim * 2, hidden=512)
        self.decoder = MLP(condition_dim + latent_dim, output_dim, hidden=512)
        self.count_head = MLP(condition_dim + latent_dim, 8, hidden=192)

    def forward(self, condition: torch.Tensor, route: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mean, log_variance = self.encoder(torch.cat([condition, route], dim=-1)).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-8, 8)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        decoded = self.decoder(torch.cat([condition, latent], dim=-1))
        counts = self.count_head(torch.cat([condition, latent], dim=-1))
        return decoded, counts, mean, log_variance

    def decode(self, condition: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([condition, latent], dim=-1)
        return self.decoder(inputs), self.count_head(inputs)


def train_cvae(train: list[dict[str, str]], targets: list[str], device: torch.device, epochs: int, tag: str) -> Path:
    formula_counts = Counter(formula for row in train for formula in split_formulas(row["precursors"]))
    vocabulary = sorted(formula for formula, count in formula_counts.items() if count >= 4)
    vocabulary_index = {formula: index for index, formula in enumerate(vocabulary)}
    retained = [row for row in train if all(formula in vocabulary_index for formula in split_formulas(row["precursors"]))]
    condition_x = np.stack([composition_vector(row["target"]) for row in retained]).astype(np.float32)
    route_y = np.zeros((len(retained), len(vocabulary)), dtype=np.float32)
    count_y = np.zeros(len(retained), dtype=np.int64)
    for row_index, row in enumerate(retained):
        precursors = split_formulas(row["precursors"])
        for formula in precursors:
            route_y[row_index, vocabulary_index[formula]] = 1.0
        count_y[row_index] = min(len(precursors), 8) - 1
    model = ConditionalVAE(condition_x.shape[1], route_y.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(condition_x), torch.from_numpy(route_y), torch.from_numpy(count_y)),
        batch_size=256,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    positive_weight = torch.tensor(max(1.0, (route_y.size - route_y.sum()) / max(route_y.sum(), 1)), device=device)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        beta = min(0.08, 0.08 * (epoch + 1) / max(epochs // 2, 1))
        for condition, route, count in loader:
            condition, route, count = condition.to(device), route.to(device), count.to(device)
            decoded, counts, mean, log_variance = model(condition, route)
            reconstruction = F.binary_cross_entropy_with_logits(decoded, route, pos_weight=positive_weight)
            count_loss = F.cross_entropy(counts, count)
            kl = -0.5 * torch.mean(1 + log_variance - mean.square() - log_variance.exp())
            loss = reconstruction + 0.2 * count_loss + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(condition)
        if epoch in {0, epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"CVAE reproduction epoch {epoch + 1}: loss={total / len(retained):.4f}")
    model.eval()
    predictions = {}
    generator = torch.Generator(device=device).manual_seed(20260724)
    with torch.no_grad():
        for target in targets:
            condition = torch.from_numpy(composition_vector(target)).to(device).unsqueeze(0)
            candidates = []
            latent_samples = torch.cat(
                [torch.zeros(1, model.latent_dim, device=device), torch.randn(511, model.latent_dim, generator=generator, device=device)],
                dim=0,
            )
            conditions = condition.repeat(len(latent_samples), 1)
            logits, count_logits = model.decode(conditions, latent_samples)
            probabilities = torch.sigmoid(logits)
            counts = count_logits.argmax(dim=-1) + 1
            for row_index in range(len(latent_samples)):
                count = int(counts[row_index])
                values, indices = torch.topk(probabilities[row_index], k=count)
                precursor_set = tuple(vocabulary[index] for index in indices.cpu().tolist())
                score = float(torch.log(values.clamp_min(1e-8)).sum())
                candidates.append((precursor_set, score))
            predictions[target] = sorted(candidates, key=lambda item: item[1], reverse=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocabulary": vocabulary}, CHECKPOINTS / f"cvae_reproduction{tag}.pt")
    return write_predictions(f"CVAE-precursor-architecture-reproduction{tag}", predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["elemwise", "retrieval", "cvae"], default=["elemwise", "retrieval", "cvae"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = read_rows(DATA / "train_target_disjoint.csv")
    benchmark = read_rows(DATA / "benchmark_24x5.csv")
    targets = sorted({row["target"] for row in benchmark})
    outputs = []
    if "elemwise" in args.models:
        outputs.append(train_elemwise(train, targets, device, args.epochs, args.tag))
    if "retrieval" in args.models:
        outputs.append(train_retrieval(train, targets, device, args.epochs, args.tag))
    if "cvae" in args.models:
        outputs.append(train_cvae(train, targets, device, args.epochs, args.tag))
    metadata = {
        "models": args.models,
        "epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "tag": args.tag,
        "training_records": len(train),
        "held_out_targets": targets,
        "outputs": [str(path) for path in outputs],
        "claim_boundary": "Architecture-level reproductions trained on the common target-disjoint Ceder corpus; these are not author-released checkpoints.",
    }
    (ROOT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
