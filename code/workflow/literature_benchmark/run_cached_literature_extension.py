#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))

from analysis import annotate_reaction_practicality, compute_phase_stability, mp_formation_energy_map, pet_energy_map, write_recommendation, write_report  # noqa: E402
from benchmark_mp_pet import METRIC_COLUMNS, add_category_summaries, plot_summary, target_metrics  # noqa: E402
from common import normalize_formula, write_json  # noqa: E402
from phase_sources import build_phase_pool, fetch_mp_phases  # noqa: E402
from precursor_library import GAS_RESERVOIR_FORMULAS, common_precursor_formulas, reservoir_phase_entries, virtual_common_precursor_entries  # noqa: E402
from reactions import enumerate_decomposition_reactions, enumerate_formation_reactions, reactions_to_frame  # noqa: E402
from targeted_routes import enumerate_targeted_formation_reactions  # noqa: E402
from synthesizability import write_synthesizability_outputs  # noqa: E402


DEFAULT_CEDER = PROJECT / 'literature_benchmark/ceder_text_mined_routes.csv'
DEFAULT_OUT = PROJECT / 'results/literature_seeded_extension_cached'
DEFAULT_PET_ROOTS = [
    PROJECT / 'results/mp_pet_benchmark_120',
    PROJECT / 'results/mp_pet_benchmark',
    PROJECT / 'results/mp_pet_smoke',
]
FORMULA_TOKEN = re.compile(r'[A-Z][A-Za-z0-9().]*')
ALLOWED_EXTRA_REACTANT_ELEMENTS = {'C', 'H', 'N', 'O'}


def normalize(formula: str) -> str:
    return Composition(str(formula).strip()).reduced_formula


def formula_elements(formula: str) -> set[str]:
    return {str(element) for element in Composition(formula).elements}


def chemical_system(formula: str) -> str:
    return '-'.join(sorted(formula_elements(formula)))


def valid_formula(formula: str) -> bool:
    try:
        comp = Composition(formula)
    except Exception:
        return False
    if not comp.as_dict() or len(comp.elements) > 5:
        return False
    return all(amount > 0 for amount in comp.as_dict().values())


def split_formulas(value: str | None) -> list[str]:
    formulas = []
    for item in str(value or '').split(';'):
        item = item.strip()
        if not item:
            continue
        try:
            formulas.append(normalize(item))
        except Exception:
            continue
    return formulas


def formulas_in_text(text: str) -> set[str]:
    formulas = set()
    for token in FORMULA_TOKEN.findall(text or ''):
        try:
            formulas.add(normalize(token))
        except Exception:
            continue
    return formulas


def product_side_contains_target(route_text: str, target: str) -> bool:
    text = route_text or ''
    if '==' in text:
        right = text.split('==', 1)[1]
    elif '->' in text:
        right = text.split('->', 1)[1]
    else:
        right = text
    return target in formulas_in_text(right)


def reactants_are_target_relevant(target: str, precursors: list[str]) -> bool:
    target_elements = formula_elements(target)
    reactant_elements = set().union(*(formula_elements(formula) for formula in precursors)) if precursors else set()
    if not target_elements.issubset(reactant_elements):
        return False
    extra = reactant_elements - target_elements
    return extra.issubset(ALLOWED_EXTRA_REACTANT_ELEMENTS)


def source_doi(note: str | None) -> str:
    match = re.search(r'DOI=([^;]+)', note or '')
    return match.group(1).strip() if match else ''


def is_auditable_route(row: dict[str, str], target: str, precursors: list[str], byproducts: list[str]) -> bool:
    if not precursors:
        return False
    if not valid_formula(target):
        return False
    if not all(valid_formula(formula) for formula in [*precursors, *byproducts]):
        return False
    if not product_side_contains_target(row.get('literature_route_text', ''), target):
        return False
    if not reactants_are_target_relevant(target, precursors):
        return False
    if len(precursors) > 4 or len(byproducts) > 4:
        return False
    text = row.get('literature_route_text', '')
    if any(token in text for token in ['x ', '-x', '+x', '−x', '[', ']']):
        return False
    return bool(source_doi(row.get('source_note')))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def load_existing_targets(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if 'target' not in df:
        return set()
    return {normalize(str(value)) for value in df['target'].dropna()}


def load_global_pet_map(roots: list[Path]) -> dict[str, float]:
    values: dict[str, float] = {}
    for root in roots:
        for cache in root.glob('**/pet_energy_cache.csv'):
            try:
                df = pd.read_csv(cache)
            except Exception:
                continue
            if not {'formula', 'PET_energy_per_formula'}.issubset(df.columns):
                continue
            for _, row in df.iterrows():
                if pd.isna(row.get('PET_energy_per_formula')):
                    continue
                try:
                    values[normalize(row['formula'])] = float(row['PET_energy_per_formula'])
                except Exception:
                    continue
    return values


def cached_systems() -> set[str]:
    root = PROJECT / 'cache/mp'
    return {path.name for path in root.iterdir() if path.is_dir() and (path / 'mp_phases.csv').exists()}


def select_literature_seeded_targets(
    ceder_routes: Path,
    existing_targets: set[str],
    pet_map: dict[str, float],
    max_targets: int,
    max_routes_per_target: int,
) -> tuple[list[str], list[dict[str, str]], dict[str, set[str]], dict[str, set[str]]]:
    systems = cached_systems()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(ceder_routes):
        try:
            target = normalize(row['target'])
        except Exception:
            continue
        if target in existing_targets:
            continue
        if target not in pet_map:
            continue
        if chemical_system(target) not in systems:
            continue
        precursors = split_formulas(row.get('literature_precursors'))
        byproducts = split_formulas(row.get('literature_byproducts'))
        if not is_auditable_route(row, target, precursors, byproducts):
            continue
        selected = dict(row)
        selected['target'] = target
        selected['literature_precursors'] = ';'.join(sorted(set(precursors)))
        selected['literature_byproducts'] = ';'.join(sorted(set(byproducts)))
        selected['source_note'] = selected.get('source_note', '') + '; selected_for_cached_literature_extension'
        grouped[target].append(selected)

    ordered_targets = [target for target, _ in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))]
    selected_targets = ordered_targets[:max_targets]
    selected_rows = []
    lit_precursors: dict[str, set[str]] = defaultdict(set)
    lit_byproducts: dict[str, set[str]] = defaultdict(set)
    for target in selected_targets:
        seen = set()
        kept = 0
        for row in grouped[target]:
            key = (row['literature_precursors'], row['literature_byproducts'])
            if key in seen:
                continue
            seen.add(key)
            selected_rows.append(row)
            lit_precursors[target].update(split_formulas(row.get('literature_precursors')))
            lit_byproducts[target].update(split_formulas(row.get('literature_byproducts')))
            kept += 1
            if kept >= max_routes_per_target:
                break
    return selected_targets, selected_rows, lit_precursors, lit_byproducts


def load_cached_precursor_entries(formulas: set[str]) -> pd.DataFrame:
    path = PROJECT / 'cache/mp_precursors/mp_precursor_phases.csv'
    if not path.exists() or not formulas:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if 'formula' not in df:
        return pd.DataFrame()
    df = df.copy()
    df['formula'] = df['formula'].map(normalize_formula)
    df = df[df['formula'].isin({normalize_formula(formula) for formula in formulas})]
    if df.empty:
        return df
    return df.sort_values(['energy_above_hull', 'formula'], na_position='last').drop_duplicates('formula', keep='first').reset_index(drop=True)


def combine_phase_pool(
    target: str,
    lit_precursors: set[str],
    lit_byproducts: set[str],
    max_phases: int,
    include_common_library: bool = True,
    include_reservoirs: bool = True,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    mp_df = fetch_mp_phases(target, max_phases=max_phases)
    phase_df = build_phase_pool(target, include_local=False, mp_df=mp_df)
    precursor_formulas = common_precursor_formulas(target, include_reservoirs=include_reservoirs) if include_common_library else set()
    precursor_formulas.update(lit_precursors)
    precursor_formulas.update(lit_byproducts)
    if include_reservoirs:
        precursor_formulas.update(GAS_RESERVOIR_FORMULAS)
    frames = [phase_df]
    cached_precursors = load_cached_precursor_entries(precursor_formulas)
    if len(cached_precursors):
        frames.append(cached_precursors)
    reservoirs = reservoir_phase_entries(precursor_formulas)
    if len(reservoirs):
        frames.append(reservoirs)
    existing = set()
    for frame in frames:
        if len(frame) and 'formula' in frame:
            existing.update(frame['formula'].dropna().map(normalize_formula))
    virtual = virtual_common_precursor_entries(precursor_formulas, existing)
    if len(virtual):
        frames.append(virtual)
    phase_df = pd.concat([frame for frame in frames if len(frame)], ignore_index=True)
    phase_df['formula'] = phase_df['formula'].map(normalize_formula)
    source_priority = {
        'materials_project': 1,
        'materials_project_precursor': 2,
        'manual_precursor': 3,
        'virtual_common_precursor': 4,
        'manual_reservoir': 5,
        'local_mp_table': 6,
    }
    phase_df['_source_priority'] = phase_df['source'].map(source_priority).fillna(9)
    phase_df = (
        phase_df.sort_values(['formula', '_source_priority', 'energy_above_hull'], na_position='last')
        .drop_duplicates('formula', keep='first')
        .drop(columns=['_source_priority'])
        .reset_index(drop=True)
    )
    reservoir_byproducts = set(GAS_RESERVOIR_FORMULAS) if include_reservoirs else set()
    return phase_df, precursor_formulas, set(lit_byproducts) | reservoir_byproducts


def write_pet_cache(target_dir: Path, phase_df: pd.DataFrame, pet_map: dict[str, float]) -> Path:
    rows = []
    for _, row in phase_df.iterrows():
        formula = normalize_formula(row['formula'])
        if formula not in pet_map:
            continue
        rows.append(
            {
                'formula': formula,
                'cif_path': row.get('cif_path'),
                'PET_energy_per_formula': pet_map[formula],
                'status': 'reused_global_cache',
            }
        )
    path = target_dir / 'pet_energy_cache.csv'
    pd.DataFrame(rows).sort_values('formula').to_csv(path, index=False)
    return path


def run_target(
    target: str,
    outdir: Path,
    pet_map_global: dict[str, float],
    lit_precursors: set[str],
    lit_byproducts: set[str],
    max_phases: int,
    targeted_max_reactants: int,
    targeted_max_byproducts: int,
    targeted_max_total_coeff: int,
    include_common_library: bool = True,
    include_reservoirs: bool = True,
) -> dict[str, object]:
    target_dir = outdir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    phase_df, precursor_formulas, byproduct_candidates = combine_phase_pool(
        target,
        lit_precursors,
        lit_byproducts,
        max_phases=max_phases,
        include_common_library=include_common_library,
        include_reservoirs=include_reservoirs,
    )
    if phase_df.empty or target not in set(phase_df['formula']):
        raise RuntimeError(f'target phase missing from cached phase pool: {target}')
    raw_phase_count = len(phase_df)
    phase_df.to_csv(target_dir / 'phase_pool_raw.csv', index=False)
    pet_cache = write_pet_cache(target_dir, phase_df, pet_map_global)
    pet_map = pet_energy_map(pet_cache)
    phase_df = compute_phase_stability(phase_df, pet_map)
    phase_df.to_csv(target_dir / 'phase_stability.csv', index=False)
    energy_maps = {}
    mp_form_map = mp_formation_energy_map(phase_df)
    if mp_form_map:
        energy_maps['MP_FORM'] = mp_form_map
    if pet_map:
        energy_maps['PET-OAM-XI'] = pet_map
    formulas = phase_df['formula'].dropna().map(normalize_formula).drop_duplicates().tolist()
    formation = enumerate_formation_reactions(
        target,
        formulas,
        max_reactants=2,
        max_byproducts=1,
        max_total_coeff=80,
        allowed_byproducts=set(GAS_RESERVOIR_FORMULAS),
    )
    targeted = enumerate_targeted_formation_reactions(
        target,
        formulas,
        reactant_candidates=precursor_formulas,
        byproduct_candidates=byproduct_candidates,
        max_reactants=targeted_max_reactants,
        max_byproducts=targeted_max_byproducts,
        max_total_coeff=targeted_max_total_coeff,
    )
    seen = {reaction.reaction for reaction in formation}
    formation.extend(reaction for reaction in targeted if reaction.reaction not in seen)
    decomposition = enumerate_decomposition_reactions(
        target,
        formulas,
        max_products=2,
        max_total_coeff=80,
    )
    formation_df = reactions_to_frame(formation, energy_maps, 'formation', target)
    decomposition_df = reactions_to_frame(decomposition, energy_maps, 'decomposition', target)
    practical_kwargs = {
        'max_atoms': 20,
        'max_stoich_ratio': 5.0,
        'max_ehull': 0.10,
        'max_coeff_sum_per_target': 80.0,
    }
    formation_df = annotate_reaction_practicality(formation_df, phase_df, target, **practical_kwargs)
    decomposition_df = annotate_reaction_practicality(decomposition_df, phase_df, target, **practical_kwargs)
    formation_df.to_csv(target_dir / 'formation_reactions.csv', index=False)
    decomposition_df.to_csv(target_dir / 'decomposition_reactions.csv', index=False)
    write_synthesizability_outputs(target_dir, target, phase_df, formation_df)
    write_report(target_dir, target, phase_df, formation_df, decomposition_df)
    write_recommendation(target_dir, target, phase_df, formation_df, decomposition_df)
    write_json(
        target_dir / 'run_metadata.json',
        {
            'target': target,
            'mode': 'cached_literature_seeded_extension',
            'n_raw_phases': raw_phase_count,
            'n_pet_cache_reused': len(pet_map),
            'n_literature_precursors': len(lit_precursors),
            'n_literature_byproducts': len(lit_byproducts),
            'n_common_or_literature_precursor_formulas': len(precursor_formulas),
            'n_formation_reactions': len(formation_df),
            'n_decomposition_reactions': len(decomposition_df),
            'targeted_max_reactants': targeted_max_reactants,
            'targeted_max_byproducts': targeted_max_byproducts,
            'targeted_max_total_coeff': targeted_max_total_coeff,
            'practical_route_thresholds': practical_kwargs,
        },
    )
    metrics = target_metrics(target_dir, target)
    metrics['status'] = 'ok'
    return metrics


def write_selected_literature(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        'target',
        'literature_precursors',
        'literature_byproducts',
        'literature_route_text',
        'synthesis_type',
        'condition_summary',
        'source_title',
        'source_note',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a cached MP/PET workflow extension on DOI-backed literature-seeded targets.')
    parser.add_argument('--ceder-routes', type=Path, default=DEFAULT_CEDER)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--pet-roots', nargs='*', type=Path, default=DEFAULT_PET_ROOTS)
    parser.add_argument('--exclude-summary', type=Path, default=PROJECT / 'results/mp_pet_benchmark_120/mp_pet_benchmark_summary.csv')
    parser.add_argument('--max-targets', type=int, default=50)
    parser.add_argument('--max-routes-per-target', type=int, default=5)
    parser.add_argument('--mp-max-phases', type=int, default=140)
    parser.add_argument('--targeted-max-reactants', type=int, default=3)
    parser.add_argument('--targeted-max-byproducts', type=int, default=3)
    parser.add_argument('--targeted-max-total-coeff', type=int, default=140)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pet_map = load_global_pet_map(args.pet_roots)
    if not pet_map:
        raise SystemExit('No reusable PET cache entries were found.')
    existing_targets = load_existing_targets(args.exclude_summary)
    targets, literature_rows, lit_precursors, lit_byproducts = select_literature_seeded_targets(
        args.ceder_routes,
        existing_targets,
        pet_map,
        max_targets=args.max_targets,
        max_routes_per_target=args.max_routes_per_target,
    )
    write_selected_literature(args.output_dir / 'selected_literature_routes.csv', literature_rows)
    (args.output_dir / 'selected_targets.txt').write_text('\n'.join(targets) + '\n', encoding='utf-8')
    rows = []
    pooled_rows = []
    for index, target in enumerate(targets, start=1):
        target_dir = args.output_dir / target
        done = target_dir / 'synthesizability_ranked_routes.csv'
        if done.exists() and not args.force:
            print(f'[{index}/{len(targets)}] skip {target}')
            metrics = target_metrics(target_dir, target)
            metrics['status'] = 'ok'
        else:
            print(f'[{index}/{len(targets)}] run {target} routes={sum(1 for row in literature_rows if row["target"] == target)}')
            try:
                metrics = run_target(
                    target,
                    args.output_dir,
                    pet_map,
                    lit_precursors.get(target, set()),
                    lit_byproducts.get(target, set()),
                    max_phases=args.mp_max_phases,
                    targeted_max_reactants=args.targeted_max_reactants,
                    targeted_max_byproducts=args.targeted_max_byproducts,
                    targeted_max_total_coeff=args.targeted_max_total_coeff,
                )
            except Exception as exc:
                metrics = {col: None for col in METRIC_COLUMNS}
                metrics['target'] = target
                metrics['status'] = 'failed'
                metrics['error'] = str(exc)
                print(f'  failed: {exc}')
        rows.append(metrics)
        formation_path = target_dir / 'formation_reactions.csv'
        if formation_path.exists() and formation_path.stat().st_size > 0:
            df = pd.read_csv(formation_path)
            form_col = 'DeltaE_MP_FORM_per_target' if 'DeltaE_MP_FORM_per_target' in df else None
            pet_col = 'DeltaE_PET-OAM-XI_per_target' if 'DeltaE_PET-OAM-XI_per_target' in df else None
            if form_col and pet_col:
                sub = df[df[form_col].notna() & df[pet_col].notna()].copy()
                sub['target'] = target
                pooled_rows.append(sub[['target', 'reaction', form_col, pet_col]])
    summary = pd.DataFrame(rows)
    for col in METRIC_COLUMNS:
        if col not in summary:
            summary[col] = None
    summary = summary[METRIC_COLUMNS]
    summary.to_csv(args.output_dir / 'mp_pet_benchmark_summary.csv', index=False)
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    pooled.to_csv(args.output_dir / 'mp_pet_reaction_data.csv', index=False)
    if len(summary):
        plot_summary(summary, pooled, args.output_dir)
        add_category_summaries(summary, args.output_dir)
    counts = Counter(row['target'] for row in literature_rows)
    lines = [
        '# Literature-seeded cached workflow extension',
        '',
        f'- Selected targets: {len(targets)}',
        f'- Selected DOI-backed literature routes: {len(literature_rows)}',
        f'- Reused PET cache entries: {len(pet_map)}',
        f'- Completed targets: {int((summary["status"] == "ok").sum()) if len(summary) else 0}',
    ]
    valid = summary[summary['n_reactions_with_dft_pet'].fillna(0) > 0] if len(summary) else pd.DataFrame()
    if len(valid):
        lines.extend(
            [
                f'- Comparable PET/MP-form reactions: {int(valid["n_reactions_with_dft_pet"].sum())}',
                f'- Mean PET vs MP-form reaction MAE: {valid["mae_pet_vs_mp_form"].mean():.4f} eV/target',
                f'- Median PET vs MP-form reaction MAE: {valid["mae_pet_vs_mp_form"].median():.4f} eV/target',
                f'- Mean sign consistency: {valid["sign_consistency"].mean():.3f}',
                f'- Top-1 match rate: {valid["top1_match"].mean():.3f}',
            ]
        )
    lines.append('')
    lines.append('## Literature route counts by target')
    lines.extend(f'- {target}: {count}' for target, count in counts.most_common())
    (args.output_dir / 'CACHED_LITERATURE_EXTENSION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
