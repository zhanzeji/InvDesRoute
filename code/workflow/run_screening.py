#!/home/zhzhan/miniconda3/envs/mattersim/bin/python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make src importable when launched directly
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import pandas as pd

from analysis import annotate_reaction_practicality, compute_phase_stability, mp_formation_energy_map, pet_energy_map, write_recommendation, write_report
from common import PROJECT, UPET_PYTHON, chemical_system, formula_from_cif, normalize_formula, write_json
from pet_energy import predict_pet_for_phase_table
from phase_sources import build_phase_pool, fetch_mp_phases
from precursor_library import (
    GAS_RESERVOIR_FORMULAS,
    common_precursor_formulas,
    fetch_mp_precursor_formulas,
    load_manual_precursor_table,
    reservoir_phase_entries,
    virtual_common_precursor_entries,
)
from reactions import enumerate_decomposition_reactions, enumerate_formation_reactions, reactions_to_frame
from synthesizability import write_synthesizability_outputs
from targeted_routes import enumerate_targeted_formation_reactions


def main() -> None:
    parser = argparse.ArgumentParser(description='Screen thermodynamic synthesis pathways for a target formula or CIF.')
    parser.add_argument('--target-formula', default=None)
    parser.add_argument('--target-cif', default=None)
    parser.add_argument('--extra-cif-dir', action='append', default=[], help='Directory containing additional CIF phases to include in the phase pool. Can be used multiple times.')
    parser.add_argument('--extra-cif-recursive', action='store_true', help='Recursively scan --extra-cif-dir for CIF files.')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--database', choices=['local', 'mp', 'local+mp'], default='local')
    parser.add_argument('--mp-api-key', default=None)
    parser.add_argument('--mp-max-phases', type=int, default=120)
    parser.add_argument('--mp-force-refresh', action='store_true')
    parser.add_argument('--include-common-precursors', action='store_true')
    parser.add_argument('--include-gas-reservoirs', action='store_true')
    parser.add_argument('--manual-precursor-table', default=None)
    parser.add_argument('--precursor-force-refresh', action='store_true')
    parser.add_argument('--include-virtual-common-precursors', action='store_true')
    parser.add_argument('--compute-pet', action='store_true')
    parser.add_argument('--pet-device', default='cuda')
    parser.add_argument('--pet-python', default=str(UPET_PYTHON))
    parser.add_argument('--pet-max-phases', type=int, default=None)
    parser.add_argument('--max-reactants', type=int, default=3)
    parser.add_argument('--max-byproducts', type=int, default=2)
    parser.add_argument('--max-products', type=int, default=3)
    parser.add_argument('--max-total-coeff', type=int, default=80)
    parser.add_argument('--byproducts-reservoir-only', action='store_true')
    parser.add_argument('--include-targeted-precursor-routes', action='store_true')
    parser.add_argument('--targeted-max-reactants', type=int, default=3)
    parser.add_argument('--targeted-max-byproducts', type=int, default=4)
    parser.add_argument('--targeted-max-total-coeff', type=int, default=120)
    parser.add_argument('--practical-max-atoms', type=int, default=10)
    parser.add_argument('--practical-max-stoich-ratio', type=float, default=5.0)
    parser.add_argument('--practical-max-ehull', type=float, default=0.10)
    parser.add_argument('--practical-max-coeff-sum-per-target', type=float, default=30.0)
    parser.add_argument('--recommendation-exergonic-threshold', type=float, default=-0.05)
    parser.add_argument('--recommendation-ehull-threshold', type=float, default=0.10)
    args = parser.parse_args()

    if not args.target_formula and not args.target_cif:
        raise SystemExit('Provide --target-formula or --target-cif.')
    target = normalize_formula(args.target_formula) if args.target_formula else normalize_formula(formula_from_cif(args.target_cif))
    outdir = Path(args.output_dir or PROJECT / 'results' / f'{target}_{args.database.replace("+", "plus")}')
    outdir.mkdir(parents=True, exist_ok=True)

    mp_df = None
    if args.database in {'mp', 'local+mp'}:
        mp_df = fetch_mp_phases(
            target,
            api_key=args.mp_api_key,
            max_phases=args.mp_max_phases,
            force_refresh=args.mp_force_refresh,
        )
    phase_df = build_phase_pool(target, include_local=args.database in {'local', 'local+mp'}, mp_df=mp_df, external_cif=args.target_cif, extra_cif_dirs=args.extra_cif_dir, extra_cif_recursive=args.extra_cif_recursive)
    precursor_frames = []
    precursor_formulas = set()
    if args.include_common_precursors:
        precursor_formulas.update(common_precursor_formulas(target, include_reservoirs=args.include_gas_reservoirs))
        if args.database in {'mp', 'local+mp'}:
            precursor_frames.append(
                fetch_mp_precursor_formulas(
                    precursor_formulas,
                    api_key=args.mp_api_key,
                    force_refresh=args.precursor_force_refresh,
                )
            )
    if args.include_gas_reservoirs:
        precursor_frames.append(reservoir_phase_entries(precursor_formulas or None))
    if args.include_virtual_common_precursors and precursor_formulas:
        existing_formulas = set(phase_df['formula'].dropna().map(normalize_formula)) if len(phase_df) else set()
        for frame in precursor_frames:
            if frame is not None and len(frame) and 'formula' in frame:
                existing_formulas.update(frame['formula'].dropna().map(normalize_formula))
        precursor_frames.append(virtual_common_precursor_entries(precursor_formulas, existing_formulas))
    manual_precursors = load_manual_precursor_table(args.manual_precursor_table)
    if len(manual_precursors):
        precursor_frames.append(manual_precursors)
        precursor_formulas.update(manual_precursors['formula'].dropna().map(normalize_formula))
    if precursor_frames:
        non_empty = [frame for frame in precursor_frames if frame is not None and len(frame)]
        if non_empty:
            phase_df = pd.concat([phase_df, *non_empty], ignore_index=True)
            phase_df['formula'] = phase_df['formula'].map(normalize_formula)
            source_priority = {
                'external_cif': 0,
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
    if phase_df.empty:
        raise SystemExit(f'No phases found for chemical system {chemical_system(target)}. Try --database mp with an MP API key.')
    phase_df.to_csv(outdir / 'phase_pool_raw.csv', index=False)

    pet_cache = outdir / 'pet_energy_cache.csv'
    if args.compute_pet:
        predict_pet_for_phase_table(
            phase_df,
            pet_cache,
            device=args.pet_device,
            max_phases=args.pet_max_phases,
            python_executable=args.pet_python,
        )
    pet_map = pet_energy_map(pet_cache)
    phase_df = compute_phase_stability(phase_df, pet_map)
    phase_df.to_csv(outdir / 'phase_stability.csv', index=False)

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
        max_reactants=args.max_reactants,
        max_byproducts=args.max_byproducts,
        max_total_coeff=args.max_total_coeff,
        allowed_byproducts=set(GAS_RESERVOIR_FORMULAS) if args.byproducts_reservoir_only else None,
    )
    if args.include_targeted_precursor_routes and precursor_formulas:
        targeted = enumerate_targeted_formation_reactions(
            target,
            formulas,
            reactant_candidates=precursor_formulas,
            byproduct_candidates=set(GAS_RESERVOIR_FORMULAS),
            max_reactants=args.targeted_max_reactants,
            max_byproducts=args.targeted_max_byproducts,
            max_total_coeff=args.targeted_max_total_coeff,
        )
        seen_reactions = {reaction.reaction for reaction in formation}
        formation.extend(reaction for reaction in targeted if reaction.reaction not in seen_reactions)
    decomposition = enumerate_decomposition_reactions(
        target,
        formulas,
        max_products=args.max_products,
        max_total_coeff=args.max_total_coeff,
    )
    formation_df = reactions_to_frame(formation, energy_maps, 'formation', target)
    decomposition_df = reactions_to_frame(decomposition, energy_maps, 'decomposition', target)
    practical_kwargs = {
        'max_atoms': args.practical_max_atoms,
        'max_stoich_ratio': args.practical_max_stoich_ratio,
        'max_ehull': args.practical_max_ehull,
        'max_coeff_sum_per_target': args.practical_max_coeff_sum_per_target,
    }
    formation_df = annotate_reaction_practicality(formation_df, phase_df, target, **practical_kwargs)
    decomposition_df = annotate_reaction_practicality(decomposition_df, phase_df, target, **practical_kwargs)
    formation_df.to_csv(outdir / 'formation_reactions.csv', index=False)
    decomposition_df.to_csv(outdir / 'decomposition_reactions.csv', index=False)
    write_synthesizability_outputs(outdir, target, phase_df, formation_df)
    write_report(outdir, target, phase_df, formation_df, decomposition_df)
    write_recommendation(
        outdir,
        target,
        phase_df,
        formation_df,
        decomposition_df,
        exergonic_threshold=args.recommendation_exergonic_threshold,
        ehull_threshold=args.recommendation_ehull_threshold,
    )
    write_json(
        outdir / 'run_metadata.json',
        {
            'target': target,
            'target_cif': str(Path(args.target_cif).resolve()) if args.target_cif else None,
            'extra_cif_dir': [str(Path(p).resolve()) for p in args.extra_cif_dir],
            'extra_cif_recursive': args.extra_cif_recursive,
            'database': args.database,
            'include_common_precursors': args.include_common_precursors,
            'include_gas_reservoirs': args.include_gas_reservoirs,
            'manual_precursor_table': args.manual_precursor_table,
            'include_virtual_common_precursors': args.include_virtual_common_precursors,
            'n_common_precursor_formulas': len(precursor_formulas),
            'byproducts_reservoir_only': args.byproducts_reservoir_only,
            'include_targeted_precursor_routes': args.include_targeted_precursor_routes,
            'targeted_max_reactants': args.targeted_max_reactants,
            'targeted_max_byproducts': args.targeted_max_byproducts,
            'targeted_max_total_coeff': args.targeted_max_total_coeff,
            'n_phases': int(len(phase_df)),
            'n_formation_reactions': int(len(formation_df)),
            'n_decomposition_reactions': int(len(decomposition_df)),
            'pet_cache': str(pet_cache) if pet_cache.exists() else None,
            'pet_python': args.pet_python if args.compute_pet else None,
            'practical_route_thresholds': practical_kwargs,
            'recommendation_exergonic_threshold': args.recommendation_exergonic_threshold,
            'recommendation_ehull_threshold': args.recommendation_ehull_threshold,
        },
    )
    print(f'Target: {target}')
    print(f'Output: {outdir}')
    print(f'Phases: {len(phase_df)}')
    print(f'Formation reactions: {len(formation_df)}')
    print(f'Decomposition reactions: {len(decomposition_df)}')


if __name__ == '__main__':
    main()
