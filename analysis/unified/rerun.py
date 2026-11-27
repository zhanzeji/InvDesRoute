from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
CODE = ROOT / 'workflow'
if not INPUT.exists():
    raise FileNotFoundError('Use scripts/reproduce_full.py with the separately held input archive.')
sys.path.insert(0, str(CODE / 'multi_route_benchmark/temporal_validation_2022'))
from evaluate_2022_temporal_benchmark import (
    CorpusPrior, COMPONENTS, PROFILES, prepare_features, rank_routes,
    normalize, parse_side, split_formulas, route_similarity,
)
from evaluate_recommendation_ablation import load_ranked_routes

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
references = []
cases = {}
case_paths = {}
for cohort, filename, folder in [
    ('initial59', 'ceder_seed80_real_literature_routes.csv', 'seed80_targets'),
    ('extension86', 'selected_literature_routes.csv', 'extension53_targets'),
]:
    frame = pd.read_csv(INPUT / 'literature/references' / filename).fillna('')
    for index, reference in frame.iterrows():
        if cohort == 'extension86' and index == 77:
            continue
        reference = reference.to_dict()
        target = normalize(reference['target'])
        key = (cohort, target)
        reference.update(cohort=cohort, target=target, record_id=f'{cohort}_{index+1:03d}')
        match = re.search(r'DOI=([^;\s]+)', reference['source_note'])
        reference['doi'] = match.group(1).lower() if match else ''
        references.append(reference)
        if key not in cases:
            case_paths[key] = INPUT / 'literature' / folder / target
            routes = pd.read_csv(case_paths[key] / 'synthesizability_ranked_routes.csv').fillna('').to_dict('records')
            for route_index, route in enumerate(routes):
                route['_audit_id'] = route_index
            cases[key] = routes
assert len(references) == 144
prior_frame = pd.read_csv(INPUT / 'split/historical_prior_routes_through_2020.csv', low_memory=False).fillna('')
development = pd.read_csv(INPUT / 'split/development_routes_2021_2022.csv').fillna('')
future = pd.read_csv(INPUT / 'split/future_routes_2023_2024.csv').fillna('')
excluded = {normalize(value) for value in pd.concat([development, future])['target']}
audit_targets = {reference['target'] for reference in references}
reference_dois = {reference['doi'] for reference in references if reference['doi']}
prior_dois = set(prior_frame['doi'].str.lower())
for reference in references:
    reference['doi_in_historical_prior'] = reference['doi'] in prior_dois
pd.DataFrame(references).to_csv(OUT / 'references_and_prior_overlap.csv', index=False)
all_detail = []
candidate_rows = []
full_detail = []
for prior_mode in ['frozen_original_prior', 'exclude_audit_targets_and_dois']:
    if prior_mode == 'frozen_original_prior':
        fitting_frame = prior_frame
        excluded_targets = excluded
    else:
        fitting_frame = prior_frame[~prior_frame['doi'].str.lower().isin(reference_dois)]
        excluded_targets = excluded | audit_targets
    prior = CorpusPrior.fit(fitting_frame, excluded_targets)
    (OUT / f'{prior_mode}.json').write_text(json.dumps(prior.serializable(excluded_targets, 'historical_prior_routes_through_2020.csv'), indent=2))
    ranking = {}
    for key, routes in cases.items():
        cohort, target = key
        prepare_features({target: routes}, prior)
        modes = {
            'new_base': (0.0, 0.0),
            'new_corpus': (0.30, 0.0),
            'new_corpus_mmr': (0.30, 0.10),
        }
        for mode, (alpha, penalty) in modes.items():
            ranking[(key, mode)] = rank_routes(target, routes, 'condition_heavy', alpha, penalty, 20)
            if mode == 'new_corpus_mmr':
                for rank, route in enumerate(ranking[(key, mode)], 1):
                    candidate_rows.append({
                        'prior_mode': prior_mode, 'cohort': cohort, 'target': target, 'rank': rank,
                        **{('archived_' + name if name in ['synth_score', 'synth_rank'] else name): value
                           for name, value in route.items() if name != 'profile_scores'},
                    })
        if prior_mode == 'frozen_original_prior':
            for mode in ['production_energy_all', 'production_energy_practical', 'synth_score', 'diverse_top20']:
                ranking[(key, 'old_' + mode)] = load_ranked_routes(case_paths[key], mode)[:20]
    print(f'{prior_mode}: ranked {len(cases)} target pools', flush=True)
    for reference_index, reference in enumerate(references):
        target = reference['target']
        key = (reference['cohort'], target)
        lit_precursors = split_formulas(reference['literature_precursors'])
        lit_products = split_formulas(reference['literature_byproducts'])
        cache = {}
        def match_route(route):
            reaction = str(route.get('reaction', ''))
            if reaction not in cache:
                similarity = route_similarity(target, lit_precursors, lit_products, route)
                strict = parse_side(route.get('reactants')) == lit_precursors and (
                    parse_side(route.get('products')) - {target}) == lit_products
                cache[reaction] = dict(similarity, strict_exact=strict)
            return cache[reaction]
        if prior_mode == 'frozen_original_prior':
            pool_matches = [match_route(route) for route in cases[key]]
            full_detail.append({
                'record_id': reference['record_id'], 'target': target,
                'strict_exact': any(match['strict_exact'] for match in pool_matches),
                'complete_recall': any(match['route_similarity_label'] == 'exact_route_recovered' for match in pool_matches),
                'strong_or_exact': any(match['route_similarity_label'] in ['exact_route_recovered', 'strong_route_family_match'] for match in pool_matches),
                'moderate_or_better': any(match['route_similarity_label'] in ['exact_route_recovered', 'strong_route_family_match', 'moderate_route_family_match'] for match in pool_matches),
            })
        for (ranking_key, mode), selected in ranking.items():
            if ranking_key != key:
                continue
            for top_k in [1, 5, 10, 20]:
                matches = [dict(match_route(route), displayed_rank=rank, reaction=route.get('reaction')) for rank, route in enumerate(selected[:top_k], 1)]
                best = max(matches, key=lambda match: (match['route_similarity_score'], -match['displayed_rank']))
                all_detail.append({
                    'prior_mode': prior_mode, 'mode': mode, 'top_k': top_k,
                    'record_id': reference['record_id'], 'cohort': reference['cohort'],
                    'target': target, 'doi': reference['doi'],
                    'strict_exact': any(match['strict_exact'] for match in matches),
                    'complete_recall': best['route_similarity_label'] == 'exact_route_recovered',
                    'strong_or_exact': best['route_similarity_label'] in ['exact_route_recovered', 'strong_route_family_match'],
                    'moderate_or_better': best['route_similarity_label'] in ['exact_route_recovered', 'strong_route_family_match', 'moderate_route_family_match'],
                    'mean_precursor_recall_candidate_best': max(match['exact_reactant_recall'] for match in matches),
                    **best,
                })
        if (reference_index + 1) % 25 == 0:
            print(f'{prior_mode}: evaluated {reference_index+1}/144 records', flush=True)
detail = pd.DataFrame(all_detail)
detail.to_csv(OUT / 'record_level_results.csv', index=False)
pd.DataFrame(candidate_rows).to_csv(OUT / 'new_policy_top20_candidates.csv', index=False)
pd.DataFrame(full_detail).to_csv(OUT / 'full_network_coverage.csv', index=False)
summaries = []
for grouping, frame in detail.groupby(['prior_mode', 'mode', 'top_k'], sort=False):
    prior_mode, mode, top_k = grouping
    for cohort in ['all144', 'initial59', 'extension86']:
        subset = frame if cohort == 'all144' else frame[frame['cohort'] == cohort]
        summaries.append({
            'prior_mode': prior_mode, 'mode': mode, 'top_k': top_k, 'cohort': cohort,
            'n_records': len(subset),
            **{metric: int(subset[metric].sum()) for metric in ['strict_exact', 'complete_recall', 'strong_or_exact', 'moderate_or_better']},
            'mean_similarity': float(subset['route_similarity_score'].mean()),
            'mean_precursor_recall': float(subset['mean_precursor_recall_candidate_best'].mean()),
        })
summary = pd.DataFrame(summaries)
summary.to_csv(OUT / 'summary.csv', index=False)
config = dict(weights=dict(zip(COMPONENTS, PROFILES['condition_heavy'])), alpha=0.30,
              diversity_penalty=0.10, parameter_selection='2021-2022 development only; no fitting on audit145',
              n_records=len(references), n_target_pools=len(cases),
              n_unique_target_formulas=len(audit_targets),
              n_candidates=sum(map(len, cases.values())),
              n_reference_records_with_doi_in_prior=sum(reference['doi_in_historical_prior'] for reference in references),
              full_network={metric: sum(row[metric] for row in full_detail) for metric in ['strict_exact','complete_recall','strong_or_exact','moderate_or_better']})
(OUT / 'configuration.json').write_text(json.dumps(config, indent=2))
checksums = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for directory in [INPUT, CODE] for path in sorted(directory.rglob('*')) if path.is_file() and '__pycache__' not in str(path)}
(OUT / 'input_sha256.json').write_text(json.dumps(checksums, indent=2))
print(summary[(summary.cohort == 'all144') & (summary.top_k == 5)].to_string(index=False), flush=True)
print(json.dumps(config, indent=2), flush=True)
