from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workflow/multi_route_benchmark/temporal_validation_2022'))
from evaluate_2022_temporal_benchmark import (
    CorpusPrior, prepare_features, rank_routes, normalize, split_formulas,
    route_similarity, class_multiset,
)
from evaluate_route_similarity import parse_side, nonvolatile_source_elements, recall, jaccard, counter_jaccard

OUT = ROOT / 'results'
summary = pd.read_csv(OUT / 'summary.csv')
expected = {'old_production_energy_all': [(1, 2), (6, 8), (7, 11), (10, 23)], 'old_production_energy_practical': [(6, 8), (8, 17), (11, 25), (20, 53)], 'old_synth_score': [(40, 57), (80, 103), (89, 120), (97, 127)], 'old_diverse_top20': [(40, 57), (83, 114), (95, 125), (102, 128)]}
checks = []
for mode, values in expected.items():
    for top_k, counts in zip([1,5,10,20], values):
        row = summary[(summary['mode'] == mode) & (summary.cohort == 'all144') & (summary.top_k == top_k)].iloc[0]
        actual = (int(row.strong_or_exact), int(row.moderate_or_better))
        assert actual == counts, (mode, top_k, actual, counts)
        checks.append(dict(mode=mode, top_k=top_k, expected=counts, actual=actual, passed=True))
(OUT / 'baseline_verification.json').write_text(json.dumps(checks, indent=2))
refs = pd.read_csv(OUT / 'references_and_prior_overlap.csv').fillna('')
prior_frame = pd.read_csv(ROOT / 'inputs/split/historical_prior_routes_through_2020.csv', low_memory=False).fillna('')
excluded = set()
for filename in ['development_routes_2021_2022.csv','future_routes_2023_2024.csv']:
    excluded.update(normalize(value) for value in pd.read_csv(ROOT / 'inputs/split' / filename)['target'])
prior = CorpusPrior.fit(prior_frame, excluded)
cases = {}
details = []
class_detail = []
for reference in refs.to_dict('records'):
    target, cohort = reference['target'], reference['cohort']
    key = (cohort, target)
    if key not in cases:
        folder = 'seed80_targets' if cohort == 'initial59' else 'extension53_targets'
        routes = pd.read_csv(ROOT / 'inputs/literature' / folder / target / 'synthesizability_ranked_routes.csv').fillna('').to_dict('records')
        prepare_features({target: routes}, prior)
        cases[key] = routes
    routes = cases[key]
    precursors = split_formulas(reference['literature_precursors'])
    products = split_formulas(reference['literature_byproducts'])
    modes = {'new_base': (0,0), 'new_corpus': (.3,0), 'new_corpus_mmr': (.3,.1)}
    for mode, (alpha, penalty) in modes.items():
        selected = rank_routes(target, routes, 'condition_heavy', alpha, penalty, 5)
        matches = []
        for route in selected:
            original = route_similarity(target, precursors, products, route)
            candidate_precursors = parse_side(route.get('reactants'))
            candidate_products = parse_side(route.get('products')) - {normalize(target)}
            reference_elements = nonvolatile_source_elements(precursors, target)
            candidate_elements = nonvolatile_source_elements(candidate_precursors, target)
            score = (.28*recall(precursors,candidate_precursors)+.18*jaccard(precursors,candidate_precursors)
                +.20*recall(reference_elements,candidate_elements)
                +.16*counter_jaccard(class_multiset(precursors,target),class_multiset(candidate_precursors,target))
                +.08*recall(products,candidate_products)
                +.05*counter_jaccard(class_multiset(products,target),class_multiset(candidate_products,target))
                +.05*jaccard(reference_elements,candidate_elements))
            complete = original['exact_reactant_recall'] >= .999 and original['byproduct_recall'] >= .999
            label = 'complete' if complete else 'strong' if score >= .78 else 'moderate' if score >= .62 else 'weak'
            matches.append((round(score,6), label))
        best = max(enumerate(matches), key=lambda item:(item[1][0],-item[0]))[1]
        details.append(dict(record_id=reference['record_id'],mode=mode,score=best[0],
                            strong_or_complete=best[1] in ['complete','strong'],moderate_or_better=best[1] != 'weak'))
        normal = max((route_similarity(target,precursors,products,route) for route in selected),key=lambda match:match['route_similarity_score'])
        for class_name in class_multiset(precursors,target):
            full_match = any(route_similarity(target,precursors,products,route)['route_similarity_label'] in
                             ['exact_route_recovered','strong_route_family_match','moderate_route_family_match'] for route in routes)
            class_detail.append(dict(record_id=reference['record_id'],mode=mode,precursor_class=class_name,
                full_covered=full_match,top5_covered=normal['route_similarity_label'] in
                ['exact_route_recovered','strong_route_family_match','moderate_route_family_match']))
zero = pd.DataFrame(details)
zero.to_csv(OUT / 'zero_matching_bonus_record_results.csv',index=False)
zero.groupby('mode')[['strong_or_complete','moderate_or_better']].sum().to_csv(OUT / 'zero_matching_bonus_summary.csv')
classes = pd.DataFrame(class_detail)
classes.to_csv(OUT / 'precursor_class_record_results.csv',index=False)
class_summary = classes.groupby(['mode','precursor_class']).agg(n_records=('record_id','count'),full_covered=('full_covered','sum'),top5_covered=('top5_covered','sum')).reset_index()
class_summary['gap_percentage_points'] = 100*(class_summary.full_covered-class_summary.top5_covered)/class_summary.n_records
class_summary.to_csv(OUT / 'precursor_class_summary.csv',index=False)
candidate_path = OUT / 'new_policy_top20_candidates.csv'
candidates = pd.read_csv(candidate_path,low_memory=False)
candidates.rename(columns={'synth_score':'archived_synth_score','synth_rank':'archived_synth_rank'},inplace=True)
candidates.to_csv(candidate_path,index=False)
print('16 archived baseline checks passed.')
print(zero.groupby('mode')[['strong_or_complete','moderate_or_better']].sum().to_string())
print(class_summary[class_summary['mode']=='new_base'].to_string(index=False))
