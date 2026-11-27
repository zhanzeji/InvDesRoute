from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workflow/multi_route_benchmark/temporal_validation_2022'))
from evaluate_2022_temporal_benchmark import normalize, route_similarity, split_formulas
from evaluate_route_similarity import (
    parse_side, nonvolatile_source_elements, recall, jaccard, class_multiset, counter_jaccard,
)
from evaluate_recommendation_ablation import load_ranked_routes

OUT = ROOT / 'results/S5'
OUT.mkdir(exist_ok=True)
THRESHOLDS = [.55,.58,.60,.62,.65,.68,.70,.75,.78,.80]
references = pd.read_csv(ROOT / 'results/references_and_prior_overlap.csv').fillna('')
detail = pd.read_csv(ROOT / 'results/record_level_results.csv').fillna('')
top5 = detail[(detail.prior_mode=='frozen_original_prior') & (detail['mode']=='new_base') & (detail.top_k==5)].set_index('record_id')

def scores(target, precursors, byproducts, route):
    original = route_similarity(target,precursors,byproducts,route)
    reactants = parse_side(route.get('reactants'))
    products = parse_side(route.get('products')) - {normalize(target)}
    reference_elements = nonvolatile_source_elements(precursors,target)
    candidate_elements = nonvolatile_source_elements(reactants,target)
    zero = (.28*recall(precursors,reactants)+.18*jaccard(precursors,reactants)
        +.20*recall(reference_elements,candidate_elements)
        +.16*counter_jaccard(class_multiset(precursors,target),class_multiset(reactants,target))
        +.08*recall(byproducts,products)
        +.05*counter_jaccard(class_multiset(byproducts,target),class_multiset(products,target))
        +.05*jaccard(reference_elements,candidate_elements))
    complete = original['exact_reactant_recall'] >= .999 and original['byproduct_recall'] >= .999
    return float(original['route_similarity_score']),round(float(np.clip(zero,0,1)),6),complete

full_maxima = []
pool_cache = {}
for position, reference in references.iterrows():
    target = reference['target']
    folder = 'seed80_targets' if reference['cohort']=='initial59' else 'extension53_targets'
    key = (folder,target)
    if key not in pool_cache:
        pool_cache[key] = pd.read_csv(ROOT / 'inputs/literature' / folder / target / 'synthesizability_ranked_routes.csv').fillna('').to_dict('records')
    precursors = split_formulas(reference['literature_precursors'])
    byproducts = split_formulas(reference['literature_byproducts'])
    values = [scores(target,precursors,byproducts,route) for route in pool_cache[key]]
    full_maxima.append(dict(record_id=reference['record_id'],full_original=max(value[0] for value in values),full_zero=max(value[1] for value in values)))
    if (position+1)%25==0:
        print(f'Full network: {position+1}/144',flush=True)
full = pd.DataFrame(full_maxima)
full.to_csv(OUT / 'full_network_max_similarity.csv',index=False)
zero_detail = pd.read_csv(ROOT / 'results/zero_matching_bonus_record_results.csv').set_index(['record_id','mode'])
top_scores = [float(top5.loc[record_id,'route_similarity_score']) for record_id in references.record_id]
zero_scores = [float(zero_detail.loc[(record_id,'new_base'),'score']) for record_id in references.record_id]
threshold_rows = []
for threshold in THRESHOLDS:
    threshold_rows.append(dict(threshold=threshold,records=144,
        full_original=int((full.full_original>=threshold).sum()),full_zero=int((full.full_zero>=threshold).sum()),
        top5_base_original=int((np.asarray(top_scores)>=threshold).sum()),top5_base_zero=int((np.asarray(zero_scores)>=threshold).sum())))

donors = []
for reference in references.to_dict('records'):
    selected = top5.loc[reference['record_id']]
    donors.append(dict(record_id=reference['record_id'],target=reference['target'],
        reactants=split_formulas(selected['recommended_reactants']),
        byproducts=split_formulas(selected['recommended_byproducts']),
        reaction=selected['recommended_reaction'],template=selected['recommended_template']))
pd.DataFrame([{name:(';'.join(sorted(value)) if isinstance(value,set) else value) for name,value in donor.items()} for donor in donors]).to_csv(OUT / 'decoy_donor_records.csv',index=False)
pair_scores = np.zeros((144,144,2))
pair_rows = []
for reference_index, reference in enumerate(references.to_dict('records')):
    target = reference['target']
    precursors = split_formulas(reference['literature_precursors'])
    byproducts = split_formulas(reference['literature_byproducts'])
    for donor_index, donor in enumerate(donors):
        if donor['target']==target:
            continue
        route = dict(reactants=' | '.join(f'1*{formula}' for formula in sorted(donor['reactants'])),
                     products=' | '.join(f'1*{formula}' for formula in sorted({target,*donor['byproducts']})),
                     reaction=donor['reaction'],route_template_reason=donor['template'])
        original,zero,_ = scores(target,precursors,byproducts,route)
        pair_scores[reference_index,donor_index] = [original,zero]
        pair_rows.append(dict(reference_id=reference['record_id'],donor_id=donor['record_id'],original_score=original,zero_bonus_score=zero))
pd.DataFrame(pair_rows).to_csv(OUT / 'decoy_pair_scores.csv',index=False)
print('Decoy scores precomputed; sampling 1000 paired draws.',flush=True)
rng = np.random.default_rng(20260722)
iterations = []
sample_indices = np.empty((1000,144,5),dtype=np.int16)
eligible = [np.asarray([index for index,donor in enumerate(donors) if donor['target']!=reference['target']]) for reference in references.to_dict('records')]
for iteration in range(1000):
    maxima = np.empty((144,2))
    for reference_index in range(144):
        selected = rng.choice(len(eligible[reference_index]),size=5,replace=False)
        donor_indices = eligible[reference_index][selected]
        sample_indices[iteration,reference_index] = donor_indices
        maxima[reference_index] = pair_scores[reference_index,donor_indices].max(axis=0)
    for threshold in THRESHOLDS:
        rates = (maxima>=threshold).mean(axis=0)
        iterations.append(dict(iteration=iteration,threshold=threshold,original_false_positive_rate=rates[0],zero_false_positive_rate=rates[1]))
np.savez_compressed(OUT / 'decoy_draw_indices.npz',donor_indices=sample_indices,record_ids=references.record_id.to_numpy(dtype=str))
iteration_frame = pd.DataFrame(iterations)
iteration_frame.to_csv(OUT / 'decoy_iterations.csv',index=False)
decoy_summary = []
for threshold in THRESHOLDS:
    subset = iteration_frame[iteration_frame.threshold==threshold]
    row = dict(threshold=threshold)
    for setting,column in [('original','original_false_positive_rate'),('zero','zero_false_positive_rate')]:
        row.update({setting+'_mean':float(subset[column].mean()),setting+'_low':float(subset[column].quantile(.025)),setting+'_high':float(subset[column].quantile(.975))})
    decoy_summary.append(row)
table = pd.DataFrame(threshold_rows).merge(pd.DataFrame(decoy_summary),on='threshold')
table.to_csv(OUT / 'S10_threshold_and_decoy.csv',index=False)

base_summary = pd.read_csv(ROOT / 'results/summary.csv')
zero_summary = pd.read_csv(ROOT / 'results/zero_matching_bonus_summary.csv').set_index('mode')
table11 = []
for label,mode in [('Energy only','old_production_energy_all'),('Practical first, then energy','old_production_energy_practical'),
                    ('Frozen base score','new_base'),('Base + historical prior','new_corpus'),('Base + historical prior + MMR','new_corpus_mmr')]:
    original = base_summary[(base_summary.prior_mode=='frozen_original_prior') & (base_summary['mode']==mode) & (base_summary.cohort=='all144') & (base_summary.top_k==5)].iloc[0]
    if mode.startswith('old_'):
        matches = []
        for reference in references.to_dict('records'):
            target = reference['target']
            folder = 'seed80_targets' if reference['cohort']=='initial59' else 'extension53_targets'
            routes = load_ranked_routes(ROOT / 'inputs/literature' / folder / target,mode.removeprefix('old_'))[:5]
            values = [scores(target,split_formulas(reference['literature_precursors']),split_formulas(reference['literature_byproducts']),route) for route in routes]
            best = max(enumerate(values),key=lambda item:(item[1][1],-item[0]))[1]
            matches.append((best[2] or best[1]>=.78,best[2] or best[1]>=.62))
        strong_zero = sum(value[0] for value in matches)
        moderate_zero = sum(value[1] for value in matches)
    else:
        strong_zero = int(zero_summary.loc[mode,'strong_or_complete'])
        moderate_zero = int(zero_summary.loc[mode,'moderate_or_better'])
    table11.append(dict(strategy=label,strong_original=int(original.strong_or_exact),strong_zero=strong_zero,
                        moderate_original=int(original.moderate_or_better),moderate_zero=moderate_zero))
table11 = pd.DataFrame(table11)
table11.to_csv(OUT / 'S11_bonus_sensitivity.csv',index=False)
configuration = dict(records=144,random_seed=20260722,iterations=1000,donors_per_draw=5,
    donor_definition='One best-matching candidate within each reference record frozen-base top-five window; record-weighted as in archived control.',
    target_exclusion='Exclude donors with the same normalized target formula.',
    target_transplant='Replace donor target product with recipient target; retain donor byproducts. This is a similarity-specific negative control, not a balanced proposed reaction.',
    scope='144 distinct records; extension86_078 excluded and initial59_059 retained. Donor and recipient populations are both deduplicated.',
    evaluation_only='Set matching B_template to zero; do not change ranking S_T, ranking B_T, weights, alpha, lambda, or displayed routes.')
(OUT / 'configuration.json').write_text(json.dumps(configuration,indent=2))
print(table[table.threshold.isin([.62,.70,.78,.80])].to_string(index=False),flush=True)
print(table11.to_string(index=False),flush=True)
