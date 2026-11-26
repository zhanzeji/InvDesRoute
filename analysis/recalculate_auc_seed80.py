import os
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
HOME = Path(os.environ['INVDESROUTE_PRIVATE_INPUTS'])
RELEASE = HOME / 'release'
WORKFLOW = RELEASE / 'code/workflow'
sys.path.insert(0,str(WORKFLOW / 'multi_route_benchmark/temporal_validation_2022'))
from evaluate_2022_temporal_benchmark import (
    COMPONENTS,PROFILES,CorpusPrior,normalize,parse_side,split_formulas,route_similarity,
    profile_score,prepare_features,rank_routes,
)

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
old = pd.read_csv(HOME / 'audit/candidate_score_labels.csv')
candidate_frames=[]
for target in sorted(old.target.unique()):
    frame=pd.read_csv(RELEASE / 'data/multi_route/cases_blind_equiformer_v3' / target / 'synthesizability_ranked_routes.csv').fillna('')
    frame['target']=target
    frame['new_base_score']=[profile_score(row,PROFILES['condition_heavy']) for row in frame.to_dict('records')]
    frame['checked_energy_predictor']=-pd.to_numeric(frame['synth_rank_deltaE_per_target'],errors='coerce')
    candidate_frames.append(frame[['target','reaction','new_base_score','checked_energy_predictor',*COMPONENTS]])
components=pd.concat(candidate_frames,ignore_index=True)
assert not components.duplicated(['target','reaction']).any()
data=old.merge(components,on=['target','reaction'],validate='one_to_one')
assert len(data)==11226
assert data.pet_energy_predictor.notna().equals(data.checked_energy_predictor.notna())
assert np.allclose(data.pet_energy_predictor.dropna(),data.checked_energy_predictor.dropna(),rtol=0,atol=1e-10)
assert np.allclose(data.precursor_prior_score_x,data.precursor_prior_score_y,rtol=0,atol=1e-10)
assert np.allclose(data.route_template_score_x,data.route_template_score_y,rtol=0,atol=1e-10)
data.to_csv(OUT / 'AUC_candidate_scores.csv',index=False)
metrics=[]
for population,frame in [('all11226',data),('common5326',data[data.pet_energy_predictor.notna()])]:
    if population=='common5326':assert len(frame)==5326
    for predictor in ['new_base_score','precursor_prior_score_x','route_template_score_x','pet_energy_predictor']:
        available=frame[frame[predictor].notna()]
        labels=available.exact_full_route.astype(int)
        values=available[predictor].astype(float)
        macro=[roc_auc_score(group.exact_full_route.astype(int),group[predictor].astype(float))
               for _,group in available.groupby('target') if group.exact_full_route.nunique()==2]
        cutoff=values.quantile(.9)
        top=labels[values>=cutoff]
        metrics.append(dict(population=population,predictor=predictor,candidates=len(available),positives=int(labels.sum()),
            ROC_AUC=float(roc_auc_score(labels,values)),AP=float(average_precision_score(labels,values)),
            target_macro_AUC=float(np.mean(macro)),enrichment=float(top.mean()/labels.mean())))
metrics=pd.DataFrame(metrics)
metrics.to_csv(OUT / 'AUC_new_policy_summary.csv',index=False)
print(metrics.to_string(index=False),flush=True)

prior_frame=pd.read_csv(RELEASE / 'data/temporal/split/historical_prior_routes_through_2020.csv',low_memory=False).fillna('')
excluded=set()
for filename in ['development_routes_2021_2022.csv','future_routes_2023_2024.csv']:
    excluded.update(pd.read_csv(RELEASE / 'data/temporal/split' / filename).target.map(normalize))
prior=CorpusPrior.fit(prior_frame,excluded)
references=pd.read_csv(RELEASE / 'data/literature/references/literature_routes_seed80.csv').fillna('')
seed_rows=[]
for index,reference in references.iterrows():
    target=normalize(reference.target)
    routes=pd.read_csv(RELEASE / 'data/literature/seed80_targets' / target / 'synthesizability_ranked_routes.csv').fillna('').to_dict('records')
    prepare_features({target:routes},prior)
    precursors=split_formulas(reference.literature_precursors)
    byproducts=split_formulas(reference.literature_byproducts)
    for mode,alpha,penalty in [('new_base',0,0),('new_full',.3,.1)]:
        ranked=rank_routes(target,routes,'condition_heavy',alpha,penalty,20)
        for window in [5,20,'all']:
            selected=routes if window=='all' else ranked[:window]
            best=max((route_similarity(target,precursors,byproducts,route) for route in selected),key=lambda match:match['route_similarity_score'])
            seed_rows.append(dict(target=target,mode=mode,window=str(window),complete_recall=best['route_similarity_label']=='exact_route_recovered',
                strong=best['route_similarity_label'] in ['exact_route_recovered','strong_route_family_match'],
                moderate=best['route_similarity_label'] in ['exact_route_recovered','strong_route_family_match','moderate_route_family_match']))
    if (index+1)%20==0:print(f'Seed80: {index+1}/80',flush=True)
seed=pd.DataFrame(seed_rows)
seed.to_csv(OUT / 'seed80_record_results.csv',index=False)
seed_summary=seed.groupby(['mode','window']).agg(records=('target','count'),complete_recall=('complete_recall','sum'),strong=('strong','sum'),moderate=('moderate','sum')).reset_index()
seed_summary.to_csv(OUT / 'seed80_new_policy_summary.csv',index=False)
print(seed_summary.to_string(index=False),flush=True)
(OUT / 'configuration.json').write_text(json.dumps(dict(profile='condition_heavy',weights=dict(zip(COMPONENTS,PROFILES['condition_heavy'])),alpha=.3,mmr=.1,
    auc_endpoint='strict normalized precursor and byproduct set identity; archived labels unchanged',
    auc_common_candidates=5326,auc_all_candidates=11226,seed_records=80),indent=2))
