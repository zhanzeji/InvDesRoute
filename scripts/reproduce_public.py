from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/workflow/multi_route_benchmark/temporal_validation_2022'))
from evaluate_2022_temporal_benchmark import rank_routes,profile_score,PROFILES

def main():
    parser=argparse.ArgumentParser(description='Reproduce v0.2.1 rankings and metrics from the public derived-data archive.')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'recomputed')
    args=parser.parse_args()
    data=args.data.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'DATA_SHA256.json').read_text())
    for filename,expected in manifest.items():
        actual=hashlib.sha256((data/filename).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Checksum mismatch: {filename}')
    features=pd.read_csv(data/'inputs/route_features.csv').fillna('')
    refs=pd.read_csv(data/'inputs/reference_manifest.csv').fillna('')
    metrics=pd.read_csv(data/'inputs/candidate_reference_metrics.csv')
    record_matches={record:frame.set_index('route_id') for record,frame in metrics.groupby('record_id')}
    ranking={}
    displays=[]
    for (pool,target),frame in features.groupby(['pool','target']):
        routes=frame.to_dict('records')
        for mode,alpha,penalty in [('energy_only',0,0),('practical_energy',0,0),('base',0,0),('base_corpus',.3,0),('final',.3,.1)]:
            if mode in ['energy_only','practical_energy']:
                def key(route):
                    try: value=float(route['baseline_energy_eV_target'])
                    except (ValueError,TypeError): value=float('inf')
                    practical=0 if str(route['baseline_practical']).lower() in ['true','1','yes'] else 1
                    return ((practical,) if mode=='practical_energy' else ())+(value,route['reaction'])
                ranked=sorted(routes,key=key)[:20]
            else:
                ranked=rank_routes(target,routes,'condition_heavy',alpha,penalty,20)
            ranking[(pool,target,mode)]=ranked
            for rank,route in enumerate(ranked,1):
                displays.append(dict(pool=pool,target=target,mode=mode,rank=rank,route_id=route['route_id'],
                    reaction=route['reaction'],base_score=route.get('evaluation_base_score',''),rank_score=route.get('evaluation_rank_score','')))
    pd.DataFrame(displays).to_csv(output/'ranked_routes_top20.csv',index=False)
    detail=[]
    for reference in refs.to_dict('records'):
        matches=record_matches[reference['record_id']]
        for mode in ['energy_only','practical_energy','base','base_corpus','final']:
            ranked=ranking[(reference['pool'],reference['target'],mode)]
            for window in [1,5,10,20]:
                selected=matches.loc[[route['route_id'] for route in ranked[:window]]]
                best=selected.iloc[int(np.argmax(selected.similarity.to_numpy()))]
                zero_best=selected.iloc[int(np.argmax(selected.zero_bonus_similarity.to_numpy()))]
                detail.append(dict(role=reference['role'],record_id=reference['record_id'],mode=mode,top_k=window,
                    strict_identity=bool(selected.strict_identity.any()),
                    complete_recall=bool(best.complete_recall),
                    strong_or_complete=bool(best.complete_recall or best.similarity>=.78),
                    moderate_or_better=bool(best.complete_recall or best.similarity>=.62),
                    zero_strong=bool(zero_best.complete_recall or zero_best.zero_bonus_similarity>=.78),
                    zero_moderate=bool(zero_best.complete_recall or zero_best.zero_bonus_similarity>=.62)))
    details=pd.DataFrame(detail)
    details.to_csv(output/'record_recovery.csv',index=False)
    recovery=details.groupby(['role','mode','top_k'])[['strict_identity','complete_recall','strong_or_complete','moderate_or_better','zero_strong','zero_moderate']].sum().reset_index()
    recovery.to_csv(output/'recovery_summary.csv',index=False)
    coverage=[]
    for reference in refs[refs.role.eq('retrospective144')].to_dict('records'):
        matches=record_matches[reference['record_id']]
        coverage.append(dict(record_id=reference['record_id'],cohort=reference['cohort'],
            complete_recall=bool(matches.complete_recall.any()),strict_identity=bool(matches.strict_identity.any()),
            strong_or_complete=bool(matches.complete_recall.any() or matches.similarity.max()>=.78),
            moderate_or_better=bool(matches.complete_recall.any() or matches.similarity.max()>=.62)))
    coverage=pd.DataFrame(coverage)
    coverage.to_csv(output/'full_network_coverage.csv',index=False)
    auc=pd.read_csv(data/'inputs/auc_candidates.csv')
    recalculated=np.asarray([profile_score(route,PROFILES['condition_heavy']) for route in auc.fillna('').to_dict('records')])
    assert np.allclose(recalculated,auc.new_base_score.to_numpy(),rtol=0,atol=1e-12)
    auc['new_base_score']=recalculated
    common=auc[auc.negative_static_reaction_energy.notna()]
    auc_rows=[]
    for predictor in ['new_base_score','precursor_score','template_score','negative_static_reaction_energy']:
        auc_rows.append(dict(predictor=predictor,candidates=len(common),positives=int(common.exact_full_route.sum()),
            ROC_AUC=float(roc_auc_score(common.exact_full_route,common[predictor])),AP=float(average_precision_score(common.exact_full_route,common[predictor]))))
    pd.DataFrame(auc_rows).to_csv(output/'auc_common_subset.csv',index=False)
    displayed_energy=pd.read_csv(data/'inputs/temporal_displayed_energy.csv')
    physical=[]
    for configuration,frame in displayed_energy.groupby('configuration'):
        energy=pd.to_numeric(frame.synth_rank_deltaE_per_target,errors='coerce')
        physical.append(dict(configuration=configuration,displayed=len(frame),complete=int(energy.notna().sum()),
            missing=int(energy.isna().sum()),strongly_uphill=int(energy.gt(1).sum()),
            uphill_fraction=float(energy.gt(1).sum()/energy.notna().sum())))
    physical=pd.DataFrame(physical).set_index('configuration')
    physical.to_csv(output/'temporal_energy_fractions.csv')
    phase=pd.read_csv(data/'inputs/hydride_phase_residuals.csv')
    reaction=pd.read_csv(data/'inputs/hydride_reaction_residuals.csv')
    hydride=[]
    for model,frame in reaction.groupby('model'):
        phase_group=phase[phase.model==model]
        hydride.append(dict(model=model,phases=len(phase_group),reactions=len(frame),
            phase_MAE_eV_atom=float(phase_group.error_eV_atom.abs().mean()),reaction_MAE_eV_written=float(frame.error_eV_written.abs().mean()),
            reaction_RMSE_eV_written=float(np.sqrt(np.mean(frame.error_eV_written**2))),sign_agreement=float(frame.sign_correct.mean())))
    pd.DataFrame(hydride).to_csv(output/'hydride_reconstruction_metrics.csv',index=False)
    temporal=pd.read_csv(data/'inputs/temporal_record_metrics.csv')
    temporal=temporal.groupby(['configuration','top_k'])[['strict_exact','strong_or_exact','moderate_or_better']].sum().reset_index()
    temporal.to_csv(output/'temporal_recovery.csv',index=False)
    external=pd.read_csv(data/'inputs/external_record_metrics.csv')
    external=external[external.top_k.eq(5)].groupby('model')[['strict_exact_precursor_set','strong_or_better','moderate_or_better']].sum()
    external.to_csv(output/'external_recovery.csv')
    own_external=external[external.index.str.startswith('InvDesRoute')]
    assert len(own_external)==1
    network=pd.read_csv(data/'inputs/network_metrics_by_target.csv')
    network_rows=[]
    for cohort,frame in network.groupby('cohort'):
        network_rows.append(dict(cohort=cohort,targets=len(frame),reactions=int(frame.comparable_routes.sum()),
            mean_MAE=float(frame.mae.mean()),mean_sign=float(frame.sign.mean()),top1=int(frame.top1.sum()),mean_overlap=float(frame.overlap.mean())))
    pd.DataFrame(network_rows).to_csv(output/'network_summary.csv',index=False)
    final=recovery[(recovery.role=='retrospective144') & (recovery['mode']=='final') & recovery.top_k.eq(5)].iloc[0]
    seed=recovery[(recovery.role=='positive_control80') & (recovery['mode']=='final') & recovery.top_k.eq(5)].iloc[0]
    future=temporal[(temporal.configuration=='selected_with_equiformer_energy') & temporal.top_k.eq(5)].iloc[0]
    checks=dict(
        data_checksums=True,
        temporal_energy_with=physical.loc['selected_with_equiformer_energy',['displayed','complete','missing','strongly_uphill']].tolist()==[280,266,14,96],
        temporal_energy_without=physical.loc['selected_without_energy',['displayed','complete','missing','strongly_uphill']].tolist()==[280,263,17,154],

        deduplicated_references=int(refs.role.eq('retrospective144').sum())==144 and 'extension86_078' not in set(refs.record_id),
        complete_network_coverage=[int(coverage.strong_or_complete.sum()),int(coverage.moderate_or_better.sum())]==[140,142],
        generic_complete_recall=int(coverage[coverage.cohort.eq('initial59')].complete_recall.sum())==40,

        retrospective144=[int(final.strict_identity),int(final.strong_or_complete),int(final.moderate_or_better)]==[62,91,114],
        zero_bonus_final=[int(final.zero_strong),int(final.zero_moderate)]==[80,105],
        positive_control80=[int(seed.complete_recall),int(seed.moderate_or_better)]==[78,80],
        auc_common_candidates=len(common)==5326,
        auc_base=abs(auc_rows[0]['ROC_AUC']-.8315041514737507)<1e-12,
        all_seven_reconstruction_models=len(hydride)==7 and all(row['phases']==107 and row['reactions']==416 for row in hydride),
        temporal=[int(future.strict_exact),int(future.strong_or_exact),int(future.moderate_or_better)]==[47,49,53],
        external_invdesroute=own_external.iloc[0].tolist()==[64,79,110])
    for mode,expected in [('energy_only',[6,8]),('practical_energy',[8,17])]:
        baseline=recovery[(recovery.role=='retrospective144') & (recovery['mode']==mode) & recovery.top_k.eq(5)].iloc[0]
        checks[mode+'_recovery']=[int(baseline.strong_or_complete),int(baseline.moderate_or_better)]==expected
    (output/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    if not all(checks.values()):raise SystemExit('One or more v0.2.1 reproduction checks failed.')

if __name__=='__main__':main()
