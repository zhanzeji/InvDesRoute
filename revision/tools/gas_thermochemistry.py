from pathlib import Path
from functools import lru_cache
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pymatgen.core import Composition

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'inputs/release'
if not RELEASE.exists():RELEASE=Path('/home/zhzhan/InvDesRoute_audited_release_20261016')
AUDIT=ROOT/'inputs/audit_results'
if not AUDIT.exists():AUDIT=Path('/home/zhzhan/InvDesRoute_full_audit_20261016/results')
WORKFLOW=RELEASE/'code/workflow'
RESULTS=ROOT/'results'
sys.path.insert(0,str(WORKFLOW/'multi_route_benchmark/temporal_validation_2022'))
sys.path.insert(0,str(WORKFLOW/'multi_route_benchmark'))
sys.path.insert(0,str(WORKFLOW/'literature_benchmark'))
import evaluate_2022_temporal_benchmark as evaluation
from optimize_ranking_target_disjoint import CorpusPrior
import evaluate_route_similarity as similarity

KJ_MOL_PER_EV=96.48533212331002
KB=8.617333262145e-5
GASES={'O2':'O-029','CO2':'C-095','H2':'H-050','H2O':'H-064','N2':'N-023','NH3':'H-083','CO':'C-093'}
RAW_GAS={Composition(name).reduced_formula:name for name in GASES}
UNSUPPORTED={Composition(name).reduced_formula for name in ['F2','Cl2','Br2','I2']}


def read_table(name,code):
    path=ROOT/'source_snapshots'/f'JANAF_{name}_{code}.txt'
    records=[]
    for line in path.read_text().splitlines()[2:]:
        parts=line.split()
        if len(parts)<5:continue
        try:records.append((float(parts[0]),float(parts[2]),float(parts[4])))
        except ValueError:continue
    return dict((temperature,dict(S_J_molK=entropy,H_minus_H298_kJ_mol=enthalpy)) for temperature,entropy,enthalpy in records)


TABLES={name:read_table(name,code) for name,code in GASES.items()}


def mu(name,temperature,pressure=1.0,reference='H0'):
    record=TABLES[name][float(temperature)]
    h=record['H_minus_H298_kJ_mol']
    if reference=='H0':h-=TABLES[name][0.0]['H_minus_H298_kJ_mol']
    return (h-temperature*record['S_J_molK']/1000)/KJ_MOL_PER_EV+KB*temperature*np.log(pressure)


@lru_cache(maxsize=None)
def gas_stoichiometry(reactants,products,target_coefficient):
    net={}
    for side,sign in [(reactants,-1),(products,1)]:
        for item in str(side).split('|'):
            if '*' not in item:continue
            number,formula=item.strip().split('*',1)
            formula=Composition(formula.strip()).reduced_formula
            if formula in RAW_GAS or formula in UNSUPPORTED:
                net[formula]=net.get(formula,0)+sign*float(number)/float(target_coefficient)
    return {formula:number for formula,number in net.items() if abs(number)>1e-12}


def corrected(row,temperature,reference='H0'):
    value=pd.to_numeric(row.get('synth_rank_deltaE_per_target'),errors='coerce')
    net=gas_stoichiometry(str(row.get('reactants','')),str(row.get('products','')),float(row['target_coefficient']))
    missing=sorted(set(net)&UNSUPPORTED)
    if missing or not np.isfinite(value):return float('nan'),';'.join(missing) or 'missing_static_energy'
    change=sum(number*mu(RAW_GAS[formula],temperature,reference=reference) for formula,number in net.items())
    return float(value+change),''


def main():
    records=[]
    for name in TABLES:
        for temperature in [298.15,700,1000,1300]:
            if temperature not in TABLES[name]:continue
            records.append(dict(species=name,T_K=temperature,pressure_bar=1,
                S_J_molK=TABLES[name][temperature]['S_J_molK'],
                H_T_minus_H_0_kJ_mol=TABLES[name][temperature]['H_minus_H298_kJ_mol']-TABLES[name][0]['H_minus_H298_kJ_mol'],
                delta_mu_H0_eV_molecule=mu(name,temperature),delta_mu_H298_eV_molecule=mu(name,temperature,reference='H298')))
    pd.DataFrame(records).to_csv(RESULTS/'TBD06_JANAF_chemical_potentials.csv',index=False)
    flags=[]
    for target in ['BaTiO3','SrTiO3']:
        path=RELEASE/'data/multi_route/cases_blind_equiformer_v3'/target/'synthesizability_ranked_routes.csv'
        frame=pd.read_csv(path)
        for row in frame.to_dict('records'):
            net=gas_stoichiometry(row['reactants'],row['products'],row['target_coefficient'])
            if set(net)=={Composition('CO2').reduced_formula} and len(similarity.parse_side(row['reactants']))==2:
                for temperature in [700,1000,1300]:
                    energy,status=corrected(row,temperature)
                    flags.append(dict(target=target,reaction=row['reaction'],T_K=temperature,
                        static_baseline_cohort='24_target_generic_library',
                        static_eV_target=row['synth_rank_deltaE_per_target'],gas_only_H0_eV_target=energy,
                        gas_only_H298_eV_target=corrected(row,temperature,'H298')[0],status=status))
    examples=pd.read_csv(AUDIT/'figure1_cases_recomputed.csv')
    for row in examples[examples['rank']==1].to_dict('records'):
        for temperature in [700,1000,1300]:
            flags.append(dict(target=row['target'],reaction=row['reaction'],T_K=temperature,
                static_baseline_cohort='frozen_policy_illustrative_cases_target_excluded',
                static_eV_target=row['delta_e'],gas_only_H0_eV_target=row['delta_e']+mu('CO2',temperature),
                gas_only_H298_eV_target=row['delta_e']+mu('CO2',temperature,reference='H298'),status='gas_only_sensitivity'))
    pd.DataFrame(flags).to_csv(RESULTS/'TBD06_carbonate_examples.csv',index=False)
    legacy=pd.read_csv(AUDIT/'discrimination_recomputed/candidate_score_labels.csv')
    data=[]
    for target,group in legacy.groupby('target'):
        routes=pd.read_csv(RELEASE/'data/multi_route/cases_blind_equiformer_v3'/target/'synthesizability_ranked_routes.csv').fillna('')
        lookup={row['reaction']:row for row in routes.to_dict('records')}
        for label in group.to_dict('records'):
            row=lookup[label['reaction']]
            value,status=corrected(row,1000)
            label.update(static_current_eV_target=pd.to_numeric(row['synth_rank_deltaE_per_target'],errors='coerce'),
                gas_only_H0_1000K_eV_target=value,gas_status=status)
            data.append(label)
    frame=pd.DataFrame(data)
    frame.to_csv(RESULTS/'TBD16_candidate_gas_thermal_scores.csv',index=False)
    valid=frame.dropna(subset=['static_current_eV_target','gas_only_H0_1000K_eV_target'])
    auc=[]
    for endpoint in ['exact_full_route','exact_precursor_set','moderate_or_better']:
        y=valid[endpoint].astype(str).str.lower().eq('true')
        for name,column in [('archived_static_model_reservoir','static_current_eV_target'),('JANAF_gas_only_H0_1000K','gas_only_H0_1000K_eV_target')]:
            auc.append(dict(endpoint=endpoint,variant=name,common_candidates=len(valid),positives=int(y.sum()),
                roc_auc=float(roc_auc_score(y,-valid[column])),average_precision=float(average_precision_score(y,-valid[column]))))
    pd.DataFrame(auc).to_csv(RESULTS/'TBD16_gas_thermal_auc_common_subset.csv',index=False)
    refs=pd.read_csv(RELEASE/'data/temporal/split/future_routes_2023_2024.csv').fillna('')
    prior_frame=pd.read_csv(RELEASE/'data/temporal/split/historical_prior_routes_through_2020.csv',low_memory=False).fillna('')
    prior=CorpusPrior.fit(prior_frame,set(refs.target))
    original_cases={}
    for target in refs.target:
        frame=pd.read_csv(RELEASE/'data/temporal/cases'/target/'synthesizability_ranked_routes.csv').replace({np.nan:''})
        original_cases[target]=frame.to_dict('records')
    evaluation.prepare_features(original_cases,prior)
    outcomes=[]
    displays=[]
    for variant in ['archived_static_with_floor','static_no_floor','JANAF_gas_only_H0_1000K_no_floor']:
        for reference in refs.to_dict('records'):
            target=reference['target']
            cases=[]
            for original in original_cases[target]:
                row=dict(original)
                if variant=='static_no_floor':row['energy_score']=evaluation.energy_component(row.get('synth_rank_deltaE_per_target'))
                elif variant=='JANAF_gas_only_H0_1000K_no_floor':
                    value,status=corrected(row,1000)
                    row['diagnostic_static_energy']=row.get('synth_rank_deltaE_per_target')
                    row['diagnostic_gas_status']=status
                    row['synth_rank_deltaE_per_target']=value if np.isfinite(value) else ''
                    row['energy_score']=evaluation.energy_component(value)
                cases.append(row)
            ranked=evaluation.rank_routes(target,cases,'condition_heavy',.3,.1,5)
            pre=similarity.split_formulas(reference['literature_precursors'])
            by=similarity.split_formulas(reference['literature_byproducts'])
            matches=[similarity.route_similarity(target,pre,by,row) for row in ranked]
            strict=any(similarity.parse_side(row['reactants'])==pre and similarity.parse_side(row['products'])-{target}==by for row in ranked)
            outcomes.append(dict(target=target,variant=variant,strict_exact=strict,
                strong_or_exact=any(row['route_similarity_label'] in {'exact_route_recovered','strong_route_family_match'} for row in matches),
                moderate_or_better=any(row['route_similarity_label'] in {'exact_route_recovered','strong_route_family_match','moderate_route_family_match'} for row in matches)))
            for index,row in enumerate(ranked,1):
                displays.append(dict(target=target,variant=variant,rank=index,reaction=row['reaction'],
                    evaluated_energy_eV_target=row['synth_rank_deltaE_per_target'],
                    diagnostic_gas_status=row.get('diagnostic_gas_status','not_applied')))
    pd.DataFrame(outcomes).to_csv(RESULTS/'TBD16_future_gas_ablation_detail.csv',index=False)
    pd.DataFrame(outcomes).groupby('variant')[['strict_exact','strong_or_exact','moderate_or_better']].sum().to_csv(RESULTS/'TBD16_future_gas_ablation_summary.csv')
    pd.DataFrame(displays).to_csv(RESULTS/'TBD16_future_gas_displayed.csv',index=False)
    temperature_counts=[]
    for temperature in [700,1000,1300]:
        total=complete=negative=changed=0
        for target,cases in original_cases.items():
            for row in cases:
                value,status=corrected(row,temperature)
                total+=1
                if np.isfinite(value):
                    complete+=1
                    negative+=value<-.05
                    base=float(row['synth_rank_deltaE_per_target'])
                    changed+=(base<0)!=(value<0)
        temperature_counts.append(dict(T_K=temperature,candidates=total,thermal_complete=complete,
            negative_margin_0p05=int(negative),sign_changed_vs_static=int(changed)))
    pd.DataFrame(temperature_counts).to_csv(RESULTS/'TBD06_future_temperature_flags.csv',index=False)
    selectivity=pd.read_csv(RESULTS/'TBD15_precursor_selectivity.csv')
    rows=[]
    for record in selectivity.to_dict('records'):
        for temperature in [700,1000,1300]:
            for pressure in [1,100]:
                thermal=3*mu('H2',temperature,pressure)
                rows.append(dict(structure_pressure_GPa=record['pressure_GPa'],T_K=temperature,pH2_bar=pressure,
                    desired_route_gas_increment_eV_target=0,
                    desired_route_static_eV_target=record['desired_route_eV_per_target'],
                    H2_releasing_complete_conversion_static_eV_per_LiH=record['complete_LiH_conversion_competitor_eV_per_LiH'],
                    H2_releasing_complete_conversion_gas_only_eV_per_LiH=record['complete_LiH_conversion_competitor_eV_per_LiH']+thermal/6,
                    model_only_scope='Gas-only sensitivity on finite single-point phase pool; not a full Gibbs phase diagram.'))
    pd.DataFrame(rows).to_csv(RESULTS/'TBD15_H2_temperature_pressure_sensitivity.csv',index=False)
    (RESULTS/'TBD06_16_scope.json').write_text(json.dumps(dict(gas_species=list(GASES),
        gas_table_reference='H(T)-H(0)-T*S(T), 1 bar; H298 alternative also reported for examples.',
        archive_static_reservoir_zero_point_certified=False,solid_phonons_included=False,
        MP2020_compatibility_problem_resolved=False,interpretation='Exploratory gas-only increment on archived static baseline; not calibrated absolute free energies.'),indent=2))
    print('JANAF and gas-only diagnostics completed')


if __name__=='__main__':main()
