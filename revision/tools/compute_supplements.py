from pathlib import Path
from functools import lru_cache
from collections import Counter
import argparse
import importlib.util
import json
import re
import sys
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.optimize import linprog
from sympy import Matrix, Rational

ROOT = Path(__file__).resolve().parents[1]
HOME_ROOT = Path('/home/zhzhan')
RELEASE = ROOT / 'inputs/release'
if not RELEASE.exists():
    RELEASE = HOME_ROOT / 'InvDesRoute_audited_release_20261016'
AUDIT = ROOT / 'inputs/audit_results'
if not AUDIT.exists():
    AUDIT = HOME_ROOT / 'InvDesRoute_full_audit_20261016/results'
WORKFLOW = RELEASE / 'code/workflow'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(WORKFLOW / 'literature_benchmark'))
sys.path.insert(0, str(WORKFLOW / 'multi_route_benchmark'))
import evaluate_route_similarity as similarity
import evaluate_recommendation_ablation as ablation


@lru_cache(maxsize=None)
def amounts(formula):
    return Composition(formula).as_dict()


def parse_pipe(text):
    result = {}
    for part in str(text).split('|'):
        coefficient, formula = part.strip().split('*', 1)
        result[formula.strip()] = result.get(formula.strip(), 0) + float(coefficient)
    return result


def parse_written(text):
    result = {}
    for part in text.strip().split('+'):
        match = re.match(r'\s*(?:(\d+(?:\.\d+)?)\s+)?(\S+)\s*$', part)
        coefficient = float(match.group(1) or 1)
        formula = Composition(match.group(2)).reduced_formula
        result[formula] = result.get(formula, 0) + coefficient
    return result


def save_json(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


@lru_cache(maxsize=None)
def nullity(species):
    elements = sorted({element for formula in species for element in amounts(formula)})
    matrix = Matrix([[Rational(str(amounts(formula).get(element, 0))) for formula in species] for element in elements])
    return len(species) - matrix.rank()


def nullspace_and_bins():
    details = []
    by_target = []
    for path in sorted((RELEASE / 'data/core120/targets').glob('*/formation_reactions.csv')):
        frame = pd.read_csv(path)
        counts = Counter()
        for row in frame.to_dict('records'):
            species = tuple(sorted(set(parse_pipe(row['reactants'])) | set(parse_pipe(row['products']))))
            dimension = nullity(species)
            counts[dimension] += 1
            details.append(dict(target=path.parent.name, reaction=row['reaction'], species_count=len(species), nullity=dimension))
        by_target.append(dict(target=path.parent.name, retained_reactions=len(frame), nullity_one=counts[1],
                             nullity_greater_than_one=sum(value for key, value in counts.items() if key > 1)))
    pd.DataFrame(details).to_csv(RESULTS / 'TBD02_retained_reaction_nullity.csv', index=False)
    pd.DataFrame(by_target).to_csv(RESULTS / 'TBD02_nullity_by_target.csv', index=False)
    energy = pd.read_csv(AUDIT / 'independent_EquiformerV3-DeNS-OMat24-MPtrj-sAlex_reactions.csv')
    rows = []
    for lower, upper in [(0, .05), (.05, .1), (.1, .25), (.25, .5), (.5, 1), (1, float('inf'))]:
        mask = energy.dft_eV_written.abs().le(upper) & (energy.dft_eV_written.abs().gt(lower) if lower else True)
        group = energy[mask]
        rows.append(dict(lower_abs_DFT_eV_written=lower, upper_abs_DFT_eV_written=upper if np.isfinite(upper) else '>1',
            reactions=len(group), mae_eV_written=float(group.abs_error_eV_written.mean()) if len(group) else None,
            rmse_eV_written=float(np.sqrt(np.mean((group.predicted_eV_written-group.dft_eV_written)**2))) if len(group) else None,
            sign_correct=int((group.predicted_eV_written*group.dft_eV_written>0).sum())))
    pd.DataFrame(rows).to_csv(RESULTS / 'TBD05_hydride_error_bins.csv', index=False)
    thresholds = []
    for margin in [0, .025, .05, .10, .15, .20]:
        group = energy[energy.predicted_eV_written.lt(-margin)]
        thresholds.append(dict(predicted_negative_margin_eV_written=margin, accepted=len(group),
            accepted_reference_negative=int(group.dft_eV_written.lt(0).sum()),
            wrong_sign=int(group.dft_eV_written.ge(0).sum())))
    pd.DataFrame(thresholds).to_csv(RESULTS / 'TBD05_hydride_margin_sensitivity.csv', index=False)
    confidence_rows = []
    formation = {path.parent.name: pd.read_csv(path) for path in (RELEASE / 'data/core120/targets').glob('*/formation_reactions.csv')}
    for margin in [0, .025, .05, .10, .15, .20]:
        count = 0
        for target, frame in formation.items():
            practical = frame[frame.practical_route.astype(str).str.lower().eq('true')]
            energy_column = 'DeltaE_PRODUCTION_FORM_ALIGNED_per_target'
            count += bool(practical[energy_column].lt(-margin).any())
        confidence_rows.append(dict(margin_eV_per_target=margin, targets_with_exergonic_practical=count, targets=120))
    pd.DataFrame(confidence_rows).to_csv(RESULTS / 'TBD05_core120_margin_sensitivity.csv', index=False)
    save_json('TBD02_05_summary.json', dict(retained_routes=len(details), nullity_counts=dict(Counter(row['nullity'] for row in details)),
        scope='Retained formation-reaction supports only, not all attempted species combinations.',
        calibrated_covariance_available=False))


ORIGINAL_SIMILARITY = similarity.route_similarity


def without_bonus(target, precursors, byproducts, candidate):
    record = ORIGINAL_SIMILARITY(target, precursors, byproducts, candidate)
    reactants=similarity.parse_side(candidate.get('reactants'))
    candidate_byproducts=similarity.parse_side(candidate.get('products'))-{similarity.normalize(target)}
    reference_elements=similarity.nonvolatile_source_elements(precursors,target)
    candidate_elements=similarity.nonvolatile_source_elements(reactants,target)
    score=(.28*similarity.recall(precursors,reactants)+.18*similarity.jaccard(precursors,reactants)
        +.20*similarity.recall(reference_elements,candidate_elements)
        +.16*similarity.counter_jaccard(similarity.class_multiset(precursors,target),similarity.class_multiset(reactants,target))
        +.08*similarity.recall(byproducts,candidate_byproducts)
        +.05*similarity.counter_jaccard(similarity.class_multiset(byproducts,target),similarity.class_multiset(candidate_byproducts,target))
        +.05*similarity.jaccard(reference_elements,candidate_elements))
    score = min(1, max(0, score))
    exact = record['exact_reactant_recall'] >= .999 and record['byproduct_recall'] >= .999
    record['route_similarity_score'] = round(score, 6)
    record['route_similarity_label'] = ('exact_route_recovered' if exact else 'strong_route_family_match' if score >= .78
        else 'moderate_route_family_match' if score >= .62 else 'same_target_element_sources')
    return record


def matching_sensitivity():
    outputs = []
    rows_out = []
    for bonus, function in [('original', ORIGINAL_SIMILARITY), ('zero_bonus', without_bonus)]:
        similarity.route_similarity = function
        for cohort, case_folder, references in [('59','seed80_targets','ceder_seed80_real_literature_routes.csv'),
                ('86','extension53_targets','selected_literature_routes.csv')]:
            refs = similarity.read_csv(RELEASE / 'data/literature/references' / references)
            for mode in ['production_energy_all','production_energy_practical','synth_score','diverse_top20']:
                detail, summary = ablation.evaluate_mode(RELEASE / 'data/literature' / case_folder, refs, mode, [5])
                for record in detail:
                    record.update(bonus_setting=bonus, cohort=cohort)
                    rows_out.append(record)
                for record in summary:
                    record.update(bonus_setting=bonus, cohort=cohort)
                    outputs.append(record)
    similarity.route_similarity = ORIGINAL_SIMILARITY
    frame = pd.DataFrame(outputs)
    frame.to_csv(RESULTS / 'TBD07_template_bonus_by_cohort.csv', index=False)
    pd.DataFrame(rows_out).to_csv(RESULTS / 'TBD07_template_bonus_record_detail.csv', index=False)
    frame.groupby(['bonus_setting','mode'], as_index=False)[['n_cases','strong_or_exact','moderate_or_better']].sum().to_csv(
        RESULTS / 'TBD07_template_bonus_combined.csv', index=False)
    detail = []
    temp_path = WORKFLOW / 'multi_route_benchmark/temporal_validation_2022'
    sys.path.insert(0,str(temp_path))
    import evaluate_2022_temporal_benchmark as evaluation
    refs = pd.read_csv(RELEASE / 'data/temporal/split/future_routes_2023_2024.csv').fillna('')
    prior = pd.read_csv(RELEASE / 'data/temporal/split/historical_prior_routes_through_2020.csv', low_memory=False).fillna('')
    from optimize_ranking_target_disjoint import CorpusPrior
    model = CorpusPrior.fit(prior, set(refs.target))
    cases = {}
    for target in refs.target:
        frame = pd.read_csv(RELEASE / 'data/temporal/cases' / target / 'synthesizability_ranked_routes.csv').fillna('')
        cases[target] = frame.to_dict('records')
    evaluation.prepare_features(cases, model)
    for reference in refs.to_dict('records'):
        target = reference['target']
        ranked = evaluation.rank_routes(target, cases[target], 'condition_heavy', .3, .1, 5)
        lit_pre = similarity.split_formulas(reference['literature_precursors'])
        lit_by = similarity.split_formulas(reference['literature_byproducts'])
        strict = any(similarity.parse_side(row['reactants'])==lit_pre and
            similarity.parse_side(row['products'])-{similarity.normalize(target)}==lit_by for row in ranked)
        precursor_identity=any(similarity.parse_side(row['reactants'])==lit_pre for row in ranked)
        for bonus, function in [('original',ORIGINAL_SIMILARITY),('zero_bonus',without_bonus)]:
            matches=[function(target,lit_pre,lit_by,row) for row in ranked]
            strong=any(row['route_similarity_label'] in {'exact_route_recovered','strong_route_family_match'} for row in matches)
            moderate=any(row['route_similarity_label'] in {'exact_route_recovered','strong_route_family_match','moderate_route_family_match'} for row in matches)
            detail.append(dict(target=target,bonus_setting=bonus,strict_exact=strict,strong_or_exact=strong,moderate_or_better=moderate,
                strict_precursor_identity=precursor_identity,reference_all_elemental=all(len(amounts(formula))==1 for formula in lit_pre)))
    pd.DataFrame(detail).to_csv(RESULTS / 'TBD07_future_bonus_detail.csv', index=False)
    pd.DataFrame(detail).groupby('bonus_setting')[['strict_exact','strong_or_exact','moderate_or_better','strict_precursor_identity']].sum().to_csv(RESULTS/'TBD07_future_bonus_summary.csv')


def pressure_and_selectivity():
    selections = []
    summaries = []
    hulls = []
    competitors = []
    for pressure in [0,10]:
        frame = pd.read_csv(AUDIT / f'pressure_gpu_replay/phase_energies_{pressure}GPa.csv')
        selected = frame.sort_values(['enthalpy_per_formula_eV','cif_name']).drop_duplicates('reduced_formula')
        legacy = frame.drop_duplicates('formula_key', keep='last')
        old = dict(zip(legacy.reduced_formula,legacy.cif_name))
        for row in selected.to_dict('records'):
            row.update(selection='minimum_single_point_enthalpy',changed_from_legacy=row['cif_name']!=old.get(row['reduced_formula']))
            selections.append(row)
        enthalpy = dict(zip(selected.reduced_formula,selected.enthalpy_per_formula_eV))
        for system in ['Ag','Au']:
            refs = pd.read_csv(AUDIT / f'pressure_gpu_replay/{system}_{pressure}GPa_reaction_details.csv')
            predicted=[]
            for expression in refs.reaction:
                left,right=expression.split('->')
                predicted.append(sum(value*enthalpy[formula] for formula,value in parse_written(right).items())-
                    sum(value*enthalpy[formula] for formula,value in parse_written(left).items()))
            refs['minH_representative_model']=predicted
            refs.to_csv(RESULTS / f'TBD09_{system}_{pressure}GPa_reactions.csv',index=False)
            error=refs.minH_representative_model-refs.DFT
            summaries.append(dict(system=system,pressure_GPa=pressure,reactions=len(refs),
                mae_eV_written=float(error.abs().mean()),rmse_eV_written=float(np.sqrt(np.mean(error**2))),
                same_sign=int((refs.minH_representative_model*refs.DFT>0).sum()),
                changed_selected_compositions=sum(row['changed_from_legacy'] for row in selections if row['pressure_GPa']==pressure)))
        target=Composition('Li2AuH6').reduced_formula
        subset = selected[selected.reduced_formula.map(lambda formula:set(amounts(formula)) <= {'Li','Au','H'})]
        missing_hydrogen_reference=True
        if pressure==0:
            extra=pd.read_csv(RELEASE/'data/Li2AuH6/equiformer_mp_structure_energies.csv')
            hydrogen=extra[extra.formula.map(lambda formula:Composition(formula).reduced_formula)=='H2'].iloc[0]
            added=pd.DataFrame([dict(reduced_formula='H2',enthalpy_per_formula_eV=float(hydrogen.energy_per_formula_eV))])
            subset=pd.concat([subset,added],ignore_index=True)
            enthalpy['H2']=float(hydrogen.energy_per_formula_eV)
            missing_hydrogen_reference=False
        formulas = list(subset.reduced_formula)
        matrix = np.asarray([[amounts(formula).get(element,0) for formula in formulas] for element in ['Li','Au','H']],float)
        energies = subset.enthalpy_per_formula_eV.to_numpy()
        result=linprog(energies,A_eq=matrix,b_eq=[2,1,6],bounds=(0,None),method='highs')
        assert result.success
        hulls.append(dict(pressure_GPa=pressure,model='EquiformerV3',target=target,
            model_single_point_Ehull_eV_atom=float((enthalpy[target]-result.fun)/9),
            phase_pool_size=len(formulas),products=';'.join(f'{amount:.10g}*{formula}' for formula,amount in zip(formulas,result.x) if amount>1e-8),
            H2_reference_missing=missing_hydrogen_reference,
            scope='At 0 GPa local pool augmented by archived MP H2 single-point; at 10 GPa H2 missing. No relaxation; finite-pool diagnostic only.'))
        result=linprog(energies,A_eq=matrix,b_eq=[6,13,6],bounds=(0,None),method='highs')
        assert result.success
        alternative_mask=np.asarray([formula!=target for formula in formulas])
        alternative=linprog(energies[alternative_mask],A_eq=matrix[:,alternative_mask],b_eq=[6,13,6],bounds=(0,None),method='highs')
        assert alternative.success
        conversion_mask=np.asarray([formula not in {target,Composition('LiH').reduced_formula} for formula in formulas])
        conversion=linprog(energies[conversion_mask],A_eq=matrix[:,conversion_mask],b_eq=[6,13,6],bounds=(0,None),method='highs')
        reactant_energy=13*enthalpy[Composition('Au').reduced_formula]+6*enthalpy[Composition('LiH').reduced_formula]
        desired=enthalpy[target]+4*enthalpy[Composition('LiAu3').reduced_formula]
        competitors.append(dict(pressure_GPa=pressure,minimum_product_enthalpy_eV=float(result.fun),
            competing_reaction_eV_per_LiH=float((result.fun-reactant_energy)/6),
            desired_route_eV_per_target=float(desired-reactant_energy),
            selectivity_gap_desired_minus_equilibrium_eV_per_LiH=float((desired-result.fun)/6),
            target_excluded_competitor_eV_per_LiH=float((alternative.fun-reactant_energy)/6),
            desired_minus_target_excluded_competitor_eV_per_LiH=float((desired-alternative.fun)/6),
            complete_LiH_conversion_competitor_eV_per_LiH=float((conversion.fun-reactant_energy)/6) if conversion.success else float('nan'),
            complete_LiH_conversion_products=';'.join(f'{amount:.10g}*{formula}' for formula,amount in zip(np.asarray(formulas)[conversion_mask],conversion.x) if amount>1e-8) if conversion.success else 'Unavailable: no pressure-matched H2 reference',
            target_excluded_competitor_products=';'.join(f'{amount:.10g}*{formula}' for formula,amount in zip(np.asarray(formulas)[alternative_mask],alternative.x) if amount>1e-8),
            equilibrium_products=';'.join(f'{amount:.10g}*{formula}' for formula,amount in zip(formulas,result.x) if amount>1e-8)))
    pd.DataFrame(selections).to_csv(RESULTS/'TBD09_minimum_enthalpy_structure_selection.csv',index=False)
    pd.DataFrame(summaries).to_csv(RESULTS/'TBD09_pressure_minH_metrics.csv',index=False)
    shifts=[]
    for system in ['Ag','Au']:
        low=pd.read_csv(RESULTS/f'TBD09_{system}_0GPa_reactions.csv')
        high=pd.read_csv(RESULTS/f'TBD09_{system}_10GPa_reactions.csv')
        paired=low[['reaction','DFT','minH_representative_model']].merge(high[['reaction','DFT','minH_representative_model']],on='reaction',suffixes=('_0','_10'))
        difference=(paired.minH_representative_model_10-paired.minH_representative_model_0)-(paired.DFT_10-paired.DFT_0)
        shifts.append(dict(system=system,matched_reactions=len(paired),shift_mae_eV_written=float(difference.abs().mean())))
    pd.DataFrame(shifts).to_csv(RESULTS/'TBD09_pressure_shift_minH_metrics.csv',index=False)
    pd.DataFrame(hulls).to_csv(RESULTS/'TBD15_finite_pool_model_hulls.csv',index=False)
    pd.DataFrame(competitors).to_csv(RESULTS/'TBD15_precursor_selectivity.csv',index=False)


def statistics_and_baselines():
    stats = pd.read_csv(AUDIT/'external_element_complete_paired_statistics.csv')
    order = np.argsort(stats.p_unadjusted.to_numpy())
    adjusted = np.empty(len(stats))
    current=0
    for rank,index in enumerate(order):
        current=max(current,min(1,float(stats.iloc[index].p_unadjusted)*(len(stats)-rank)))
        adjusted[index]=current
    stats['p_holm_eight_tests']=adjusted
    stats.to_csv(RESULTS/'TBD_extra_Holm_statistics.csv',index=False)
    detail=pd.read_csv(RELEASE/'recomputed_full_final/external_common_endpoint_detail.csv')
    inv=detail[detail.model=='InvDesRoute-corpus-informed']
    rows=[]
    for model,group in detail[detail.model!='InvDesRoute-corpus-informed'].groupby('model'):
        targets=set(group[group.family_label.ne('no_valid_prediction') & group.best_target_element_coverage.ge(1)].target)
        if len(targets)==24:
            valid=pd.read_csv(AUDIT/'external_element_complete_validity.csv')
            expected=int(valid[valid.model==model].targets_with_predictions.iloc[0])
            if expected != len(targets):
                raise RuntimeError('Validity identification requires inspection.')
        reference=inv[inv.target.isin(targets)]
        group=group[group.target.isin(targets)]
        for name,frame in [(model,group),('InvDesRoute',reference)]:
            rows.append(dict(baseline_condition=model,model=name,valid_targets=len(targets),records=len(frame),
                strict_exact=int(frame.strict_exact_precursor_set.sum()),strong=int(frame.strong_or_better.sum()),moderate=int(frame.moderate_or_better.sum())))
    pd.DataFrame(rows).to_csv(RESULTS/'TBD_extra_valid_target_conditioned_comparison.csv',index=False)
    prior=pd.read_csv(RELEASE/'data/temporal/split/historical_prior_routes_through_2020.csv',low_memory=False).fillna('')
    refs=pd.read_csv(RELEASE/'data/temporal/split/future_routes_2023_2024.csv').fillna('')
    counters={}
    for row in prior.to_dict('records'):
        for formula in similarity.split_formulas(row['literature_precursors']):
            elements=set(amounts(formula))
            sources=elements-{'H','C','N','O'}
            if len(sources)==1:
                counters.setdefault(next(iter(sources)),Counter())[formula]+=1
    rows=[]
    for ref in refs.to_dict('records'):
        target=ref['target']
        lit=similarity.split_formulas(ref['literature_precursors'])
        for baseline in ['elemental_sources','per_element_historical_frequency']:
            predicted=set()
            target_elements=set(amounts(target))
            source_elements=target_elements-{'H','C','N','O'}
            for element in (target_elements if baseline=='elemental_sources' else source_elements):
                if baseline=='elemental_sources':
                    formula={'H':'H2','O':'O2','N':'N2','F':'F2','Cl':'Cl2'}.get(element,element)
                    predicted.add(similarity.normalize(formula))
                else:
                    counts=counters.get(element,Counter())
                    candidates=[formula for formula in counts if set(amounts(formula)) <= (target_elements|{'H','C','N','O'})]
                    predicted.add(sorted(candidates,key=lambda formula:(-counts[formula],formula))[0] if candidates else similarity.normalize(element))
            supplied=set().union(*(set(amounts(formula)) for formula in predicted)) if predicted else set()
            for element in target_elements-supplied:
                predicted.add(similarity.normalize({'H':'H2','O':'O2','N':'N2','F':'F2','Cl':'Cl2'}.get(element,element)))
            candidate=dict(reactants=' | '.join('1*'+formula for formula in sorted(predicted)),products='1*'+target,
                           route_template_reason='external_precursor_prediction')
            match=ORIGINAL_SIMILARITY(target,lit,set(),candidate)
            rows.append(dict(target=target,baseline=baseline,prediction=';'.join(sorted(predicted)),
                strict_precursor_identity=predicted==lit,reference_all_elemental=all(len(amounts(formula))==1 for formula in lit),
                strong=match['route_similarity_label'] in {'exact_route_recovered','strong_route_family_match'},
                moderate=match['route_similarity_label'] in {'exact_route_recovered','strong_route_family_match','moderate_route_family_match'}))
    frame=pd.DataFrame(rows)
    frame.to_csv(RESULTS/'TBD13_simple_baselines_detail.csv',index=False)
    frame.groupby(['baseline','reference_all_elemental'])[['strict_precursor_identity','strong','moderate']].agg(['sum','count']).to_csv(RESULTS/'TBD13_simple_baselines_by_group.csv')
    frame.groupby('baseline')[['strict_precursor_identity','strong','moderate']].sum().to_csv(RESULTS/'TBD13_simple_baselines_summary.csv')


parser=argparse.ArgumentParser()
parser.add_argument('analysis',choices=['nullspace','matching','pressure','statistics'])
args=parser.parse_args()
{'nullspace':nullspace_and_bins,'matching':matching_sensitivity,'pressure':pressure_and_selectivity,
 'statistics':statistics_and_baselines}[args.analysis]()
print(args.analysis,'completed',flush=True)
