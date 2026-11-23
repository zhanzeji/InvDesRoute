from pathlib import Path
import argparse
import json
import sys
import subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
WORKFLOW=ROOT/'code/workflow'

def hydride(output):
    sys.path.insert(0,str(WORKFLOW/'matbench_top_model_reaction_benchmark/scripts'))
    import benchmark_common as common
    common.DATA=DATA/'hydride/input'
    dft,dft_epa,atoms=common.load_dft_reference()
    reactions=pd.read_csv(common.DATA/'reactions_416.csv')
    rows=[]
    for path in sorted((DATA/'hydride/model_predictions').glob('*_phase_energies.csv')):
        frame=pd.read_csv(path)
        frame=frame[frame.status.eq('ok')].drop_duplicates('formula',keep='last')
        energy=dict(zip(frame.formula,frame.energy_per_formula_eV))
        differences=[]
        correct=[]
        for expression in reactions.reaction:
            try:
                predicted=common.reaction_energy(expression,energy)
                reference=common.reaction_energy(expression,dft)
            except KeyError:
                continue
            differences.append(predicted-reference)
            correct.append(predicted*reference>0)
        phase=[energy[formula]/atoms[formula]-dft_epa[formula] for formula in sorted(energy.keys()&dft.keys())]
        rows.append(dict(model=path.stem.removesuffix('_phase_energies'),phases=len(phase),reactions=len(differences),
            phase_mae_eV_atom=float(np.mean(np.abs(phase))),reaction_mae_eV_written=float(np.mean(np.abs(differences))),
            reaction_rmse_eV_written=float(np.sqrt(np.mean(np.square(differences)))),sign=float(np.mean(correct))))
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'seven_model_recomputed.csv',index=False)
    return frame.to_dict('records')

def core(output):
    rows=[]
    for path in sorted((DATA/'core120/targets').glob('*/formation_reactions.csv')):
        frame=pd.read_csv(path)
        model='DeltaE_EquiformerV3+DeNS-OAM2_per_target'
        reference='DeltaE_MP_FORM_per_target'
        differences=frame[model]-frame[reference]
        rows.append(dict(target=path.parent.name,reactions=len(frame),mae=float(differences.abs().mean()),
            sign=float((frame[model]*frame[reference]>0).mean()),top1=frame.nsmallest(1,model).reaction.iloc[0]==frame.nsmallest(1,reference).reaction.iloc[0],
            overlap=len(set(frame.nsmallest(5,model).reaction)&set(frame.nsmallest(5,reference).reaction))))
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'core120_by_target.csv',index=False)
    return dict(targets=len(frame),reactions=int(frame.reactions.sum()),mae=float(frame.mae.mean()),sign=float(frame.sign.mean()),top1=int(frame.top1.sum()),overlap=float(frame.overlap.mean()))

def external(output):
    sys.path.insert(0,str(WORKFLOW/'generative_model_benchmark'))
    import evaluate_models as evaluator
    from pymatgen.core import Composition
    evaluator.BENCHMARK=DATA/'external/benchmark_24x5.csv'
    original=evaluator.normalize_prediction
    def valid(row):
        formulas=original(row)
        if not formulas:
            return None
        elements=set().union(*(set(Composition(formula).as_dict()) for formula in formulas))
        return formulas if set(Composition(row['target']).as_dict()).issubset(elements) else None
    evaluator.normalize_prediction=valid
    paths=[DATA/'external/predictions'/name for name in ['chemllm-7b-chat.csv','chemdfm-v1.5-8b.csv','llm4syn-tgt2ceq-v1.2.1.csv']]
    paths += [DATA/'external/csllm_precursor_top5.csv',DATA/'external/invdesroute_frozen_2022_predictions.csv']
    details=[]
    for path in paths:
        rows,_=evaluator.evaluate_file(path,[5])
        details.extend(rows)
    frame=pd.DataFrame(details)
    frame.to_csv(output/'external_common_endpoint_detail.csv',index=False)
    rows=[]
    for name,group in frame.groupby('model'):
        rows.append(dict(model=name,records=len(group),exact=int(group.strict_exact_precursor_set.sum()),strong=int(group.strong_or_better.sum()),moderate=int(group.moderate_or_better.sum()),recall=float(group.best_precursor_recall.mean())))
    pd.DataFrame(rows).to_csv(output/'external_common_endpoint_summary.csv',index=False)
    return rows

def full_analysis(output):
    temporal=WORKFLOW/'multi_route_benchmark/temporal_validation_2022'
    subprocess.run([sys.executable,str(temporal/'evaluate_2022_temporal_benchmark.py'),
        '--split-dir',str(DATA/'temporal/split'),'--case-root',str(DATA/'temporal/cases'),
        '--output-dir',str(output/'temporal')],check=True)
    subprocess.run([sys.executable,str(temporal/'analyze_mlip_route_contribution.py'),
        '--split-dir',str(DATA/'temporal/split'),'--case-root',str(DATA/'temporal/cases'),
        '--benchmark-summary',str(output/'temporal/summary.json'),'--output-dir',str(output/'mlip')],check=True)
    for label,cases,references in [
        ('59',DATA/'literature/seed80_targets',DATA/'literature/references/ceder_seed80_real_literature_routes.csv'),
        ('86',DATA/'literature/extension53_targets',DATA/'literature/references/selected_literature_routes.csv')]:
        for script in ['evaluate_route_similarity.py','evaluate_recommendation_ablation.py']:
            subprocess.run([sys.executable,str(WORKFLOW/'literature_benchmark'/script),
                '--case-dir',str(cases),'--literature',str(references),'--outdir',str(output/('literature'+label))],check=True)
    sys.path.insert(0,str(temporal))
    import evaluate_external_models_frozen_score as frozen
    frozen.GENERATION=DATA/'external_generation_layout'
    frozen.SPLIT=DATA/'temporal/split'
    frozen.BENCHMARK=DATA/'multi_route'
    frozen.RESULTS=output/'external_frozen_policy'
    frozen.main()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-analysis',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'recomputed')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    results=dict(hydride=hydride(output),core120=core(output),external=external(output))
    if args.full_analysis:
        full_analysis(output)
    (output/'summary.json').write_text(json.dumps(results,indent=2))
    expected=json.loads((DATA/'expected_recomputed_metrics.json').read_text())
    checks={
        'core_targets':results['core120']['targets']==120,
        'core_reactions':results['core120']['reactions']==18063,
        'core_top1':results['core120']['top1']==100,
        'core_sign':abs(results['core120']['sign']-expected['core120']['mean_sign'])<1e-10,
        'external_counts':{row['model']:(row['exact'],row['strong'],row['moderate']) for row in results['external']}==
            {name:tuple(values) for name,values in expected['external_counts'].items()},
        'all_seven_models_have_416_reactions':all(row['reactions']==416 for row in results['hydride']) and len(results['hydride'])==7,
    }
    checks['seven_model_numeric_metrics']=all(
        abs(row['reaction_mae_eV_written']-float(expected['hydride_models'][row['model']]['reaction_mae_eV_written']))<1e-10
        and abs(row['phase_mae_eV_atom']-float(expected['hydride_models'][row['model']]['phase_mae_eV_atom']))<1e-10
        and abs(row['sign']-float(expected['hydride_models'][row['model']]['sign_fraction']))<1e-10
        for row in results['hydride'])
    if args.full_analysis:
        future=json.loads((output/'temporal/summary.json').read_text())
        transferred=json.loads((output/'external_frozen_policy/invdesroute_frozen_2022_summary.json').read_text())
        checks['future_temporal_counts']=[future['future_top5'][key] for key in ['strict_exact','strong_or_exact','moderate_or_better']]==[47,49,53]
        checks['transferred_invdesroute_counts']=[transferred['metrics'][key] for key in ['strict_exact','strong_or_exact','moderate_or_better']]==[64,79,110]
    (output/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(dict(checks=checks,core120=results['core120'],external=results['external']),indent=2))
    if not all(checks.values()):
        raise SystemExit('Recomputed data disagree with frozen audited counts.')

if __name__=='__main__':
    main()
