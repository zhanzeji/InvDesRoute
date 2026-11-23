from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'inputs/release'
if not RELEASE.exists():RELEASE=Path('/home/zhzhan/InvDesRoute_audited_release_20261016')
HOME_ROOT=Path('/home/zhzhan')
results=ROOT/'results'
reference=pd.read_csv(RELEASE/'data/hydride/input/mp_dft_phase_energies.csv')
reference['valid_reference']=pd.to_numeric(reference['energy_per_atom (eV)'],errors='coerce').notna()
reference=reference[reference.valid_reference].copy()
for column in ['DFT_code_version','functional','Hubbard_U','PAW_set','ENCUT','kpoint_density','relaxation_criteria',
               'OMat24_structure_match','MPtrj_structure_match','sAlex_structure_match']:
    reference[column]='not_established_from_archived_reference_table'
reference.to_csv(results/'TBD08_hydride_reference_provenance_gaps.csv',index=False)
future=pd.read_csv(RELEASE/'data/temporal/split/future_routes_2023_2024.csv')
future[['target','target_mp_id','doi','publication_year']].assign(MP_entry_creation_date='not_archived',
    created_after_first_publication='unresolved_without_entry_creation_metadata').to_csv(results/'TBD12_MP_entry_age_unresolved.csv',index=False)
phase_metadata=[]
for path in (RELEASE/'data/core120/targets').glob('*/phase_pool_raw.csv'):
    columns=set(pd.read_csv(path,nrows=0).columns)
    phase_metadata.append(dict(target=path.parent.name,records=len(pd.read_csv(path)),database_version_saved='database_version' in columns,
        query_date_saved='retrieval_date' in columns,thermo_type_saved='thermo_type' in columns,
        run_type_saved='run_type' in columns,energy_adjustments_saved='energy_adjustments' in columns))
pd.DataFrame(phase_metadata).to_csv(results/'TBD01_04_MP_archival_metadata_coverage.csv',index=False)
outcars=[]
outcar_root=ROOT/'inputs/local_OUTCAR_header_excerpts'
if not outcar_root.exists():outcar_root=HOME_ROOT/'LiAuH/dataset'
for path in sorted(outcar_root.glob('*/OUTCAR')):
    evidence={}
    with path.open(errors='replace') as stream:
        for number,line in enumerate(stream):
            if number>8000:break
            if number==0:evidence['code_banner']=line.strip()
            for name,pattern in [('ENCUT',r'ENCUT\s*=\s*([\d.]+)'),('EDIFF',r'EDIFF\s*=\s*([\d.Ee+\-]+)'),
                ('GGA',r'GGA\s*=\s*(\S+)'),('LDAU',r'LDAU\s*=\s*(\S+)'),('NKPTS',r'NKPTS\s*=\s*(\d+)')]:
                match=re.search(pattern,line)
                if match and name not in evidence:evidence[name]=match.group(1)
    outcars.append(dict(path=str(path),**evidence,scope='Local LiAuH calculation only; not mapped to all 107 MP-sourced hydride reference phases.'))
pd.DataFrame(outcars).to_csv(results/'TBD08_available_local_OUTCAR_headers.csv',index=False)
corpus=RELEASE/'data/temporal/split/historical_prior_routes_through_2020.csv'
prior=pd.read_csv(corpus,low_memory=False).fillna('')
reference_rows=[]
for cohort,name in [('59','ceder_seed80_real_literature_routes.csv'),('86','selected_literature_routes.csv')]:
    frame=pd.read_csv(RELEASE/'data/literature/references'/name).fillna('')
    for row in frame.to_dict('records'):
        doi=str(row.get('doi','')).strip().lower()
        if not doi:
            match=re.search(r'DOI=(10\.[^;\s]+)',str(row.get('source_note','')))
            if match:doi=match.group(1).rstrip('.').lower()
        reference_rows.append(dict(cohort=cohort,target=row['target'],doi=doi,
            same_doi_in_historical_corpus=doi in set(prior.doi.str.lower()),
            scoring_protocol='Archived fixed heuristic base score plus family-first display, not a corpus-trained DOI-held-out test.'))
pd.DataFrame(reference_rows).to_csv(results/'TBD11_retrospective_DOI_overlap.csv',index=False)
root=RELEASE/'data/literature'
protocol=[]
for cohort,folder in [('59','seed80_targets'),('86','extension53_targets')]:
    for path in (root/folder).glob('*/synthesizability_ranked_routes.csv'):
        frame=pd.read_csv(path,nrows=1)
        protocol.append(dict(cohort=cohort,target=path.parent.name,
            historical_corpus_score_present='historical_corpus_score' in frame,
            corpus_precursor_prior_present='corpus_precursor_prior' in frame,
            optimized_rank_score_present='optimized_rank_score' in frame))
pd.DataFrame(protocol).to_csv(results/'TBD11_archived_score_columns.csv',index=False)
summary=dict(valid_hydride_reference_rows=len(reference),target_age_unknown=len(future),
    archived_MP_database_version_recoverable=False,all_107_DFT_settings_certified=False,
    training_structure_overlap_quantified=False,public_archive_identifiers_issued=False,
    candidate_creation_timestamp_distinct_from_file_mtime=True,
    limitations_are_not_replaced_by_fabricated_zero_counts=True)
(results/'unresolved_evidence_summary.json').write_text(json.dumps(summary,indent=2))
weights={'energy_score':.18,'precursor_prior_score':.25,'route_template_score':.25,
    'practical_filter_score':.12,'route_complexity_score':.08,'phase_availability_score':.07,'condition_simplicity_score':.05}
reconstructed=[]
for folder in ['seed80_targets','extension53_targets']:
    for path in (RELEASE/'data/literature'/folder).glob('*/synthesizability_ranked_routes.csv'):
        frame=pd.read_csv(path)
        base=sum(frame[name]*value for name,value in weights.items())
        bonus=np.select([frame.route_template_score.ge(.95),frame.route_template_score.ge(.90)],[.06,.03],default=0)
        computed=(base+bonus).clip(0,1)
        reconstructed.append(dict(cohort=folder,target=path.parent.name,candidates=len(frame),
            max_difference=float((computed-frame.synth_score).abs().max()),
            mismatches=int(((computed-frame.synth_score).abs()>1e-8).sum())))
pd.DataFrame(reconstructed).to_csv(results/'TBD11_archived_base_weight_reconstruction.csv',index=False)
print(json.dumps(summary,indent=2))
