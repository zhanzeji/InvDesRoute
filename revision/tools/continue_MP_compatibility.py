from pathlib import Path
from copy import deepcopy
import argparse
import datetime
import json
import os
import re
import numpy as np
import pandas as pd
from monty.json import MontyEncoder
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure, Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'inputs/release'
if not RELEASE.exists():RELEASE=Path('/home/zhzhan/InvDesRoute_audited_release_20261016')


def data_path(old_path):
    if Path(old_path).exists():return Path(old_path)
    manifest=pd.read_csv(RELEASE/'data/structure_manifest.csv').fillna('')
    selected=manifest[manifest.original_path.eq(old_path)] if 'original_path' in manifest else pd.DataFrame()
    if len(selected):return RELEASE/selected.iloc[0].archive_path
    raise FileNotFoundError('Resolve the archived CIF path using structure_manifest.csv: '+old_path)


def fetch_entries(output):
    from mp_api.client import MPRester
    key=os.environ.get('MP_API_KEY') or os.environ.get('PMG_MAPI_KEY')
    if not key:raise SystemExit('Set your own MP_API_KEY or PMG_MAPI_KEY; do not write it into files.')
    identifiers=set()
    for path in (RELEASE/'data/core120/targets').glob('*/phase_pool_equiformer_v3.csv'):
        table=pd.read_csv(path).fillna('')
        identifiers.update(value for value in table.source_id if re.fullmatch(r'mp-\d+',value))
    entries=[]
    with MPRester(key) as client:
        version=client.get_database_version()
        identifiers=sorted(identifiers)
        for start in range(0,len(identifiers),100):
            entries.extend(client.get_entries(identifiers[start:start+100],compatible_only=True,
                additional_criteria={'thermo_types':['GGA_GGA+U']}))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(database_version=version,thermo_type='GGA_GGA+U',
        retrieval_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        entries=[entry.as_dict() for entry in entries]),cls=MontyEncoder))
    print('Saved',len(entries),'entries, without credentials. These are current entries, not a reconstructed old snapshot.')


def compare_entries(input_path,output):
    payload=json.loads(input_path.read_text())
    if payload.get('thermo_type')!='GGA_GGA+U':raise ValueError('Only explicit GGA/GGA+U entries are supported here; r2SCAN mixing is a different calculation.')
    entries=[ComputedStructureEntry.from_dict(row) for row in payload['entries']]
    indexed={}
    for entry in entries:
        identifier=str(entry.data.get('material_id') or entry.entry_id)
        match=re.search(r'mp-\d+',identifier)
        if match:indexed.setdefault(match.group(0),[]).append(entry)
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=False)
    corrector=MaterialsProject2020Compatibility()
    rows=[]
    maps={}
    for path in sorted((RELEASE/'data/core120/targets').glob('*/phase_pool_equiformer_v3.csv')):
        phases=pd.read_csv(path).fillna('')
        maps[path.parent.name]={}
        for phase in phases.to_dict('records'):
            record=dict(target=path.parent.name,formula=phase['formula'],source_id=phase['source_id'],status='missing_entry')
            structure=None
            try:structure=Structure.from_file(data_path(phase['cif_path']))
            except Exception as error:record['status']='unresolved_CIF_path:'+type(error).__name__
            candidates=[entry for entry in indexed.get(phase['source_id'],[]) if structure is not None and matcher.fit(structure,entry.structure)]
            if len(candidates)==1:
                reference=candidates[0]
                run_type=reference.parameters.get('run_type')
                if run_type not in {'GGA','GGA+U'}:
                    record['status']='unsupported_or_missing_run_type'
                else:
                    try:
                        _,model_factor=structure.composition.get_reduced_composition_and_factor()
                        model_entry=ComputedStructureEntry(structure=structure.copy(),
                            energy=float(phase['EquiformerV3+DeNS-OAM2_energy_per_formula'])*float(model_factor),
                            parameters=deepcopy(reference.parameters),data=deepcopy(reference.data),entry_id=reference.entry_id)
                        processed=corrector.process_entry(model_entry,clean=True,inplace=False,on_error='raise')
                        if processed is None:raise ValueError('Entry rejected')
                        _,reference_factor=reference.composition.get_reduced_composition_and_factor()
                        values=dict(ML_raw=float(phase['EquiformerV3+DeNS-OAM2_energy_per_formula']),
                            ML_corrected=processed.energy/float(model_factor),
                            DFT_raw=reference.uncorrected_energy/float(reference_factor),
                            DFT_corrected=reference.energy/float(reference_factor))
                        maps[path.parent.name][phase['formula']]=values
                        record.update(values,status='ok',run_type=run_type,model_correction_per_formula=processed.correction/float(model_factor),
                            reference_correction_per_formula=reference.correction/float(reference_factor),entry_id=str(reference.entry_id))
                    except Exception as error:record['status']='compatibility_rejected:'+type(error).__name__
            elif len(candidates)>1:record['status']='ambiguous_structure_matched_entries'
            rows.append(record)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'phase_correction_audit.csv',index=False)
    results=[]
    for target,values in maps.items():
        frame=pd.read_csv(RELEASE/'data/core120/targets'/target/'formation_reactions.csv')
        for row in frame.to_dict('records'):
            species={}
            for text,sign in [(row['reactants'],-1),(row['products'],1)]:
                for token in text.split('|'):
                    number,formula=token.strip().split('*',1)
                    species[formula]=species.get(formula,0)+sign*float(number)
            record=dict(target=target,reaction=row['reaction'],status='complete' if set(species)<=set(values) else 'excluded_missing_matched_reference')
            if record['status']=='complete':
                for convention in ['ML_raw','ML_corrected','DFT_raw','DFT_corrected']:
                    record[convention+'_eV_target']=sum(number*values[formula][convention] for formula,number in species.items())/float(row['target_coefficient'])
            results.append(record)
    pd.DataFrame(results).to_csv(output/'same_convention_reactions.csv',index=False)
    summary=[]
    frame=pd.DataFrame(results)
    common=frame[frame.status=='complete']
    if len(common):
        for model,reference in [('ML_raw','DFT_raw'),('ML_corrected','DFT_corrected'),('ML_raw','DFT_corrected')]:
            for target,group in common.groupby('target'):
                left=group[model+'_eV_target'];right=group[reference+'_eV_target']
                summary.append(dict(model=model,reference=reference,target=target,reactions=len(group),mae_eV_target=float((left-right).abs().mean()),sign=float((left*right>0).mean())))
    pd.DataFrame(summary).to_csv(output/'metrics_common_subset_by_target.csv',index=False)
    print('Completed current-entry matched subset. Report exclusions and do not equate it to the historical summary endpoint.')


parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['fetch','compare'])
parser.add_argument('--entries',type=Path,required=True)
parser.add_argument('--output',type=Path,default=ROOT/'results/new_MP_compatibility')
args=parser.parse_args()
if args.action=='fetch':fetch_entries(args.entries)
else:compare_entries(args.entries,args.output)
