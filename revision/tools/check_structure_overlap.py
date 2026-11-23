from pathlib import Path
import argparse
import json
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

parser=argparse.ArgumentParser()
parser.add_argument('--reference-manifest',type=Path,required=True)
parser.add_argument('--training-manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
reference=pd.read_csv(args.reference_manifest).fillna('')
training=pd.read_csv(args.training_manifest).fillna('')
for frame in [reference,training]:
    if 'structure_file' not in frame:raise ValueError('Manifests require structure_file pointing to a real structure. Complete training coverage must be certified separately.')
matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=False)
indexed={}
for row in training.to_dict('records'):
    path=Path(row['structure_file'])
    if not path.is_absolute():path=args.training_manifest.parent/path
    structure=Structure.from_file(path)
    indexed.setdefault(structure.composition.reduced_formula,[]).append((path,structure))
rows=[]
for row in reference.to_dict('records'):
    path=Path(row['structure_file'])
    if not path.is_absolute():path=args.reference_manifest.parent/path
    structure=Structure.from_file(path)
    matches=[str(source) for source,candidate in indexed.get(structure.composition.reduced_formula,[]) if matcher.fit(structure,candidate)]
    rows.append(dict(reference_file=str(path),formula=structure.composition.reduced_formula,matched_in_supplied_training_manifest=bool(matches),matching_files=';'.join(matches)))
args.output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(args.output,index=False)
print(json.dumps(dict(references=len(rows),matched_in_supplied_manifest=sum(row['matched_in_supplied_training_manifest'] for row in rows),
    scope='Only the supplied training manifest; missing structures are not proof of non-overlap with a full dataset.'),indent=2))
