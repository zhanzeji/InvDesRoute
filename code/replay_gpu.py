from pathlib import Path
import argparse
import sys
import csv
import hashlib
import json
from ase.io import read
from pymatgen.core import Structure

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cohort',choices=['hydride','core120'],default='hydride')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args()
    checkpoint_hash=hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            checkpoint_hash.update(block)
    observed=checkpoint_hash.hexdigest()
    if observed!='429ccded98163122e7ba588d78e2441653f37f3e091e106c432807fe373c8f98':
        raise SystemExit('The checkpoint hash does not match the audited EquiformerV3 checkpoint.')
    sys.path.insert(0,str(ROOT/'code/vendor/equiformer_v3'))
    sys.path.insert(0,str(ROOT/'code/vendor/equiformer_v3/src'))
    import torch
    from fairchem.core import OCPCalculator
    import fairchem.core
    if not Path(fairchem.core.__file__).resolve().is_relative_to(ROOT/'code/vendor/equiformer_v3'):
        raise SystemExit('The imported Fair-Chem implementation is not the archived source snapshot.')
    torch.set_num_threads(4)
    calculator=OCPCalculator(checkpoint_path=str(args.checkpoint.resolve()),cpu=args.cpu,seed=0)
    manifest=list(csv.DictReader((ROOT/'data/structure_manifest.csv').open()))
    selected={row['sha256']:row for row in manifest if args.cohort in row['cohorts'].split(';')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['structure_sha256','formula','energy_cell_eV','formula_factor','energy_per_formula_eV'])
        writer.writeheader()
        for row in selected.values():
            path=ROOT/row['archive_path']
            structure=Structure.from_file(path)
            _,factor=structure.composition.get_reduced_composition_and_factor()
            atoms=read(path)
            atoms.calc=calculator
            value=float(atoms.get_potential_energy())
            writer.writerow(dict(structure_sha256=row['sha256'],formula=structure.composition.reduced_formula,
                energy_cell_eV=value,formula_factor=float(factor),energy_per_formula_eV=value/float(factor)))
            handle.flush()
    print('Replayed',len(selected),'unique structure contents.')
    args.output.with_suffix('.metadata.json').write_text(json.dumps(dict(checkpoint_sha256=observed,cohort=args.cohort,
        structures=len(selected),seed=0,source_snapshot='code/vendor/equiformer_v3',imported_module=str(fairchem.core.__file__),normalization='energy_cell / reduced_formula_factor'),indent=2))

if __name__=='__main__':
    main()
