from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Recompute source-dependent scoring with separately held private inputs.')
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    inputs=args.inputs.resolve()
    output=args.output.resolve()
    unified=output/'unified'
    calibration=output/'calibration'
    shutil.copytree(ROOT/'analysis/unified',unified,dirs_exist_ok=True)
    shutil.copytree(ROOT/'code/workflow',unified/'workflow',dirs_exist_ok=True)
    shutil.copytree(inputs/'unified_inputs',unified/'inputs',dirs_exist_ok=True)
    calibration.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'analysis/recalculate_auc_seed80.py',calibration/'recalculate.py')
    for name in ['rerun.py','check_and_supplement.py','recompute_s5.py']:
        subprocess.run([sys.executable,str(unified/name)],check=True,cwd=unified)
    environment=dict(os.environ,INVDESROUTE_PRIVATE_INPUTS=str(inputs))
    subprocess.run([sys.executable,str(calibration/'recalculate.py')],check=True,cwd=calibration,env=environment)
    print(f'Fresh source-dependent outputs: {output}')

if __name__=='__main__':main()
