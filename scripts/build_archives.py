from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT.parent)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
version=json.loads((ROOT/'release-config.json').read_text())['version']
data_names = [
    'docs/DATA_SOURCES.md',
    'recomputed_full_final/core120_by_target.csv',
    'recomputed_full_final/seven_model_recomputed.csv',
    'recomputed_full_final/external_common_endpoint_summary.csv',
    'recomputed_full_final/checks.json',
]
code_names=['code','scripts','docs','revision/tools','.github','README.md','LICENSE','LICENSE.md','CITATION.cff','CONTRIBUTING.md','CHANGELOG.md','release-config.json','requirements.txt','.gitignore']
for kind,names in [('code',code_names),('data',data_names)]:
    files=[]
    for name in names:
        path=ROOT/name
        if path.is_file():files.append(path)
        elif path.is_dir():files.extend(item for item in path.rglob('*') if item.is_file() and not item.is_symlink() and '__pycache__' not in item.parts)
    if kind == 'data' and len(files) != len(data_names):
        raise SystemExit('The processed benchmark results are required to build the public data archive.')
    manifest={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output=args.output/f'InvDesRoute-{kind}-{version}.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,arcname=path.relative_to(ROOT).as_posix())
        archive.writestr('DATA_SHA256.json' if kind=='data' else 'CODE_SHA256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    print(output)
