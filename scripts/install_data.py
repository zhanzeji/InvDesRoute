from pathlib import Path
import argparse,zipfile,hashlib,json
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);args=parser.parse_args()
with zipfile.ZipFile(args.archive) as archive:
    manifest=json.loads(archive.read('DATA_SHA256.json'))
    for entry in archive.infolist():
        target=(root/entry.filename).resolve()
        if not target.is_relative_to(root):raise SystemExit('Unsafe archive path')
        if entry.filename not in {'DATA_SHA256.json','README.md'} and target.exists():raise SystemExit('Refusing overwrite: '+str(target))
    for name,digest in manifest.items():
        if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise SystemExit('Checksum mismatch: '+name)
    for entry in archive.infolist():
        if entry.filename=='README.md':continue
        archive.extract(entry,root)
print('Installed and verified',len(manifest),'data files')
