from pathlib import Path
root=Path(__file__).resolve().parents[1]
layout=root/'revision/inputs';layout.mkdir(parents=True,exist_ok=True)
(layout/'release').mkdir(exist_ok=True)
for name in ['code','data','recomputed_full_final']:
    target=layout/'release'/name
    if not (root/name).exists():raise SystemExit('Install data first: missing '+name)
    if not target.exists():target.symlink_to('../../../'+name,target_is_directory=True)
for name,source in [('audit_results','audit_data'),('local_OUTCAR_header_excerpts','local_OUTCAR_header_excerpts')]:
    target=layout/name
    if not target.exists():target.symlink_to('../../'+source,target_is_directory=True)
(root/'revision/results').mkdir(exist_ok=True)
print('Prepared relative revision input layout')
