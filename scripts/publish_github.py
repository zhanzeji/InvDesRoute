from pathlib import Path
import argparse,json,subprocess
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--owner',required=True);args=parser.parse_args()
config=json.loads((root/'release-config.json').read_text())
if not config.get('code_license') or not (root/'LICENSE').exists():raise SystemExit('Select code_license and add the full LICENSE text before publication.')
subprocess.run(['gh','auth','status'],check=True)
if (root/'.git').exists():raise SystemExit('Existing checkout: use the standard Git update workflow; no repository changes made.')
subprocess.run(['git','config','user.name'],check=True);subprocess.run(['git','config','user.email'],check=True)
subprocess.run(['git','init','-b','main'],cwd=root,check=True)
subprocess.run(['git','add','.'],cwd=root,check=True)
subprocess.run(['git','commit','-m','Initial reproducibility release'],cwd=root,check=True)
repository=args.owner+'/'+config['repository_name']
subprocess.run(['gh','repo','create',repository,'--public','--source',str(root),'--remote','origin','--push'],cwd=root,check=True)
print('Created https://github.com/'+repository)
