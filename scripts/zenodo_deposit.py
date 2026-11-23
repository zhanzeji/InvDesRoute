from pathlib import Path
import argparse,json,os,requests
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--code-zip',type=Path,required=True);parser.add_argument('--data-zip',type=Path,required=True);args=parser.parse_args()
config=json.loads((root/'release-config.json').read_text())
if not config.get('code_license') or not config.get('data_license'):raise SystemExit('Confirm code_license and data_license in release-config.json first.')
token=os.environ.get('ZENODO_ACCESS_TOKEN')
if not token:raise SystemExit('Set your own ZENODO_ACCESS_TOKEN in the environment.')
state=root/'publish_state';state.mkdir(exist_ok=True)
session=requests.Session();session.headers['Authorization']='Bearer '+token
for kind,path,upload_type in [('code',args.code_zip,'software'),('data',args.data_zip,'dataset')]:
    marker=state/(kind+'_draft.json')
    if marker.exists():raise SystemExit('Draft already recorded; inspect publish_state instead of creating duplicates.')
    response=session.post('https://zenodo.org/api/deposit/depositions',json={},timeout=60);response.raise_for_status();record=response.json()
    marker.write_text(json.dumps(record,indent=2))
    metadata=dict(title=config['title']+' — '+kind,upload_type=upload_type,description=config['description'],creators=config['creators'],version=config['version'],access_right='open',license=config[kind+'_license'].lower(),prereserve_doi=True)
    response=session.put('https://zenodo.org/api/deposit/depositions/'+str(record['id']),json={'metadata':metadata},timeout=60);response.raise_for_status();record=response.json();marker.write_text(json.dumps(record,indent=2))
    with path.open('rb') as stream:
        response=session.put(record['links']['bucket']+'/'+path.name,data=stream,timeout=1800);response.raise_for_status()
    print(kind,'draft',record['id'],'reserved DOI',record.get('metadata',{}).get('prereserve_doi',{}).get('doi'))
print('Inspect both drafts in Zenodo and publish there after final review. No publish action was called.')
