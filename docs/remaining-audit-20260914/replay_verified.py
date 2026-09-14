import subprocess,os,json
from pathlib import Path
b=Path(__file__).resolve().parent;s=b/'simulator';f=b.parent/'focused-validation';e=f/'evidence'
cmd=['python',str(s/'tools/enriched_replay.py'),'--cohort',str(f/'cohort.json'),'--prices','/home/spadef/Aave-Action-Recommender/Aave-Simulator/data/reserves/price_history.json','--evidence',str(e/'cohort-evidence.json'),'--assets',str(e/'asset-map.json'),'--aligned',str(e/'aligned-checkpoints.json'),'--variant','transfers','--order','source','--funding','profile','--revision','remaining-audit-staged-20260914','--output',str(b/'verified-historical-transfers.json')]
with (b/'verified-historical-replay.log').open('w') as log:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH=str(s),MPLCONFIGDIR='/tmp/aave-contract-mpl',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
x=json.loads((b/'verified-historical-transfers.json').read_text());print(x['metrics']['strict'])
