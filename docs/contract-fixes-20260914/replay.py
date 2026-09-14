import subprocess,os,json,hashlib
from pathlib import Path
b=Path(__file__).resolve().parent;f=b.parent/'focused-validation';e=f/'evidence';old=Path('/home/spadef/Aave-Action-Recommender/Aave-Simulator')
for label,sim in [('before',old),('after',b/'simulator')]:
 cmd=['python',str(sim/'tools/enriched_replay.py'),'--cohort',str(f/'cohort.json'),'--prices',str(old/'data/reserves/price_history.json'),'--evidence',str(e/'cohort-evidence.json'),'--assets',str(e/'asset-map.json'),'--aligned',str(e/'aligned-checkpoints.json'),'--variant','transfers','--order','source','--funding','profile','--revision',label+'-contract-fixes-20260914','--output',str(b/(label+'-transfers.json'))]
 env=dict(os.environ,PYTHONPATH=str(sim),MPLCONFIGDIR='/tmp/aave-contract-mpl',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 with (b/(label+'-replay.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((b/(label+'-transfers.json')).read_text());print(label,result['metrics']['strict'],flush=True)
