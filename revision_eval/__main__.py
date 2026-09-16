import argparse,json
from pathlib import Path
from .policies import propose,POLICIES
from .metrics import horizon_metrics
from .cohorts import representative
p=argparse.ArgumentParser();p.add_argument('task',choices=['policies','metrics','cohort']);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--horizon',type=float,default=604800);p.add_argument('--budget-usd',type=float,default=1000);p.add_argument('--sample-size',type=int,default=200);a=p.parse_args()
rows=json.loads(a.input.read_text())
if a.task=='policies':result=[propose(r,policy,budget_usd=a.budget_usd) for r in rows for policy in POLICIES]
elif a.task=='metrics':result=horizon_metrics(rows,a.horizon)
else:result=representative(rows,a.sample_size)
# Avoid writing Infinity for an estimated HF after full repayment.
def clean(v):
 import math
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
with a.output.open('x') as f:json.dump(clean(result),f,indent=2,allow_nan=False)
