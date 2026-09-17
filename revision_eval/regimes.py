"""Trailing WETH volatility and position-size slices; no future regime labels."""
import argparse,json,bisect,math,collections,sys
from pathlib import Path
import numpy as np

def trailing_volatility(prices,timestamp):
 times=sorted(prices);samples=[]
 for day in range(7,-1,-1):
  t=timestamp-day*86400;i=bisect.bisect_right(times,t)-1
  if i<0 or t-times[i]>86400:return None
  samples.append(prices[times[i]])
 if min(samples)<=0:return None
 return float(np.std(np.diff(np.log(samples)),ddof=1))
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--training-cutoff',type=int,required=True);a=p.parse_args()
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Aave-Simulator'))
 from simulator.utils import get_price_history
 prices=get_price_history()['WETH'];times=sorted(prices)
 # Cut points fitted on the final pre-test year, using preceding seven days only.
 train=[trailing_volatility(prices,t) for t in range(max(times[0]+7*86400,a.training_cutoff-365*86400),a.training_cutoff,86400)];train=[v for v in train if v is not None]
 if not train:raise ValueError('No historical volatility coverage for training cut points')
 lo,hi=np.quantile(train,[1/3,2/3]);rows=json.loads((a.run/'case-metrics.json').read_text());groups={};cache={}
 for r in rows:
  t=r['checkpoint']
  if t not in cache:cache[t]=trailing_volatility(prices,t)
  v=cache[t];regime='missing' if v is None else 'low' if v<=lo else 'middle' if v<=hi else 'high'
  size='<100' if r['debt_usd']<100 else '100-1000' if r['debt_usd']<1000 else '1000-10000' if r['debt_usd']<10000 else '>=10000'
  activity=r.get('activity_30d');activity_bin='missing' if activity is None else '<=1' if activity<=1 else '2-10' if activity<=10 else '>10'
  for dimension,value in [('trailing_WETH_volatility',regime),('debt_USD',size),('past_30d_core_checkpoints',activity_bin)]:
   key=':'.join([r['cohort'],r['variant'],r['policy'],dimension,value]);c=groups.setdefault(key,collections.Counter());c['paired']+=1;c['baseline']+=r['baseline'];c['rescued']+=r['baseline'] and not r['intervention'];c['worsened']+=r['intervention'] and not r['baseline']
 result=dict(training_cutoff=a.training_cutoff,volatility_tertiles=[float(lo),float(hi)],training_days=len(train),definition='Sample SD of seven trailing daily WETH log returns; at-or-before quotes with at most one-day staleness; not independent oracle history',groups=groups)
 (a.run/'regimes.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
